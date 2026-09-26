#!/usr/bin/env python3
"""Build the training megakernel for a model of Qwen3-8B's width.

    python kernel/build.py --depth D --sequence S --out DIR

D is the number of decoder layers (2 to 82) and S the tokens per GPU per step (a multiple of
1,024 from 1,024 to 32,768); the defaults are the released 36 and 32,768. The build compiles the
program's one cooperative CUfunction on one GPU, rewrites the step's predicated exit into the
branch back to the kernel's first instruction that makes the step loop, and writes a new release
directory, outside the repository, holding three files: ``image.o``, ``launch_abi.json`` and
``manifest.json``. Run it on one H100 with CUDA_VISIBLE_DEVICES selecting the GPU; the compile
allocates the full per-GPU training state there, because it reads pointer alignment from real
device allocations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

KERNEL = Path(__file__).resolve().parent
PROGRAM = KERNEL / "program"
TOOLS = KERNEL / "tools"

# The per-thread CUDA stack the runtime sets; the kernel's static stack must fit.
STACK_LIMIT_BYTES = 2048
# The most steps one launch may run.
STEP_LIMIT = 2**31 - 1
# Steps the compile's sample step-loop operands hold; the launch ABI records the number.
COMPILE_SAMPLE_STEPS = 2
FATBIN_MAGIC = bytes.fromhex("50ed55ba")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True, help="new release directory")
    parser.add_argument("--depth", type=int, default=36, help="decoder layers, 2 to 82")
    parser.add_argument(
        "--sequence",
        type=int,
        default=32_768,
        help="tokens per GPU per step, a multiple of 1,024 from 1,024 to 32,768",
    )
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    if KERNEL.parent in args.out.parents:
        parser.error("build outside the repository so binaries do not enter the source tree")
    from training_megakernel.shape import Shape

    try:
        args.shape = Shape(args.depth, args.sequence)
    except ValueError as error:
        parser.error(str(error))
    return args


def synthetic_workload(sequence: int):
    """Token ids and packed segments for the compile: three documents and a 64-row pad.

    Returns the four segment lengths, the token ids and the valid-token count. The compile reads
    only shapes from them: the segment count is compiled into argument shapes, and the runtime
    installs each run's segment lengths at launch. Token ids stay below 4,096 so no embedding
    owner receives more distinct ids than the route capacity.
    """

    import torch

    pad = 64
    documents = (sequence - pad) // 3
    lengths = (sequence - pad - 2 * documents, documents, documents, pad)
    ids = torch.arange(sequence, dtype=torch.int32) % 4096
    return lengths, ids, sequence - pad


def extract_device_images(object_path: Path, fatbin_path: Path, cubin_path: Path) -> None:
    """Write the fatbin the exported object embeds (its ``kernels_binary`` symbol) and its cubin.

    The fatbin must hold a single cubin, the sm_90a image.
    """

    symbols = subprocess.run(
        ["nm", "-a", "-S", str(object_path)], check=True, capture_output=True, text=True
    ).stdout
    match = re.search(
        r"^([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+[rR]\s+kernels_binary$", symbols, re.MULTILINE
    )
    if match is None:
        raise RuntimeError("exported object has no kernels_binary symbol")
    offset, size = int(match.group(1), 16), int(match.group(2), 16)
    section = object_path.with_suffix(".lrodata")
    scratch = object_path.with_suffix(".scratch.o")
    subprocess.run(
        ["objcopy", "--dump-section", f".lrodata={section}", str(object_path), str(scratch)],
        check=True,
        capture_output=True,
    )
    fatbin = section.read_bytes()[offset : offset + size]
    section.unlink()
    scratch.unlink()
    if len(fatbin) != size or fatbin[:4] != FATBIN_MAGIC:
        raise RuntimeError("kernels_binary is not a complete CUDA fatbin")
    fatbin_path.write_bytes(fatbin)
    extract = object_path.parent / "elf"
    extract.mkdir()
    subprocess.run(
        ["cuobjdump", "--extract-elf", "all", str(fatbin_path)],
        cwd=extract,
        check=True,
        capture_output=True,
    )
    cubins = sorted(extract.glob("*.cubin"))
    if len(cubins) != 1:
        raise RuntimeError(f"expected one sm_90a cubin in the fatbin, found {cubins}")
    shutil.move(cubins[0], cubin_path)
    shutil.rmtree(extract)


def static_stack_bytes(cubin_path: Path) -> int:
    """The kernel's static stack per thread, from ``cuobjdump -res-usage``."""

    usage = subprocess.run(
        ["cuobjdump", "-res-usage", str(cubin_path)], check=True, capture_output=True, text=True
    ).stdout
    values = {int(value) for value in re.findall(r"STACK:(\d+)", usage)}
    if len(values) != 1:
        raise RuntimeError(f"expected one kernel stack size, found {sorted(values)}")
    return values.pop()


def register_split_present(cubin_path: Path) -> bool:
    """Whether the SASS has ``USETMAXREG``, that is, whether ptxas kept the register split."""

    sass = subprocess.run(
        ["cuobjdump", "-sass", str(cubin_path)], check=True, capture_output=True, text=True
    ).stdout
    return "USETMAXREG" in sass


def file_record(path: Path) -> dict[str, object]:
    """Name, size and SHA-256 of one release file, for the manifest."""

    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def check_memory(shape, device) -> None:
    """Stop before allocating when the shape cannot fit one GPU at run time."""

    import torch

    from training_megakernel.shape import reserved_context_bytes

    gib = 2**30
    estimate = shape.memory_estimate_bytes()
    needed = sum(estimate.values()) + reserved_context_bytes()
    _free, capacity = torch.cuda.mem_get_info(device)
    print(f"estimated memory per GPU for {shape.depth} layers and {shape.sequence} tokens:")
    for name, nbytes in estimate.items():
        print(f"  {name:<24} {nbytes / gib:6.2f} GiB")
    print(f"  {'context and comms':<24} {reserved_context_bytes() / gib:6.2f} GiB")
    print(f"  {'total':<24} {needed / gib:6.2f} GiB of {capacity / gib:.2f} GiB", flush=True)
    if needed > capacity:
        raise SystemExit(
            f"{shape.depth} layers and {shape.sequence} tokens per GPU need about "
            f"{needed / gib:.1f} GiB per GPU, more than the {capacity / gib:.1f} GiB this GPU has"
        )


def main() -> int:
    args = parse_args()
    shape = args.shape
    # The program reads its shape when it is imported (program/model.py).
    os.environ["TMK_DEPTH"] = str(shape.depth)
    os.environ["TMK_SEQUENCE"] = str(shape.sequence)
    # The program's modules, and patch_step_backedge, are imported as top-level modules.
    sys.path[:0] = [str(PROGRAM), str(TOOLS)]

    import model
    import torch
    import training_program
    from communication import embedding_route, shell_fabric
    from communication.decoder_fabric import attach_decoder_fabric
    from gemm_members import PROGRAM_SMEM_PAGE_BYTES, SMEM_OUTSIDE_PAGE_BYTES
    from optimizer_state import allocate_optimizer_state
    from patch_step_backedge import patch_backedge
    from training_tensors import TrainingTensors

    from training_megakernel import resident_protocol as RP
    from training_megakernel import runtime as R
    from training_megakernel.abi import runtime_contract

    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if properties.multi_processor_count != model.PROGRAM_CTAS or properties.major != 9:
        raise SystemExit(f"the program targets an H100 with {model.PROGRAM_CTAS} SMs")
    check_memory(shape, device)

    lengths, input_ids, local_valid = synthetic_workload(model.SEQUENCE)
    tensors = TrainingTensors.allocate(lengths, device, logical_depth=model.DEPTH)
    fabric = attach_decoder_fabric(tensors, device=device)
    optimizer = allocate_optimizer_state(tensors, fabric, device=device)
    shell = shell_fabric.allocate_shell_fabric(
        tensors,
        input_ids=input_ids.to(device),
        local_valid_tokens=local_valid,
        device=device,
    )
    torch.cuda.empty_cache()
    training_program.allocate_checkpoint_banks(tensors.shell, device)
    training_program.store_residual_mid_slot_addresses(tensors)

    compiled = training_program.compile_training_program(
        tensors,
        decoder_fabric=fabric,
        optimizer_tail=optimizer,
        full_shell=shell,
        nstep=training_program.allocate_step_loop_operands(device, COMPILE_SAMPLE_STEPS),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{args.out.name}.", dir=args.out.parent) as work:
        work = Path(work)
        object_path = work / "image.o"
        compiled.function.export_to_c(
            object_file_path=str(object_path), function_name=R.FUNCTION_NAME
        )
        cubin_path = work / "image.cubin"
        fatbin_path = work / "image.fatbin"
        extract_device_images(object_path, fatbin_path, cubin_path)
        stack = static_stack_bytes(cubin_path)
        if stack > STACK_LIMIT_BYTES:
            raise SystemExit(
                f"the kernel needs {stack} bytes of stack per thread, above the "
                f"{STACK_LIMIT_BYTES} the runtime sets"
            )
        if not register_split_present(cubin_path):
            print(
                "warning: no USETMAXREG instruction: ptxas dropped the register split, "
                "so the kernel will run far slower",
                file=sys.stderr,
            )
        patch_backedge(work)

        layout = embedding_route.ROUTE_LAYOUT
        launch = {
            "schema": R.LAUNCH_ABI_SCHEMA,
            "function_name": R.FUNCTION_NAME,
            "grid": [model.PROGRAM_CTAS, 1, 1],
            "block": [model.PROGRAM_THREADS, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": True,
            "use_pdl": True,
            "min_blocks_per_mp": 1,
            "dynamic_shared_memory_bytes": (
                PROGRAM_SMEM_PAGE_BYTES + SMEM_OUTSIDE_PAGE_BYTES
            ),
            "environment_stream": True,
            "required_cuda_stack_limit_bytes": STACK_LIMIT_BYTES,
            "runtime_abi": runtime_contract(
                prefix=compiled.runtime_prefix,
                suffix=compiled.runtime_suffix,
                shell=compiled.runtime_shell,
                fabric=compiled.runtime_fabric,
                optimizer=compiled.runtime_optimizer,
                full_shell=compiled.runtime_full_shell,
                nstep=compiled.runtime_nstep,
            ),
            "embedding_route_records_per_source_owner": layout.records_per_source_owner,
            "embedding_route_compact_owner_gradient": layout.compact_owner_gradient,
            "resident_step_limit": STEP_LIMIT,
            "resident_compile_sample_steps": COMPILE_SAMPLE_STEPS,
            "checkpoint_stored_surfaces": list(R.COMPACT_CHECKPOINT_STORED_SURFACES),
            "checkpoint_chunk_bytes": RP.CHECKPOINT_CHUNK_BYTES,
            "checkpoint_mapped_payload_slots": RP.CHECKPOINT_PAYLOAD_SLOTS,
            "logical_depth": shape.depth,
            "sequence": shape.sequence,
        }
        launch_path = work / "launch_abi.json"
        launch_path.write_text(json.dumps(launch, indent=2, sort_keys=True) + "\n")

        release = work / "release"
        release.mkdir()
        shutil.copyfile(object_path, release / "image.o")
        shutil.copyfile(launch_path, release / "launch_abi.json")
        manifest = {
            "schema": R.COMPILED_RELEASE_SCHEMA,
            "function_name": R.FUNCTION_NAME,
            "files": [file_record(release / "image.o"), file_record(release / "launch_abi.json")],
        }
        (release / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        release.rename(args.out)

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
