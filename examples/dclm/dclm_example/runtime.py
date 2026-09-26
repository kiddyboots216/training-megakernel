"""WORLD8 DCLM training with circular refill plus in-launch checkpointing.

``checkpoint`` runs advancing windows in one application CUfunction and
publishes durable checkpoints under ``ROOT/step-NNNNNNNN`` while that
CUfunction is parked.  ``resume`` restores the latest complete checkpoint under
the same root and runs the following windows in one new application
CUfunction, publishing its checkpoints into that root.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from itertools import chain, pairwise
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from training_megakernel import contract
from training_megakernel import resident_protocol as P
from training_megakernel.checkpoint_protocol import compact_stored_surfaces
from training_megakernel.circular_refill_runtime import (
    CircularTokenRefillSession,
    gloo_generation_gate,
)
from training_megakernel.geometry import Geometry
from training_megakernel.nstep import DEFAULT_RING_SLOTS
from training_megakernel.runtime import ApplicationState, load_image, read_bundle
from training_megakernel.shape import reserved_context_bytes

from .checkpoint_history import latest_checkpoint
from .checkpoint_service import (
    CheckpointRankMailbox,
    CheckpointService,
    create_checkpoint_control_group,
)
from .checkpoint_storage import load_checkpoint_manifest, load_raw_checkpoint
from .corpus import DclmStreamCursors, load_stream_cursors
from .model_loader import HFSafetensorSource, load_hf_rank_into_optimizer
from .system import (
    clear_gpu_cache,
    initialize_world8,
    set_cuda_stack_limit,
)
from .workload_store import ShardedWorkloadSource, load_shard_manifest

# The data cursor a checkpoint stores.
TRAINING_CURSOR_SCHEMA = "training_megakernel_training_cursor_v1"
DCLM_SOURCE_CURSOR_FIELD = "dclm_source_cursor"
# How long the device, and the refill producer, wait for a peer or a token slot.
DEVICE_TIMEOUT_MS = 300_000
# How long checkpoint publication waits for a device chunk or for its peers.
CHECKPOINT_TIMEOUT_SECONDS = 1_800.0
TELEMETRY_POLL_MS = 20
TELEMETRY_LOGGER_TIMEOUT_SECONDS = 120.0


def _optimizer_control(args: argparse.Namespace) -> tuple[float, float] | None:
    """Validate an optional paired optimizer override for one invocation."""

    if args.learning_rate is None and args.weight_decay is None:
        return None
    if args.learning_rate is None or args.weight_decay is None:
        raise ValueError("learning rate and weight decay must be supplied together")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("learning rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or not 0.0 <= args.weight_decay <= 1.0:
        raise ValueError("weight decay must be finite and in [0, 1]")
    return args.learning_rate, args.weight_decay


def _preflight_hf_model_source(
    args: argparse.Namespace, *, depth: int
) -> HFSafetensorSource | None:
    """Validate a local snapshot of the bundle's depth before any state allocation."""

    snapshot = args.hf_model_snapshot
    if args.arm != "checkpoint":
        if snapshot is not None:
            raise ValueError("--hf-model-snapshot is valid only for the fresh checkpoint arm")
        return None
    if snapshot is None:
        raise ValueError(
            "the fresh checkpoint arm trains from --hf-model-snapshot: a pretrained "
            "snapshot or one written by random_init.py"
        )
    return HFSafetensorSource(snapshot, depth=depth)


def _parse_checkpoint_steps(values: list[str] | None) -> tuple[int, ...] | None:
    """Parse repeated and/or comma-separated global optimizer steps."""

    if not values:
        return None
    pieces: list[str] = []
    for value in values:
        pieces.extend(value.split(","))
    try:
        steps = tuple(int(piece.strip()) for piece in pieces)
    except ValueError as error:
        raise ValueError("checkpoint steps must be comma-separated integers") from error
    if any(current >= following for current, following in pairwise(steps)):
        raise ValueError("checkpoint steps must be strictly increasing")
    return steps


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _validate_resume_manifest(
    manifest: dict[str, Any],
    *,
    stream_cursors: DclmStreamCursors,
) -> dict[str, Any]:
    """Accept a checkpoint whose data cursor continues this shard set.

    Loading the manifest already required its stored tensors to have this
    bundle's names, dtypes and shapes, so it was written at the same depth.
    """

    cursor = manifest["data_cursor"]
    next_stream = cursor.get("next_stream")
    # The DCLM position the checkpoint stopped at must be the position this
    # shard set records for the same stream.
    if not (
        isinstance(next_stream, int)
        and stream_cursors.covers(next_stream)
        and cursor.get(DCLM_SOURCE_CURSOR_FIELD) == stream_cursors.at(next_stream)
    ):
        raise RuntimeError(
            f"checkpoint stopped at stream {next_stream!r}, which this shard set does not "
            "continue from the same DCLM position"
        )
    return cursor


def _check_memory(geometry: Geometry, device: torch.device, *, rank: int) -> None:
    """Stop before allocation when the state cannot fit this GPU."""

    free_bytes, _total = torch.cuda.mem_get_info(device)
    estimate = geometry.shape.memory_estimate_bytes()
    needed = sum(estimate.values())
    gib = 2**30
    if rank == 0:
        print(
            f"estimated memory per GPU for {geometry.describe()} "
            f"(plus about {reserved_context_bytes() / gib:.1f} GiB of CUDA context "
            "and communication, already in use):",
            flush=True,
        )
        for name, nbytes in estimate.items():
            print(f"  {name:<24} {nbytes / gib:6.2f} GiB", flush=True)
        print(f"  {'total':<24} {needed / gib:6.2f} GiB", flush=True)
    if needed > free_bytes:
        raise RuntimeError(
            f"{geometry.describe()} needs about {needed / gib:.1f} GiB per GPU for its "
            f"training state, more than the {free_bytes / gib:.1f} GiB free on rank {rank}"
        )


def _rank_result(
    state: ApplicationState, *, rank: int, start_step: int, steps: int, elapsed_seconds: float
) -> dict[str, Any]:
    """This rank's step records and the device's status words and exit sentinel."""

    records = state.nstep.records.detach().view(steps, P.RECORD_WIDTH).cpu().tolist()
    control = state.nstep.control.cpu().tolist()
    completed = start_step + steps
    status = {
        "optimizer": state.optimizer.status.cpu().tolist(),
        "fabric": state.fabric.status.cpu().tolist(),
        "shell": state.full_shell.status.cpu().tolist(),
        "embedding_route": state.full_shell.embedding_route_status.cpu().tolist(),
    }
    checks = [
        [int(row[3]) for row in records] == list(range(start_step + 1, completed + 1)),
        all(math.isfinite(value) for row in records for value in row[:3]),
        int(state.optimizer.completed_steps.item()) == completed,
        status["optimizer"] == [0, 0, 0, completed],
        not any(status["fabric"] + status["shell"] + status["embedding_route"]),
        control[P.CONTROL_STEP] == steps,
        control[P.CONTROL_CHECKPOINT] == P.EXIT_SENTINEL,
    ]
    return {
        "rank": rank,
        "pass": all(checks),
        "records": records,
        "status": status,
        "exit_sentinel": control[P.CONTROL_CHECKPOINT] == P.EXIT_SENTINEL,
        "elapsed_seconds": elapsed_seconds,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    optimizer_control = _optimizer_control(args)
    # The bundle's depth and tokens per GPU size everything below: the model
    # snapshot, the shards, the checkpoints it may resume and the state.
    _manifest, launch = read_bundle(args.bundle)
    geometry = Geometry.from_launch_abi(launch)
    shard_manifest = load_shard_manifest(
        args.workload,
        sequence=geometry.sequence,
        embedding_route_records=geometry.route_records,
    )
    stream_cursors = load_stream_cursors(args.workload, shard_manifest)
    if not 1 <= args.steps <= geometry.resident_step_limit:
        raise ValueError(
            f"--steps must be in [1, {geometry.resident_step_limit}] for this bundle, "
            f"got {args.steps}"
        )
    hf_model_source = _preflight_hf_model_source(args, depth=geometry.depth)
    checkpoint_root = args.checkpoint_root
    stored_surfaces = compact_stored_surfaces(geometry.optimizer_elements)
    resume_directory: Path | None = None
    if args.arm == "resume":
        start_step, resume_directory = latest_checkpoint(checkpoint_root)
        resume_manifest = load_checkpoint_manifest(
            resume_directory,
            step=start_step,
            surfaces=stored_surfaces,
        )
        start_stream = int(
            _validate_resume_manifest(resume_manifest, stream_cursors=stream_cursors)[
                "next_stream"
            ]
        )
    else:
        if checkpoint_root.exists():
            raise FileExistsError(f"checkpoint root already exists: {checkpoint_root}")
        start_step = start_stream = 0
    final_step = start_step + args.steps
    invocation_cursor = {
        "schema": TRAINING_CURSOR_SCHEMA,
        "next_stream": start_stream,
        "completed_steps": start_step,
        DCLM_SOURCE_CURSOR_FIELD: dict(stream_cursors.at(start_stream)),
    }
    checkpoint_steps = _parse_checkpoint_steps(args.checkpoint_steps) or (final_step,)
    for checkpoint_step in checkpoint_steps:
        if not start_step < checkpoint_step <= final_step:
            raise ValueError(
                f"checkpoint step {checkpoint_step} is outside this run's steps "
                f"({start_step}, {final_step}]"
            )
    args.output.mkdir(parents=True, exist_ok=True)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    session: CircularTokenRefillSession | None = None
    mailbox: CheckpointRankMailbox | None = None
    service: CheckpointService | None = None
    try:
        rank, refill_group = initialize_world8(device=device)
        checkpoint_group = create_checkpoint_control_group(
            timeout_seconds=CHECKPOINT_TIMEOUT_SECONDS
        )
        set_cuda_stack_limit(launch["required_cuda_stack_limit_bytes"])

        source = ShardedWorkloadSource.open(args.workload, shard_manifest, rank=rank)
        windows = source.iter_windows(start_stream=start_stream, count=args.steps)
        first_workload = next(windows)

        clear_gpu_cache(device)
        _check_memory(geometry, device, rank=rank)
        state = ApplicationState.allocate(
            geometry=geometry,
            rank=rank,
            device=device,
            timeout_ms=DEVICE_TIMEOUT_MS,
            workload=first_workload,
        )
        if hf_model_source is not None:
            load_hf_rank_into_optimizer(state.optimizer, hf_model_source, rank=rank)
        mailbox = CheckpointRankMailbox(
            run_id=args.run_id, rank=rank, ring_slots=DEFAULT_RING_SLOTS
        )
        mailbox.register_cuda(device_index=local_rank)
        header = mailbox.checkpoint_header

        def checkpoint_progress() -> tuple[int, int]:
            # A checkpoint holds the token slots while its chunks advance.
            return (
                header.load("device_ready_generation"),
                header.load("host_ack_generation"),
            )

        session = CircularTokenRefillSession(
            workloads=chain((first_workload,), windows),
            workload_count=args.steps,
            device=device,
            run_id=f"{args.run_id}-refill-rank{rank}",
            geometry=geometry,
            base_generation=start_stream + 1,
            timeout_ns=DEVICE_TIMEOUT_MS * 1_000_000,
            generation_gate=gloo_generation_gate(refill_group),
            mailbox_address=mailbox.device_address,
            telemetry_ring_slots=mailbox.ring_slots,
            progress=checkpoint_progress,
        )
        del first_workload
        session.bind(state)

        if resume_directory is not None:
            load_raw_checkpoint(
                state,
                resume_directory,
                rank=rank,
                completed_steps=start_step,
                timeout_ns=DEVICE_TIMEOUT_MS * 1_000_000,
            )
        if optimizer_control is not None:
            # On resume this replaces the restored values: a run may change its
            # learning rate or weight decay at a checkpoint.
            with torch.no_grad():
                state.optimizer.hyperparameters[0] = optimizer_control[0]
                state.optimizer.hyperparameters[4] = optimizer_control[1]
        clear_gpu_cache(device)
        image = load_image(args.bundle, state)

        service = CheckpointService(
            mailbox=mailbox,
            root=checkpoint_root,
            rank=rank,
            world_size=contract.WORLD_SIZE,
            steps=checkpoint_steps,
            invocation_cursor=invocation_cursor,
            surfaces=stored_surfaces,
            source_cursor_resolver=stream_cursors.at,
            control_group=checkpoint_group,
            timeout_seconds=CHECKPOINT_TIMEOUT_SECONDS,
        ).start()

        if rank == 0:
            mailbox.start_logger(
                resident_steps=args.steps,
                start_completed_step=start_step,
                output_dir=args.output / "telemetry",
                poll_ms=TELEMETRY_POLL_MS,
                wandb_mode=args.wandb_mode,
                wandb_project=args.wandb_project,
                loss_shm_paths=[
                    mailbox.path_for(f"{args.run_id}-rank{peer}")
                    for peer in range(contract.WORLD_SIZE)
                ],
            )
            mailbox.wait_logger_ready()

        torch.cuda.synchronize(device)
        dist.barrier(group=refill_group)
        session.start()
        dist.barrier(group=refill_group)
        started = time.perf_counter()
        image(state)
        torch.cuda.synchronize(device)
        elapsed_seconds = time.perf_counter() - started

        if rank == 0:
            mailbox.mark_final_and_wait(
                final_step, timeout_seconds=TELEMETRY_LOGGER_TIMEOUT_SECONDS
            )
        manifests = service.wait(timeout_seconds=CHECKPOINT_TIMEOUT_SECONDS)
        session.finish()
        local = _rank_result(
            state,
            rank=rank,
            start_step=start_step,
            steps=args.steps,
            elapsed_seconds=elapsed_seconds,
        )
        gathered: list[dict[str, Any] | None] = [None] * contract.WORLD_SIZE
        dist.all_gather_object(gathered, local, group=refill_group)
        rows = [row for row in gathered if row is not None]
        result = {
            "pass": all(row["pass"] for row in rows),
            "arm": args.arm,
            "shape": {"depth": geometry.depth, "sequence": geometry.sequence},
            "step_range": [start_step + 1, final_step],
            # Each rank's loss is its share of the WORLD8 mean; the norm is global.
            "steps": [
                {
                    "step": int(rows[0]["records"][index][3]),
                    "loss": math.fsum(row["records"][index][0] for row in rows),
                    "global_norm": rows[0]["records"][index][1],
                    "clip": rows[0]["records"][index][2],
                }
                for index in range(args.steps)
            ],
            "checkpoint_steps": [manifest["completed_steps"] for manifest in manifests],
            "restored_checkpoint": None if resume_directory is None else str(resume_directory),
            "elapsed_seconds": max(row["elapsed_seconds"] for row in rows),
            "ranks": [
                {name: row[name] for name in ("rank", "pass", "status", "exit_sentinel")}
                for row in rows
            ],
        }
        if rank == 0:
            _write(args.output / "result.json", result)
            print(
                json.dumps(
                    {
                        "pass": result["pass"],
                        "arm": result["arm"],
                        "step_range": result["step_range"],
                        "checkpoint_steps": result["checkpoint_steps"],
                        "elapsed_seconds": result["elapsed_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        dist.barrier(group=refill_group)
        if not result["pass"]:
            raise RuntimeError("one or more ranks failed their end-of-run checks")
        return result
    except BaseException:
        if service is not None:
            service.abort()
        if session is not None:
            session.abort()
        raise
    finally:
        service_closed = service is None or service.close()
        if mailbox is not None and service_closed:
            mailbox.close()
        if session is not None:
            session.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--arm", choices=("checkpoint", "resume"), required=True)
    result.add_argument("--bundle", type=Path, required=True)
    result.add_argument(
        "--output", type=Path, required=True, help="run directory; result.json goes here"
    )
    result.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help=(
            "directory of step-NNNNNNNN checkpoints; resume restores the latest "
            "complete one, and both arms publish their checkpoints here"
        ),
    )
    result.add_argument(
        "--hf-model-snapshot",
        type=Path,
        help=(
            "local Qwen3-8B-width safetensors snapshot with the bundle's depth, "
            "for a fresh step-zero checkpoint arm; resume restores model state "
            "from its checkpoint"
        ),
    )
    result.add_argument("--workload", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument(
        "--learning-rate",
        type=float,
        help=(
            "constant AdamW learning rate for this resident invocation, also on "
            "resume; requires --weight-decay"
        ),
    )
    result.add_argument(
        "--weight-decay",
        type=float,
        help=(
            "constant AdamW weight decay for this resident invocation, also on "
            "resume; requires --learning-rate"
        ),
    )
    result.add_argument("--steps", type=int, required=True)
    result.add_argument(
        "--checkpoint-steps",
        action="append",
        help=(
            "ordered global checkpoint steps; may be repeated or comma-separated. "
            "The default is the invocation's last step"
        ),
    )
    result.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
    )
    result.add_argument("--wandb-project")
    return result


def main(argv: list[str] | None = None) -> None:
    """Run the packaged WORLD8 rolling-checkpoint entry point."""

    run(parser().parse_args(argv))


if __name__ == "__main__":
    main()
