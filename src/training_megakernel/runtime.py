"""Allocate, load, and invoke an externally distributed resident image."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from . import abi, arenas
from . import resident_protocol as P
from .circular_refill_runtime import CircularNstepRuntimeState
from .distributed import (
    HEAD_FORWARD_EPOCHS,
    NORM_PARTIAL_ELEMENTS,
    SCHEDULER_WORDS,
    DecoderFabricState,
    FullShellState,
    OptimizerState,
    SymmetricArena,
    fabric_table_entries,
)
from .geometry import Geometry
from .layout import MAX_DOCS
from .shards import TokenShardRing
from .state import HIDDEN, ModelState
from .workload import TokenWindow

FUNCTION_NAME = "training_megakernel"
# The only supported bundle format: image.o, launch_abi.json, and a manifest
# listing the size and SHA-256 of those two files.
COMPILED_RELEASE_SCHEMA = "training_megakernel_release_v1"
LAUNCH_ABI_SCHEMA = "training_megakernel_launch_abi_v1"
CHECKPOINT_SCHEMA = "training_megakernel_checkpoint_v1"
COMPACT_CHECKPOINT_STORED_SURFACES = [
    "parameter",
    "exp_avg",
    "exp_avg_sq",
    "final_parameter",
    "final_exp_avg",
    "final_exp_avg_sq",
    "hyperparameters",
]
BUNDLE_FILES = ("image.o", "launch_abi.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bundle(path: str | Path) -> tuple[dict, dict]:
    """Return a release bundle's manifest and launch ABI, after checking both files' SHA-256.

    The launch ABI must also describe the checkpoint chunks the host drains.
    """

    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != COMPILED_RELEASE_SCHEMA:
        raise RuntimeError(f"{path} is not a {COMPILED_RELEASE_SCHEMA} release bundle")
    records = {record["name"]: record for record in manifest["files"]}
    if sorted(records) != sorted(BUNDLE_FILES):
        raise RuntimeError(f"bundle manifest lists {sorted(records)}, not {list(BUNDLE_FILES)}")
    for name, record in records.items():
        artifact = path / name
        if artifact.stat().st_size != record["bytes"] or _sha256(artifact) != record["sha256"]:
            raise RuntimeError(f"bundle file differs from its manifest: {artifact}")
    launch = json.loads((path / "launch_abi.json").read_text())
    if (
        launch.get("checkpoint_chunk_bytes") != P.CHECKPOINT_CHUNK_BYTES
        or launch.get("checkpoint_mapped_payload_slots") != P.CHECKPOINT_PAYLOAD_SLOTS
        or launch.get("checkpoint_stored_surfaces") != COMPACT_CHECKPOINT_STORED_SURFACES
    ):
        raise RuntimeError("bundle lacks the supported in-launch checkpoint layout")
    return manifest, launch


def bundle_geometry(path: str | Path) -> Geometry:
    """Return the depth, sequence and launch limits a bundle was built for."""

    return Geometry.from_launch_abi(read_bundle(path)[1])


@dataclass
class ApplicationState:
    model: ModelState
    fabric: DecoderFabricState
    optimizer: OptimizerState
    full_shell: FullShellState
    nstep: CircularNstepRuntimeState | None = None
    token_shards: TokenShardRing | None = None

    @property
    def geometry(self) -> Geometry:
        return self.model.geometry

    @classmethod
    def allocate(
        cls,
        *,
        geometry: Geometry,
        rank: int,
        device: torch.device,
        timeout_ms: int,
        workload: TokenWindow,
    ) -> ApplicationState:
        """Allocate one rank's state for the bundle ``geometry`` describes.

        ``workload`` sets the packing every window of the run shares.  The
        circular refill session then supplies ``nstep`` and ``token_shards``.
        """

        timeout_ns = timeout_ms * 1_000_000
        model = ModelState.allocate(
            workload.segment_extents,
            device,
            geometry=geometry,
            rotary_segment_extents=workload.rotary_segment_extents,
        )
        fabric = DecoderFabricState.allocate(model, rank=rank, device=device, timeout_ns=timeout_ns)
        optimizer = OptimizerState.allocate(model, fabric, rank=rank, device=device)
        full_shell = FullShellState.allocate(
            model,
            rank=rank,
            device=device,
            timeout_ns=timeout_ns,
            input_ids=workload.input_ids,
            labels=workload.labels,
            local_valid_tokens=workload.local_valid_tokens,
        )
        model.shell.attach_checkpoint_storage(model.slabs["residual_mid"])
        state = cls(model=model, fabric=fabric, optimizer=optimizer, full_shell=full_shell)
        state.prepare(timeout_ns=timeout_ns)
        return state

    @classmethod
    def meta(cls, geometry: Geometry) -> ApplicationState:
        """Construct the pointer/shape graph without allocating GPU memory."""

        route_layout = geometry.route_layout
        owner_elements, sequence = geometry.optimizer_elements, geometry.sequence
        device = torch.device("meta")

        def tensor(elements: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(elements, dtype=dtype, device=device)

        def arena(elements: int) -> SymmetricArena:
            return SymmetricArena(tensor(elements, torch.uint8), object(), 0)

        model = ModelState.allocate(
            (geometry.sequence // MAX_DOCS,) * MAX_DOCS, device, geometry=geometry
        )
        entries = fabric_table_entries(geometry.depth)
        fabric = DecoderFabricState(
            arena(arenas.DECODER_WEIGHT_ARENA_BYTES),
            arena(arenas.DECODER_GRADIENT_ARENA_BYTES),
            tensor(entries["weight_source_table"], torch.int64),
            tensor(entries["gradient_destination_table"], torch.int64),
            tensor(4, torch.int64),
            tensor(1, torch.int32),
            tensor(5, torch.int32),
            0,
        )
        fabric.bind_model(model)
        optimizer = OptimizerState(
            arena(arenas.ALL_REDUCE_ARENA_BYTES),
            tensor(owner_elements, torch.float32),
            tensor(owner_elements, torch.float32),
            tensor(owner_elements, torch.float32),
            tensor(owner_elements, torch.float32),
            tensor(owner_elements, torch.bfloat16),
            tensor(HIDDEN, torch.float32),
            model.shell.final_norm_grad,
            tensor(HIDDEN, torch.float32),
            tensor(HIDDEN, torch.float32),
            model.shell.final_norm_weight,
            tensor(NORM_PARTIAL_ELEMENTS, torch.float32),
            tensor(4, torch.float32),
            tensor(4, torch.int32),
            tensor(1, torch.int64),
            tensor(3, torch.int64),
            tensor(6, torch.float32),
            0,
            geometry,
        )
        scheduler_states = tuple(
            tensor(
                HEAD_FORWARD_EPOCHS * SCHEDULER_WORDS if index == 14 else SCHEDULER_WORDS,
                torch.int32,
            )
            for index in range(15)
        )
        embedding_route_arena = arena(route_layout.arena_bytes)
        full_shell = FullShellState(
            arena(arenas.HEAD_WEIGHT_ARENA_BYTES),
            arena(arenas.HEAD_GRADIENT_ARENA_BYTES),
            arena(arenas.ALL_REDUCE_ARENA_BYTES),
            tensor(6, torch.int64),
            tensor(sequence, torch.int32),
            tensor(1, torch.int32),
            model.shell.head_weight,
            tensor(5, torch.int32),
            embedding_route_arena,
            tuple(embedding_route_arena.tensor for _ in range(8)),
            tensor(8, torch.int64),
            tensor(sequence, torch.int64),
            tensor(sequence, torch.int32),
            tensor(8, torch.int32),
            tensor(8, torch.int32),
            tensor(5, torch.int32),
            scheduler_states,
            route_layout,
        )
        return cls(model, fabric, optimizer, full_shell)

    def base_groups(self) -> dict[str, tuple]:
        prefix, suffix = self.model.runtime_prefix_suffix()
        return {
            "prefix": prefix,
            "suffix": suffix,
            "shell": self.model.shell.runtime_tensors(),
            "fabric": self.fabric.runtime_tensors(),
            "optimizer": self.optimizer.runtime_tensors(),
            "full_shell": self.full_shell.runtime_tensors(),
        }

    def groups(self) -> dict[str, tuple]:
        result = self.base_groups()
        if self.token_shards is not None:
            # The token-shard ring's views replace the per-step token tensors wherever the
            # groups pass them.
            (
                input_ids,
                labels,
                local_valid_tokens,
                route_unique,
                route_inverse,
                route_offsets,
                route_counts,
            ) = self.token_shards.runtime_views()
            full_shell_state = self.full_shell
            replacements = {
                id(self.model.shell.labels): labels,
                id(full_shell_state.input_ids): input_ids,
                id(full_shell_state.local_valid_tokens): local_valid_tokens,
                id(full_shell_state.embedding_route_unique_ids): route_unique,
                id(full_shell_state.embedding_route_inverse): route_inverse,
                id(full_shell_state.embedding_route_owner_offsets): route_offsets,
                id(full_shell_state.embedding_route_owner_counts): route_counts,
            }
            for group in ("shell", "full_shell"):
                result[group] = tuple(
                    replacements.get(id(value), value) for value in result[group]
                )
        if self.nstep is not None:
            result["nstep"] = self.nstep.runtime_tensors()
        return result

    def abi_contract(self, *, target_device_type: str | None = None) -> dict[str, object]:
        groups = self.groups()
        result = abi.runtime_contract(**groups)
        if target_device_type is not None:
            for row in result["argument_rows"]:
                if isinstance(row, dict) and "device_type" in row:
                    row["device_type"] = target_device_type
        return result

    def runtime_arguments(self) -> tuple[object, ...]:
        from cutlass import Int32

        groups = self.groups()
        return (
            *groups["prefix"],
            Int32(self.geometry.depth),
            *groups["suffix"],
            *groups["shell"],
            *groups["fabric"],
            *groups["optimizer"],
            *groups["full_shell"],
            *groups.get("nstep", ()),
        )

    def prepare(self, *, timeout_ns: int) -> None:
        self.model.initialize_rotary()
        self.fabric.reset()
        self.model.attention.configure_forward_schedule()
        self.model.attention.reset_metadata()
        self.optimizer.prepare_step(epoch=1, timeout_ns=timeout_ns)
        self.full_shell.prepare_step(1, timeout_ns=timeout_ns)


@dataclass
class Image:
    function: object
    module: object

    def __call__(self, state: ApplicationState) -> None:
        self.function(*state.runtime_arguments())


def _runtime_abi_matches(launch: dict[str, object], state: ApplicationState) -> bool:
    """Whether the bundle's published argument rows are the ones ``state`` passes.

    The per-step records hold one row per step; the build compiled them for
    ``resident_compile_sample_steps`` steps, so that one shape is compared at the sample count.
    """

    target = "cuda" if state.model.activation_chain.device.type == "meta" else None
    current = state.abi_contract(target_device_type=target)
    rows = [dict(row) for row in current["argument_rows"]]
    records = next(row for row in rows if row.get("name") == "nstep.1")
    records["shape"] = [launch["resident_compile_sample_steps"] * P.RECORD_WIDTH]
    return launch.get("runtime_abi") == {**current, "argument_rows": rows}


def load_image(path: str | Path, state: ApplicationState) -> Image:
    """Load a bundle's CUfunction once its argument layout matches ``state``."""

    bundle = Path(path).resolve()
    _, launch = read_bundle(bundle)
    if not _runtime_abi_matches(launch, state):
        raise RuntimeError("bundle runtime ABI differs from allocated state")
    import cutlass

    module = cutlass.runtime.load_module(str(bundle / "image.o"), enable_tvm_ffi=True)
    return Image(function=getattr(module, FUNCTION_NAME), module=module)
