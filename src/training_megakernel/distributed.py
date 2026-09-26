"""WORLD8 symmetric arenas and optimizer/shell launch state."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import arenas
from .geometry import Geometry
from .layout import (
    LAYER_OWNER_ELEMENTS,
    REDUCTION_SITES,
    SITE_OFFSET,
    WEIGHT_PANELS,
)
from .route_layout import EmbeddingRouteLayout
from .shards import embedding_route
from .state import HIDDEN, PHYSICAL_SLOTS, ModelState

WORLD_SIZE = 8
EMBEDDING_ROUTE_STATUS_WORDS = 5
NORM_PARTIAL_ELEMENTS = 52_416
SCHEDULER_WORDS = 1
HEAD_FORWARD_EPOCHS = 8

WEIGHT_SLAB_BY_PANEL = {
    "qkv_dw": "qkv_weight",
    "o_dw": "o_weight",
    "gate_up_dw": "gate_up_weight",
    "down_dw": "down_weight",
}
GRADIENT_SLAB_BY_SITE = {
    "qkv_dw": "qkv_grad",
    "o_dw": "o_grad",
    "gate_up_dw": "gate_up_grad",
    "down_dw": "down_grad",
}
NORM_FIELDS = (
    ("input_norm", "input_norm_grad", HIDDEN),
    ("q_norm", "q_norm_grad", 128),
    ("k_norm", "k_norm_grad", 128),
    ("post_attention_norm", "post_attn_norm_grad", HIDDEN),
)


def fabric_table_entries(depth: int) -> dict[str, int]:
    """Entries of the decoder-fabric tables at ``depth`` layers: one pointer per layer and
    weight panel or reduction site."""

    panels, sites = len(WEIGHT_PANELS), len(REDUCTION_SITES)
    return {
        "weight_source_table": depth * panels,
        "gradient_destination_table": depth * sites,
    }


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    result: list[int] = []
    stride = 1
    for extent in reversed(shape):
        result.append(stride)
        stride *= extent
    return tuple(reversed(result))


def _two_slot_view(
    arena: torch.Tensor,
    *,
    byte_offset: int,
    slot_stride_bytes: int,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    element_bytes = torch.empty((), dtype=dtype).element_size()
    if byte_offset % element_bytes or slot_stride_bytes % element_bytes:
        raise AssertionError("symmetric view is not element aligned")
    typed = arena[byte_offset:].view(dtype)
    return torch.as_strided(
        typed,
        size=(PHYSICAL_SLOTS, *shape),
        stride=(slot_stride_bytes // element_bytes, *_contiguous_strides(shape)),
    )


@dataclass
class SymmetricArena:
    tensor: torch.Tensor
    handle: object
    multicast_base: int

    @classmethod
    def allocate(cls, nbytes: int, device: torch.device) -> SymmetricArena:
        import torch.distributed as dist
        from torch.distributed import _symmetric_memory as symm

        if not dist.is_initialized() or dist.get_world_size() != WORLD_SIZE:
            raise RuntimeError("symmetric megakernel state requires an initialized WORLD8 group")
        tensor = symm.empty(nbytes, dtype=torch.uint8, device=device)
        tensor.zero_()
        handle = symm.rendezvous(tensor, dist.group.WORLD)
        multicast_base = int(getattr(handle, "multicast_ptr", 0) or 0)
        if multicast_base == 0:
            raise RuntimeError("WORLD8 NVLS multicast mapping is unavailable")
        return cls(tensor=tensor, handle=handle, multicast_base=multicast_base)


@dataclass
class DecoderFabricState:
    weight_arena: SymmetricArena
    gradient_arena: SymmetricArena
    weight_source_table: torch.Tensor
    gradient_destination_table: torch.Tensor
    control: torch.Tensor
    step: torch.Tensor
    status: torch.Tensor
    rank: int

    @classmethod
    def allocate(
        cls,
        model: ModelState,
        *,
        rank: int,
        device: torch.device,
        timeout_ns: int,
    ) -> DecoderFabricState:
        if not 0 <= rank < WORLD_SIZE or timeout_ns <= 0:
            raise ValueError("invalid decoder-fabric control")
        weight_arena = SymmetricArena.allocate(arenas.DECODER_WEIGHT_ARENA_BYTES, device)
        gradient_arena = SymmetricArena.allocate(arenas.DECODER_GRADIENT_ARENA_BYTES, device)
        entries = fabric_table_entries(model.geometry.depth)
        state = cls(
            weight_arena=weight_arena,
            gradient_arena=gradient_arena,
            weight_source_table=torch.zeros(
                entries["weight_source_table"], dtype=torch.int64, device=device
            ),
            gradient_destination_table=torch.zeros(
                entries["gradient_destination_table"], dtype=torch.int64, device=device
            ),
            control=torch.tensor(
                [weight_arena.multicast_base, gradient_arena.multicast_base, rank, timeout_ns],
                dtype=torch.int64,
                device=device,
            ),
            step=torch.zeros(1, dtype=torch.int32, device=device),
            status=torch.zeros(5, dtype=torch.int32, device=device),
            rank=rank,
        )
        state.bind_model(model)
        return state

    def bind_model(self, model: ModelState) -> None:
        weight_stacks: dict[str, torch.Tensor] = {}
        for index, panel in enumerate(WEIGHT_PANELS):
            weight_stacks[panel.name] = _two_slot_view(
                self.weight_arena.tensor,
                byte_offset=(
                    arenas.DECODER_WEIGHT_PAYLOAD_OFFSET
                    + arenas.DECODER_WEIGHT_PANEL_OFFSETS[index]
                ),
                slot_stride_bytes=arenas.DECODER_WEIGHT_SLOT_STRIDE,
                dtype=torch.bfloat16,
                shape=(panel.rows, panel.columns),
            )
        gradient_stacks: dict[str, torch.Tensor] = {}
        for index, site in enumerate(REDUCTION_SITES):
            gradient_stacks[site.name] = _two_slot_view(
                self.gradient_arena.tensor,
                byte_offset=(
                    arenas.DECODER_GRADIENT_PAYLOAD_OFFSET
                    + arenas.DECODER_GRADIENT_SITE_OFFSETS[index]
                ),
                slot_stride_bytes=arenas.DECODER_GRADIENT_SLOT_STRIDE,
                dtype=torch.float32,
                shape=(site.rows, site.columns),
            )

        norm_weight = weight_stacks["norm"]
        norm_gradient = gradient_stacks["norm"]
        offset = 0
        for weight_name, gradient_name, extent in NORM_FIELDS:
            model.slabs[weight_name] = norm_weight[:, 0, offset : offset + extent]
            model.slabs[gradient_name] = norm_gradient[:, 0, offset : offset + extent]
            offset += extent
        for panel_name, slab_name in WEIGHT_SLAB_BY_PANEL.items():
            model.slabs[slab_name] = weight_stacks[panel_name]
            model.slabs[f"{slab_name}_t"] = model.slabs[slab_name].mT
        for site_name, slab_name in GRADIENT_SLAB_BY_SITE.items():
            model.slabs[slab_name] = gradient_stacks[site_name]
        model.rebuild_row_table()

    def runtime_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.weight_arena.tensor,
            self.gradient_arena.tensor,
            self.weight_source_table,
            self.gradient_destination_table,
            self.control,
            self.step,
            self.status,
        )

    def reset(self) -> None:
        self.weight_arena.tensor.zero_()
        self.gradient_arena.tensor.zero_()
        for tensor in self.runtime_tensors()[5:]:
            tensor.zero_()


def _decoder_view(
    flat: torch.Tensor, site_name: str, elements: int, depth: int
) -> torch.Tensor:
    return torch.as_strided(
        flat,
        size=(depth, elements),
        stride=(LAYER_OWNER_ELEMENTS, 1),
        storage_offset=SITE_OFFSET[site_name],
    )


def _pointer_table(views: dict[str, torch.Tensor], names: tuple[str, ...]) -> torch.Tensor:
    first = next(iter(views.values()))
    return torch.tensor(
        [views[name][layer].data_ptr() for layer in range(first.shape[0]) for name in names],
        dtype=torch.int64,
        device=first.device,
    )


@dataclass
class OptimizerState:
    reduction_arena: SymmetricArena
    parameter: torch.Tensor
    gradient: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    bf16_parameter: torch.Tensor
    final_parameter: torch.Tensor
    final_gradient: torch.Tensor
    final_exp_avg: torch.Tensor
    final_exp_avg_sq: torch.Tensor
    final_bf16_parameter: torch.Tensor
    norm_partials: torch.Tensor
    outputs: torch.Tensor
    status: torch.Tensor
    completed_steps: torch.Tensor
    integer_control: torch.Tensor
    hyperparameters: torch.Tensor
    rank: int
    geometry: Geometry

    @classmethod
    def allocate(
        cls,
        model: ModelState,
        fabric: DecoderFabricState,
        *,
        rank: int,
        device: torch.device,
    ) -> OptimizerState:
        if rank != fabric.rank:
            raise ValueError("optimizer and fabric ranks differ")
        geometry = model.geometry
        elements, depth = geometry.optimizer_elements, geometry.depth
        bf16_parameter = torch.empty(elements, dtype=torch.bfloat16, device=device)
        gradient = torch.zeros(elements, dtype=torch.float32, device=device)
        decoder_weights = {
            panel.name: _decoder_view(bf16_parameter, panel.name, panel.owner_elements, depth)
            for panel in WEIGHT_PANELS
        }
        decoder_gradients = {
            site.name: _decoder_view(gradient, site.name, site.owner_elements, depth)
            for site in REDUCTION_SITES
        }
        fabric.weight_source_table = _pointer_table(
            decoder_weights, tuple(panel.name for panel in WEIGHT_PANELS)
        )
        fabric.gradient_destination_table = _pointer_table(
            decoder_gradients, tuple(site.name for site in REDUCTION_SITES)
        )
        reduction_arena = SymmetricArena.allocate(arenas.ALL_REDUCE_ARENA_BYTES, device)
        parameter = torch.empty(elements, dtype=torch.float32, device=device)
        final_parameter = torch.ones(HIDDEN, dtype=torch.float32, device=device)
        state = cls(
            reduction_arena=reduction_arena,
            parameter=parameter,
            gradient=gradient,
            exp_avg=torch.zeros_like(parameter),
            exp_avg_sq=torch.zeros_like(parameter),
            bf16_parameter=bf16_parameter,
            final_parameter=final_parameter,
            final_gradient=model.shell.final_norm_grad,
            final_exp_avg=torch.zeros_like(final_parameter),
            final_exp_avg_sq=torch.zeros_like(final_parameter),
            final_bf16_parameter=model.shell.final_norm_weight,
            norm_partials=torch.zeros(NORM_PARTIAL_ELEMENTS, dtype=torch.float32, device=device),
            outputs=torch.zeros(4, dtype=torch.float32, device=device),
            status=torch.zeros(4, dtype=torch.int32, device=device),
            completed_steps=torch.zeros(1, dtype=torch.int64, device=device),
            integer_control=torch.tensor(
                [reduction_arena.multicast_base, 0, 0],
                dtype=torch.int64,
                device=device,
            ),
            hyperparameters=torch.zeros(6, dtype=torch.float32, device=device),
            rank=rank,
            geometry=geometry,
        )
        return state

    def runtime_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.reduction_arena.tensor,
            self.parameter,
            self.gradient,
            self.exp_avg,
            self.exp_avg_sq,
            self.bf16_parameter,
            self.final_parameter,
            self.final_exp_avg,
            self.final_exp_avg_sq,
            self.norm_partials,
            self.outputs,
            self.status,
            self.completed_steps,
            self.integer_control,
            self.hyperparameters,
        )

    def prepare_step(
        self,
        epoch: int = 1,
        *,
        timeout_ns: int,
        learning_rate: float = 3e-4,
        beta1: float = 0.9,
        beta2: float = 0.95,
        epsilon: float = 1e-8,
        weight_decay: float = 0.1,
        max_grad_norm: float = 1.0,
    ) -> None:
        if epoch < 1 or timeout_ns <= 0:
            raise ValueError("invalid optimizer step")
        self.integer_control.copy_(
            torch.tensor(
                [self.reduction_arena.multicast_base, epoch, timeout_ns],
                dtype=torch.int64,
                device=self.integer_control.device,
            )
        )
        self.hyperparameters.copy_(
            torch.tensor(
                [learning_rate, beta1, beta2, epsilon, weight_decay, max_grad_norm],
                dtype=torch.float32,
                device=self.hyperparameters.device,
            )
        )


@dataclass
class FullShellState:
    weight_arena: SymmetricArena
    gradient_arena: SymmetricArena
    valid_arena: SymmetricArena
    control: torch.Tensor
    input_ids: torch.Tensor
    local_valid_tokens: torch.Tensor
    head_weight: torch.Tensor
    status: torch.Tensor
    embedding_route_arena: SymmetricArena
    embedding_route_peer_buffers: tuple[torch.Tensor, ...]
    embedding_route_peer_bases: torch.Tensor
    embedding_route_unique_ids: torch.Tensor
    embedding_route_inverse: torch.Tensor
    embedding_route_owner_offsets: torch.Tensor
    embedding_route_owner_counts: torch.Tensor
    embedding_route_status: torch.Tensor
    scheduler_states: tuple[torch.Tensor, ...]
    embedding_route_layout: EmbeddingRouteLayout

    @classmethod
    def allocate(
        cls,
        model: ModelState,
        *,
        rank: int,
        device: torch.device,
        timeout_ns: int,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        local_valid_tokens: int,
    ) -> FullShellState:
        route_layout = model.geometry.route_layout
        sequence = model.geometry.sequence
        input_ids = input_ids.to(device=device, dtype=torch.int32).contiguous()
        labels = labels.to(device=device, dtype=torch.int32).contiguous()

        route_unique_padded, route_inverse, route_offsets, route_counts = (
            tensor.to(device=device)
            for tensor in embedding_route(
                input_ids,
                sequence=sequence,
                embedding_route_records=route_layout.records_per_source_owner,
            )
        )

        weight_arena = SymmetricArena.allocate(arenas.HEAD_WEIGHT_ARENA_BYTES, device)
        gradient_arena = SymmetricArena.allocate(arenas.HEAD_GRADIENT_ARENA_BYTES, device)
        valid_arena = SymmetricArena.allocate(arenas.ALL_REDUCE_ARENA_BYTES, device)
        embedding_route_arena = SymmetricArena.allocate(route_layout.arena_bytes, device)
        model.shell.labels.copy_(labels)
        model.shell.global_valid_tokens.fill_(local_valid_tokens)
        if embedding_route_arena.handle is not None and hasattr(
            embedding_route_arena.handle, "get_buffer"
        ):
            peer_buffers = tuple(
                embedding_route_arena.handle.get_buffer(
                    peer, (route_layout.arena_bytes,), torch.uint8
                )
                for peer in range(WORLD_SIZE)
            )
        else:
            peer_buffers = tuple(embedding_route_arena.tensor for _ in range(WORLD_SIZE))
        peer_values = [peer.data_ptr() for peer in peer_buffers]
        states = tuple(
            torch.empty(
                HEAD_FORWARD_EPOCHS * SCHEDULER_WORDS if index == 14 else SCHEDULER_WORDS,
                dtype=torch.int32,
                device=device,
            )
            for index in range(15)
        )
        state = cls(
            weight_arena=weight_arena,
            gradient_arena=gradient_arena,
            valid_arena=valid_arena,
            control=torch.tensor(
                [
                    weight_arena.multicast_base,
                    gradient_arena.multicast_base,
                    valid_arena.multicast_base,
                    rank,
                    timeout_ns,
                    model.shell.head_dweight_multicast_base,
                ],
                dtype=torch.int64,
                device=device,
            ),
            input_ids=input_ids,
            local_valid_tokens=torch.tensor([local_valid_tokens], dtype=torch.int32, device=device),
            head_weight=model.shell.head_weight,
            status=torch.zeros(5, dtype=torch.int32, device=device),
            embedding_route_arena=embedding_route_arena,
            embedding_route_peer_buffers=peer_buffers,
            embedding_route_peer_bases=torch.tensor(peer_values, dtype=torch.int64, device=device),
            embedding_route_unique_ids=route_unique_padded,
            embedding_route_inverse=route_inverse,
            embedding_route_owner_offsets=route_offsets,
            embedding_route_owner_counts=route_counts,
            embedding_route_status=torch.zeros(
                EMBEDDING_ROUTE_STATUS_WORDS, dtype=torch.int32, device=device
            ),
            scheduler_states=states,
            embedding_route_layout=route_layout,
        )
        state.prepare_step(1, timeout_ns=timeout_ns)
        return state

    def prepare_step(self, epoch: int, *, timeout_ns: int) -> None:
        if epoch < 1 or timeout_ns <= 0:
            raise ValueError("invalid full-shell step control")
        self.control[4] = timeout_ns
        self.embedding_route_status.zero_()
        for state in self.scheduler_states:
            state.zero_()

    def runtime_tensors(self) -> tuple[torch.Tensor, ...]:
        result = (
            self.weight_arena.tensor,
            self.gradient_arena.tensor,
            self.valid_arena.tensor,
            self.control,
            self.input_ids,
            self.local_valid_tokens,
            self.head_weight,
            self.status,
            self.embedding_route_arena.tensor,
            self.embedding_route_peer_bases,
            self.embedding_route_unique_ids,
            self.embedding_route_inverse,
            self.embedding_route_owner_offsets,
            self.embedding_route_owner_counts,
            self.embedding_route_status,
            *self.scheduler_states,
        )
        if len(result) != 30:
            raise AssertionError("full-shell ABI drifted")
        return result
