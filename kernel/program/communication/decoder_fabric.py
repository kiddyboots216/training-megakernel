"""Storage for the decoder's weight all-gather and gradient reduce-scatter (ABI group ``fabric``).

``attach_decoder_fabric`` allocates the decoder weight ring and decoder gradient ring (layouts in
``arenas``) and rebinds the layer slabs of ``TrainingTensors`` to views of their slots, so the
decoder reads each layer's full weights from the weight ring and writes that layer's weight
gradients into the gradient ring. The all-gather fills a slot from every rank's weight shard, and
the reduce-scatter sums a slot over all ranks into this rank's gradient shard; both run in
``training_program`` (``publish_layer_weights``, ``reduce_scatter_layer_gradients``, ...).

``DecoderFabric`` holds the arenas and the tables, control, status and timing tensors those
collectives use; ``runtime_tensors`` returns them as the ``fabric`` group of kernel arguments.
Only the build calls this module; the host runtime allocates the same group in
``training_megakernel.distributed.DecoderFabricState``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import model
import torch
from training_megakernel import arenas
from communication.symmetric_memory import SymmetricArena, allocate_arena

if TYPE_CHECKING:
    from training_tensors import TrainingTensors


# The four RMSNorm weights share one packed weight panel ("norm"), and their gradients one packed
# reduction site: for each norm, its weight slab, gradient slab and extent, in packing order.
PACKED_NORM_SLABS = (
    ("input_norm", "input_norm_grad", model.HIDDEN),
    ("q_norm", "q_norm_grad", model.HEAD_DIM),
    ("k_norm", "k_norm_grad", model.HEAD_DIM),
    ("post_attention_norm", "post_attn_norm_grad", model.HIDDEN),
)
assert sum(size for _, _, size in PACKED_NORM_SLABS) == model.NORM_ELEMENTS

# The slab behind each other panel and site. Weight panels carry the reduction sites' names
# (qkv_dw, ...) because the optimizer finds a panel's owner shard by site name
# (training_megakernel.layout.SITE_OFFSET).
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
WEIGHT_PANEL_INDEX = {panel.name: index for index, panel in enumerate(model.WEIGHT_PANELS)}
GRADIENT_SITE_INDEX = {site.name: index for index, site in enumerate(model.REDUCTION_SITES)}


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = []
    stride = 1
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return tuple(reversed(strides))


def _layer_slot_view(
    arena: torch.Tensor,
    *,
    byte_offset: int,
    slot_stride_bytes: int,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """A ``[LAYER_SLOTS, *shape]`` view of ``arena`` from ``byte_offset``, one slot per layer slot.

    The GEMM operands are compiled with 16-byte alignment and the row table requires it, so every
    view must start on a 16-byte boundary.
    """

    element_bytes = torch.empty((), dtype=dtype).element_size()
    if byte_offset % element_bytes or slot_stride_bytes % element_bytes:
        raise AssertionError("arena view is not element aligned")
    typed = arena[byte_offset:].view(dtype)
    view = torch.as_strided(
        typed,
        size=(arenas.LAYER_SLOTS, *shape),
        stride=(slot_stride_bytes // element_bytes, *_contiguous_strides(shape)),
    )
    if view.data_ptr() % 16:
        raise AssertionError("arena view is not 16-byte aligned")
    return view


def _weight_panel_view(arena: SymmetricArena, panel_index: int) -> torch.Tensor:
    panel = model.WEIGHT_PANELS[panel_index]
    return _layer_slot_view(
        arena.tensor,
        byte_offset=(
            arenas.DECODER_WEIGHT_PAYLOAD_OFFSET
            + arenas.DECODER_WEIGHT_PANEL_OFFSETS[panel_index]
        ),
        slot_stride_bytes=arenas.DECODER_WEIGHT_SLOT_STRIDE,
        dtype=torch.bfloat16,
        shape=(panel.rows, panel.columns),
    )


def _gradient_site_view(arena: SymmetricArena, site_index: int) -> torch.Tensor:
    site = model.REDUCTION_SITES[site_index]
    return _layer_slot_view(
        arena.tensor,
        byte_offset=(
            arenas.DECODER_GRADIENT_PAYLOAD_OFFSET
            + arenas.DECODER_GRADIENT_SITE_OFFSETS[site_index]
        ),
        slot_stride_bytes=arenas.DECODER_GRADIENT_SLOT_STRIDE,
        dtype=torch.float32,
        shape=(site.rows, site.columns),
    )


@dataclass
class DecoderFabric:
    """The ``fabric`` group of kernel arguments: the decoder's two arenas and what drives them.

    ``weight_source_table`` and ``gradient_destination_table`` hold, per layer and weight panel or
    reduction site, the address of this rank's BF16 weight shard and FP32 gradient shard in the
    optimizer state. ``control`` holds the two multicast addresses, the rank and the wait timeout
    (``training_program.DECODER_CONTROL_*``; the runtime sets the last two, so the build leaves
    them zero), and ``step`` the step count the epochs derive from.
    ``status`` keeps the first failed wait.
    """

    weight_arena: SymmetricArena
    gradient_arena: SymmetricArena
    weight_source_table: torch.Tensor
    gradient_destination_table: torch.Tensor
    control: torch.Tensor
    step: torch.Tensor
    status: torch.Tensor

    def runtime_tensors(self) -> tuple[torch.Tensor, ...]:
        """The kernel arguments, in ``training_program.DECODER_FABRIC_ARGUMENT_NAMES`` order."""

        return (
            self.weight_arena.tensor,
            self.gradient_arena.tensor,
            self.weight_source_table,
            self.gradient_destination_table,
            self.control,
            self.step,
            self.status,
        )


def attach_decoder_fabric(
    tensors: TrainingTensors,
    *,
    device: torch.device,
) -> DecoderFabric:
    """Allocate the decoder's rings and rebind the layer slabs of ``tensors`` to views of them.

    The four GEMM weights (and their transposes), the four norm weights, the four GEMM weight
    gradients and the four norm gradients become layer-slot views into the rings. The row table is
    rebuilt. The two pointer tables stay
    zero until ``optimizer_state.allocate_optimizer_state`` fills them.
    """

    if tensors.logical_depth != model.DEPTH or tensors.capacity != 2:
        raise ValueError(f"decoder fabric expects depth {model.DEPTH} and 2 layer slots")

    weight_arena = allocate_arena(arenas.DECODER_WEIGHT_ARENA_BYTES, device)
    gradient_arena = allocate_arena(arenas.DECODER_GRADIENT_ARENA_BYTES, device)
    weight_source_table = torch.zeros(
        tensors.logical_depth * len(model.WEIGHT_PANELS),
        dtype=torch.int64,
        device=device,
    )
    gradient_destination_table = torch.zeros(
        tensors.logical_depth * len(model.REDUCTION_SITES),
        dtype=torch.int64,
        device=device,
    )

    # The control words, in the order of training_program's DECODER_CONTROL_* indices.
    control = torch.tensor(
        [
            weight_arena.multicast_base,
            gradient_arena.multicast_base,
            0,
            0,
        ],
        dtype=torch.int64,
        device=device,
    )
    fabric = DecoderFabric(
        weight_arena=weight_arena,
        gradient_arena=gradient_arena,
        weight_source_table=weight_source_table,
        gradient_destination_table=gradient_destination_table,
        control=control,
        step=torch.zeros(1, dtype=torch.int32, device=device),
        status=torch.zeros(5, dtype=torch.int32, device=device),
    )

    norm_weight = _weight_panel_view(weight_arena, WEIGHT_PANEL_INDEX["norm"])
    norm_gradient = _gradient_site_view(gradient_arena, GRADIENT_SITE_INDEX["norm"])
    norm_offset = 0
    for weight_name, gradient_name, extent in PACKED_NORM_SLABS:
        tensors.slabs[weight_name] = norm_weight[:, 0, norm_offset : norm_offset + extent]
        tensors.slabs[gradient_name] = norm_gradient[:, 0, norm_offset : norm_offset + extent]
        norm_offset += extent

    for panel_index, panel in enumerate(model.WEIGHT_PANELS):
        if panel.name == "norm":
            continue
        slab_name = WEIGHT_SLAB_BY_PANEL[panel.name]
        tensors.slabs[slab_name] = _weight_panel_view(weight_arena, panel_index)
        tensors.slabs[f"{slab_name}_t"] = tensors.slabs[slab_name].mT

    for site_index, site in enumerate(model.REDUCTION_SITES):
        if site.name == "norm":
            continue
        tensors.slabs[GRADIENT_SLAB_BY_SITE[site.name]] = _gradient_site_view(
            gradient_arena, site_index
        )

    tensors.rebuild_row_table()
    return fabric
