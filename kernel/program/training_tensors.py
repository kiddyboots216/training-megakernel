"""The training tensors the build compiles the program against.

``TrainingTensors.allocate`` starts from ``decoder_layer.LayerTensors`` (FA4's attention tensors
and every per-layer slab, in two layer slots) and adds the storage that follows the logical
layer: the activation chain, one BF16 hidden-state tensor per layer boundary (depth + 1 of them;
layer L reads entry L and writes entry L + 1), and the gradient ring, two BF16 slots that carry
each layer's output gradient down the stack. It then builds the row table, one row of addresses
per logical layer.

``kernel/build.py`` allocates these tensors for the compile; the host runtime builds the same
layout in ``training_megakernel.state.ModelState``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import model
import torch
import training_program
from attention import AttentionPlan, AttentionTensors
from decoder_layer import (
    ROW_TABLE_FIELDS,
    SLAB_SPECS,
    GemmFamily,
    LayerTensors,
    gemm_families,
)
from training_megakernel.contract import PHYSICAL_SLOTS as LAYER_SLOTS

def incoming_gradient_slot(layer: int, slots: int) -> int:
    """The ring slot with layer ``layer``'s output gradient, read by its down GEMMs and RMS2."""

    return layer % slots


def outgoing_gradient_slot(layer: int, slots: int) -> int:
    """The ring slot layer ``layer``'s RMS1 backward writes; layer ``layer - 1`` reads it."""

    return (layer - 1) % slots


def document_boundaries(document_lengths: Iterable[int]) -> tuple[int, ...]:
    """Cumulative row offsets of the packed documents, from 0 to SEQUENCE.

    Every length must be positive, and together they must cover the sequence.
    """

    result = [0]
    for raw in document_lengths:
        length = int(raw)
        if length <= 0:
            raise ValueError("document lengths must be positive")
        result.append(result[-1] + length)
    if len(result) == 1 or result[-1] != model.SEQUENCE:
        raise ValueError(
            f"document lengths must cover exactly {model.SEQUENCE} rows"
        )
    return tuple(result)


def _row_addresses(
    activation_chain: torch.Tensor,
    gradient_ring: torch.Tensor,
    slabs: dict[str, torch.Tensor],
    depth: int,
) -> tuple[tuple[int, ...], ...]:
    """Each layer's row: the address of every per-layer tensor, in ROW_TABLE_FIELDS order.

    Every address must be 16-byte aligned, because ``anchors.row_table_view`` declares that
    alignment.
    """

    rows: list[tuple[int, ...]] = []
    for layer in range(depth):
        layer_slot = layer % LAYER_SLOTS
        row: list[int] = []
        for name in ROW_TABLE_FIELDS:
            if name == "residual_in":
                target = activation_chain[layer]
            elif name == "layer_output":
                target = activation_chain[layer + 1]
            elif name == "staging_dy":
                target = gradient_ring[incoming_gradient_slot(layer, LAYER_SLOTS)]
            elif name == "layer_output_dy":
                # This field keeps the down-forward output's layer slot (slabs["layer_output"]);
                # the down dX and dW GEMMs read the incoming gradient through the stacked
                # slabs["layer_output_dy"], which is the gradient ring.
                target = slabs["layer_output"][layer_slot]
            elif name == "layer_input_dx":
                target = gradient_ring[outgoing_gradient_slot(layer, LAYER_SLOTS)]
            else:
                tensor = slabs[name]
                target = tensor if SLAB_SPECS[name].shared else tensor[layer_slot]
            if target.data_ptr() % 16:
                raise AssertionError(f"unaligned row-table entry {layer}:{name}")
            row.append(target.data_ptr())
        rows.append(tuple(row))
    return tuple(rows)


@dataclass
class TrainingTensors:
    """The tensors the program is compiled against.

    ``slabs`` maps each per-layer operand to its two-slot tensor, or to the activation chain or
    the gradient ring (see ``allocate``); ``capacity`` is the number of layer slots.
    ``communication.decoder_fabric.attach_decoder_fabric`` rebinds the weight and gradient slabs
    into the fabric's arenas.
    """

    base: AttentionTensors
    slabs: dict[str, torch.Tensor]
    families: tuple[GemmFamily, ...]
    row_table: torch.Tensor
    capacity: int
    logical_depth: int
    activation_chain: torch.Tensor
    gradient_ring: torch.Tensor
    shell: training_program.ShellTensors
    document_lengths: tuple[int, ...]

    @classmethod
    def allocate(
        cls,
        document_lengths: Iterable[int],
        device: torch.device,
        *,
        logical_depth: int = model.DEPTH,
    ) -> TrainingTensors:
        """Allocate for packed documents with ``document_lengths`` at ``logical_depth`` layers."""

        boundaries = document_boundaries(document_lengths)
        documents = tuple(
            boundaries[index + 1] - boundaries[index]
            for index in range(len(boundaries) - 1)
        )
        plan = AttentionPlan(
            f"training-depth{logical_depth}",
            [list(documents) for _ in range(LAYER_SLOTS)],
            LAYER_SLOTS,
        )
        families = gemm_families()
        parent = LayerTensors.allocate(plan, families, device)
        # Stage the attention control plane for both layer slots, with the forward attention's
        # load-balanced tile schedule (training_megakernel.schedule.build_lpt_schedule).
        parent.base.stage_control_plane()

        activation_chain = torch.empty(
            logical_depth + 1,
            model.SEQUENCE,
            model.HIDDEN,
            dtype=torch.bfloat16,
            device=device,
        )
        gradient_ring = torch.empty(
            LAYER_SLOTS,
            model.SEQUENCE,
            model.HIDDEN,
            dtype=torch.bfloat16,
            device=device,
        )

        # The down-forward GEMM's output (slabs["layer_output"]) takes the two-slot storage
        # allocated for layer_output_dy, and layer_output_dy, the A operand of the down dX and dW
        # GEMMs, becomes the gradient ring, so those GEMMs read the incoming gradient in place.
        parent.slabs["layer_output"] = parent.slabs["layer_output_dy"]
        parent.slabs["layer_output_dy"] = gradient_ring

        # The program reaches these three only through the row table. Point their slabs at the
        # activation chain and the gradient ring, so the two-slot tensors LayerTensors allocated
        # for them are freed.
        parent.slabs["residual_in"] = activation_chain[:LAYER_SLOTS]
        parent.slabs["staging_dy"] = gradient_ring
        parent.slabs["layer_input_dx"] = gradient_ring

        address_rows = _row_addresses(
            activation_chain, gradient_ring, parent.slabs, logical_depth
        )
        row_table = torch.tensor(
            [address for row in address_rows for address in row],
            dtype=torch.int64,
            device=device,
        )
        return cls(
            base=parent.base,
            slabs=parent.slabs,
            families=parent.families,
            row_table=row_table,
            capacity=LAYER_SLOTS,
            logical_depth=logical_depth,
            activation_chain=activation_chain,
            gradient_ring=gradient_ring,
            shell=training_program.ShellTensors.allocate(device),
            document_lengths=documents,
        )

    def rebuild_row_table(self) -> None:
        """Recompute the row table after slabs were rebound.

        ``attach_decoder_fabric`` rebinds the weight and gradient slabs into its arenas.
        """

        address_rows = _row_addresses(
            self.activation_chain, self.gradient_ring, self.slabs, self.logical_depth
        )
        self.row_table = torch.tensor(
            [address for row in address_rows for address in row],
            dtype=torch.int64,
            device=self.activation_chain.device,
        )
