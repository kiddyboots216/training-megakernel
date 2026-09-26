"""The rank's optimizer state for the build, and the decoder fabric tables that point into it.

``allocate_optimizer_state`` allocates an ``OptimizerState``: the rank's owner slice of every
trained parameter as flat FP32 parameter, gradient and Adam-moment vectors plus a BF16 copy of
the parameter, the FP32 state of the replicated final RMSNorm weight, the all-reduce arena, and
the control, status and output words ``clipped_adamw`` defines. It then points the decoder
fabric's weight-source and gradient-destination tables at per-layer views of the flat BF16
parameter and gradient, so the fabric all-gathers each layer's weights from, and reduce-scatters
its gradients into, the optimizer's own storage.

``kernel/build.py`` allocates this state for the compile; the host runtime allocates the same
tensors, in the same argument order, in ``training_megakernel.distributed.OptimizerState``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import model
import torch
from clipped_adamw import (
    HYPERPARAMETER_WORDS,
    NORM_PARTIAL_ELEMENTS,
    OPTIMIZER_ELEMENTS,
    STATUS_WORDS,
)
from training_megakernel.arenas import ALL_REDUCE_ARENA_BYTES, VOCAB_ROWS_PER_RANK
from communication.symmetric_memory import SymmetricArena, allocate_arena
from training_megakernel.layout import LAYER_OWNER_ELEMENTS, SITE_OFFSET, segment_by_name

# The flat vectors' owner segments: five per decoder layer, then the embedding and head rows.
OWNER_SEGMENT_BY_NAME = segment_by_name(model.DEPTH)


def _per_layer_view(
    flat: torch.Tensor,
    *,
    site_name: str,
    elements: int,
) -> torch.Tensor:
    """A (DEPTH, elements) view of one weight panel's or reduction site's slice in every layer."""

    return torch.as_strided(
        flat,
        size=(model.DEPTH, elements),
        stride=(LAYER_OWNER_ELEMENTS, 1),
        storage_offset=SITE_OFFSET[site_name],
    )


def _pointer_table(
    views: dict[str, torch.Tensor],
    names: tuple[str, ...],
    device: torch.device,
) -> torch.Tensor:
    """The views' addresses, layer-major with ``names`` in order, as an Int64 device tensor."""

    return torch.tensor(
        [views[name][layer].data_ptr() for layer in range(model.DEPTH) for name in names],
        dtype=torch.int64,
        device=device,
    )


@dataclass
class OptimizerState:
    """The optimizer step's tensors, and the per-layer views the decoder fabric reads and writes."""

    all_reduce_arena: SymmetricArena
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
    decoder_weight_views: dict[str, torch.Tensor]
    decoder_gradient_views: dict[str, torch.Tensor]

    def runtime_tensors(self) -> tuple[torch.Tensor, ...]:
        """Kernel arguments, in the order of ``training_program.OPTIMIZER_ARGUMENT_NAMES``."""

        # final_gradient and final_bf16_parameter are the shell's final-norm gradient and weight,
        # which the kernel already takes as shell_final_grad and shell_weight, and the optimizer
        # call uses those. Passing each tensor once means every read and write of it goes through
        # one argument, and the compiler never sees two pointer arguments that alias.
        return (
            self.all_reduce_arena.tensor,
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


def allocate_optimizer_state(
    tensors: Any,
    fabric: Any,
    *,
    device: torch.device,
) -> OptimizerState:
    """Allocate the rank's optimizer state and repoint ``fabric``'s weight and gradient tables.

    ``tensors`` must be the build's ``TrainingTensors`` (the build's depth, two layer slots).
    """

    if tensors.logical_depth != model.DEPTH or tensors.capacity != 2:
        raise ValueError(f"the optimizer state expects depth {model.DEPTH} and 2 layer slots")

    bf16_parameter = torch.empty(
        OPTIMIZER_ELEMENTS, dtype=torch.bfloat16, device=device
    )
    decoder_weight_views = {
        panel.name: _per_layer_view(
            bf16_parameter,
            site_name=panel.name,
            elements=panel.owner_elements,
        )
        for panel in model.WEIGHT_PANELS
    }

    embedding = OWNER_SEGMENT_BY_NAME["embedding"]
    head = OWNER_SEGMENT_BY_NAME["head"]
    # The build stands in for rank 0, whose owner rows come first.
    owner_head = tensors.shell.head_weight.narrow(0, 0, VOCAB_ROWS_PER_RANK).reshape(-1)
    bf16_parameter[embedding.begin : embedding.end].copy_(owner_head)
    bf16_parameter[head.begin : head.end].copy_(owner_head)

    gradient = torch.zeros(OPTIMIZER_ELEMENTS, dtype=torch.float32, device=device)
    decoder_gradient_views = {
        site.name: _per_layer_view(
            gradient,
            site_name=site.name,
            elements=site.owner_elements,
        )
        for site in model.REDUCTION_SITES
    }

    fabric.weight_source_table = _pointer_table(
        decoder_weight_views,
        tuple(panel.name for panel in model.WEIGHT_PANELS),
        device,
    )
    fabric.gradient_destination_table = _pointer_table(
        decoder_gradient_views,
        tuple(site.name for site in model.REDUCTION_SITES),
        device,
    )

    parameter = torch.empty(OPTIMIZER_ELEMENTS, dtype=torch.float32, device=device)
    parameter.copy_(bf16_parameter)
    final_parameter = torch.empty(model.HIDDEN, dtype=torch.float32, device=device)
    final_parameter.copy_(tensors.shell.final_norm_weight)
    all_reduce_arena = allocate_arena(ALL_REDUCE_ARENA_BYTES, device)
    exp_avg = torch.zeros_like(parameter)
    exp_avg_sq = torch.zeros_like(parameter)
    final_exp_avg = torch.zeros_like(final_parameter)
    final_exp_avg_sq = torch.zeros_like(final_parameter)
    return OptimizerState(
        all_reduce_arena=all_reduce_arena,
        parameter=parameter,
        gradient=gradient,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        bf16_parameter=bf16_parameter,
        final_parameter=final_parameter,
        final_gradient=tensors.shell.final_norm_grad,
        final_exp_avg=final_exp_avg,
        final_exp_avg_sq=final_exp_avg_sq,
        final_bf16_parameter=tensors.shell.final_norm_weight,
        norm_partials=torch.zeros(NORM_PARTIAL_ELEMENTS, dtype=torch.float32, device=device),
        outputs=torch.zeros(4, dtype=torch.float32, device=device),
        status=torch.zeros(STATUS_WORDS, dtype=torch.int32, device=device),
        completed_steps=torch.zeros(1, dtype=torch.int64, device=device),
        integer_control=torch.tensor(
            [all_reduce_arena.multicast_base, 0, 0],
            dtype=torch.int64,
            device=device,
        ),
        hyperparameters=torch.zeros(HYPERPARAMETER_WORDS, dtype=torch.float32, device=device),
        decoder_weight_views=decoder_weight_views,
        decoder_gradient_views=decoder_gradient_views,
    )
