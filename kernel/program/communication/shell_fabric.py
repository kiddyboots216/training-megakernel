"""The shell's communication, bound for the build (ABI group ``full_shell``).

``allocate_shell_fabric`` allocates one rank's head weight ring, head gradient ring and the
all-reduce arena for the valid-token count (layouts in ``arenas``), the embedding route arena and
its tables (``embedding_route``), and the control and status words. The
``full_shell`` group also carries the work-sharing scheduler state of every dynamic GEMM body, so
those are allocated here too; ``ShellFabric.runtime_tensors`` returns the group in kernel-argument
order. The collectives run in ``training_program``: ``all_reduce_valid_tokens``,
``gather_embedding_rows``, ``all_gather_head_weight``, ``reduce_scatter_head_gradient`` and
``reduce_scatter_embedding_gradient``.

Only the build calls this module, and the compile reads only the tensors' shapes, dtypes and
alignment. The host runtime allocates the same group
(``training_megakernel.distributed.FullShellState``) and passes its token ring in place of the
input IDs, the valid-token count and the route tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import model
import torch
from training_megakernel import arenas
from communication.embedding_route import (
    ROUTE_RECORDS,
    ROUTE_STATUS_WORDS,
    EmbeddingRouteArena,
    allocate_embedding_route_arena,
)
from communication.symmetric_memory import SymmetricArena, allocate_arena
from tile_schedulers import SCHEDULER_STATE_WORDS

# The embedding and the head. Each head ring advances by one epoch per vocabulary panel of each
# matrix, VOCAB_MATRICES * VOCAB_PANELS per step.
VOCAB_MATRICES = 2

# A work-sharing scheduler state is its tile queue's ticket counter. The head forward keeps one
# scheduler state per head chunk.
HEAD_FORWARD_SCHEDULER_STATE_WORDS = model.HEAD_CHUNKS * SCHEDULER_STATE_WORDS
# One scheduler state per dynamic GEMM body, in kernel-argument order.
SCHEDULER_STATE_SIZES = (
    ("head_dx", SCHEDULER_STATE_WORDS),
    ("head_dw", SCHEDULER_STATE_WORDS),
    ("gate_up_dx", SCHEDULER_STATE_WORDS),
    ("gate_up_dw", SCHEDULER_STATE_WORDS),
    ("gate_up_fwd", SCHEDULER_STATE_WORDS),
    ("down_dx", SCHEDULER_STATE_WORDS),
    ("down_dw", SCHEDULER_STATE_WORDS),
    ("down_fwd", SCHEDULER_STATE_WORDS),
    ("qkv_dx", SCHEDULER_STATE_WORDS),
    ("qkv_dw", SCHEDULER_STATE_WORDS),
    ("qkv_fwd", SCHEDULER_STATE_WORDS),
    ("o_dx", SCHEDULER_STATE_WORDS),
    ("o_dw", SCHEDULER_STATE_WORDS),
    ("o_fwd", SCHEDULER_STATE_WORDS),
    ("head_fwd", HEAD_FORWARD_SCHEDULER_STATE_WORDS),
)

# Indices of the shell's control words: the multicast addresses of the head weight ring, the head
# gradient ring and the all-reduce arena, the rank, the wait timeout, and the multicast address of
# the head weight gradient. The runtime sets the rank and the timeout; the build leaves them zero.
SHELL_CONTROL_HEAD_WEIGHT_MULTICAST = 0
SHELL_CONTROL_HEAD_GRADIENT_MULTICAST = 1
SHELL_CONTROL_VALID_TOKENS_MULTICAST = 2
SHELL_CONTROL_RANK = 3
SHELL_CONTROL_TIMEOUT_NS = 4
SHELL_CONTROL_HEAD_DW_MULTICAST = 5

# The shell's status record keeps the first failed wait: its kind, two position words and the
# expected and observed epochs. training_program's _record_timeout_status also writes the
# decoder's record with these indices.
SHELL_STATUS_KIND = 0
SHELL_STATUS_MATRIX = 1
SHELL_STATUS_PANEL = 2
SHELL_STATUS_EXPECTED = 3
SHELL_STATUS_OBSERVED = 4
SHELL_STATUS_WORDS = 5

SHELL_STATUS_OK = 0
SHELL_STATUS_WEIGHT_REUSE_TIMEOUT = 1
SHELL_STATUS_WEIGHT_READY_TIMEOUT = 2
SHELL_STATUS_GRADIENT_REUSE_TIMEOUT = 3
SHELL_STATUS_GRADIENT_READY_TIMEOUT = 4
SHELL_STATUS_VALID_TOKENS_REUSE_TIMEOUT = 5
SHELL_STATUS_VALID_TOKENS_READY_TIMEOUT = 6


@dataclass
class ShellFabric:
    """The ``full_shell`` group of kernel arguments.

    ``embedding_route_unique_ids`` holds the sorted distinct token IDs, padded with -1;
    ``embedding_route_inverse`` each token's index into them; ``embedding_route_owner_offsets`` and
    ``embedding_route_owner_counts`` the slice of them each owner rank owns. ``head_weight`` is the
    full BF16 head weight the all-gather fills.
    """

    head_weight_arena: SymmetricArena
    head_gradient_arena: SymmetricArena
    valid_tokens_arena: SymmetricArena
    control: torch.Tensor
    input_ids: torch.Tensor
    local_valid_tokens: torch.Tensor
    head_weight: torch.Tensor
    status: torch.Tensor
    embedding_route_arena: EmbeddingRouteArena
    embedding_route_unique_ids: torch.Tensor
    embedding_route_inverse: torch.Tensor
    embedding_route_owner_offsets: torch.Tensor
    embedding_route_owner_counts: torch.Tensor
    embedding_route_status: torch.Tensor
    scheduler_states: tuple[torch.Tensor, ...]

    def runtime_tensors(self) -> tuple[torch.Tensor, ...]:
        """The kernel arguments of the ``full_shell`` group, in order.

        ``training_program.SHELL_FABRIC_AND_SCHEDULER_ARGUMENT_NAMES`` names them.
        """

        return (
            self.head_weight_arena.tensor,
            self.head_gradient_arena.tensor,
            self.valid_tokens_arena.tensor,
            self.control,
            self.input_ids,
            self.local_valid_tokens,
            self.head_weight,
            self.status,
            self.embedding_route_arena.tensor,
            self.embedding_route_arena.peer_bases,
            self.embedding_route_unique_ids,
            self.embedding_route_inverse,
            self.embedding_route_owner_offsets,
            self.embedding_route_owner_counts,
            self.embedding_route_status,
            *self.scheduler_states,
        )


def allocate_shell_fabric(
    tensors: Any,
    *,
    input_ids: torch.Tensor,
    local_valid_tokens: int,
    device: torch.device,
) -> ShellFabric:
    """Allocate one rank's shell communication and scheduler states, deriving the route tables.

    The route tables come from ``input_ids``: the sorted distinct token IDs, each token's index into
    them, and each owner rank's slice of them. Raises if an owner would need more than
    ``ROUTE_RECORDS`` records.
    """

    if input_ids.shape != (model.SEQUENCE,):
        raise ValueError(
            f"full-shell input IDs require shape {(model.SEQUENCE,)}, got {tuple(input_ids.shape)}"
        )
    if not bool(((input_ids >= 0) & (input_ids < model.VOCAB)).all().item()):
        raise ValueError("full-shell input IDs exceed the vocabulary")
    if local_valid_tokens < 0 or local_valid_tokens > model.SEQUENCE:
        raise ValueError(local_valid_tokens)

    head_weight_arena = allocate_arena(arenas.HEAD_WEIGHT_ARENA_BYTES, device)
    head_gradient_arena = allocate_arena(arenas.HEAD_GRADIENT_ARENA_BYTES, device)
    valid_tokens_arena = allocate_arena(arenas.ALL_REDUCE_ARENA_BYTES, device)
    embedding_route_arena = allocate_embedding_route_arena(device)
    route_ids = input_ids.to(device=device, dtype=torch.int64).contiguous()
    route_unique, route_inverse_long = torch.unique(
        route_ids, sorted=True, return_inverse=True
    )
    route_owners = torch.div(route_unique, arenas.VOCAB_ROWS_PER_RANK, rounding_mode="floor")
    route_counts_long = torch.bincount(route_owners, minlength=model.WORLD)
    route_counts = route_counts_long.to(dtype=torch.int32).contiguous()
    route_offsets = (
        torch.cumsum(route_counts_long, dim=0) - route_counts_long
    ).to(dtype=torch.int32).contiguous()
    route_unique_padded = torch.full(
        (model.SEQUENCE,), -1, dtype=torch.int64, device=device
    )
    route_unique_padded[: route_unique.numel()].copy_(route_unique)
    route_inverse = route_inverse_long.to(dtype=torch.int32).contiguous()
    if int(route_counts.max().item()) > ROUTE_RECORDS:
        raise ValueError(
            f"embedding route needs {int(route_counts.max().item())} records, "
            f"capacity is {ROUTE_RECORDS}"
        )
    control = torch.tensor(
        [
            head_weight_arena.multicast_base,
            head_gradient_arena.multicast_base,
            valid_tokens_arena.multicast_base,
            0,
            0,
            tensors.shell.head_dweight_multicast_base,
        ],
        dtype=torch.int64,
        device=device,
    )
    scheduler_states = tuple(
        torch.empty(words, dtype=torch.int32, device=device)
        for _name, words in SCHEDULER_STATE_SIZES
    )
    fabric = ShellFabric(
        head_weight_arena=head_weight_arena,
        head_gradient_arena=head_gradient_arena,
        valid_tokens_arena=valid_tokens_arena,
        control=control,
        input_ids=input_ids.to(device=device, dtype=torch.int32).contiguous(),
        local_valid_tokens=torch.tensor(
            [local_valid_tokens], dtype=torch.int32, device=device
        ),
        head_weight=tensors.shell.head_weight,
        status=torch.zeros(SHELL_STATUS_WORDS, dtype=torch.int32, device=device),
        embedding_route_arena=embedding_route_arena,
        embedding_route_unique_ids=route_unique_padded,
        embedding_route_inverse=route_inverse,
        embedding_route_owner_offsets=route_offsets,
        embedding_route_owner_counts=route_counts,
        embedding_route_status=torch.zeros(
            ROUTE_STATUS_WORDS, dtype=torch.int32, device=device
        ),
        scheduler_states=scheduler_states,
    )
    # Each queue starts at 0, the value the step loop restores before each step (resident_step.py).
    for scheduler_state in fabric.scheduler_states:
        scheduler_state.zero_()
    return fabric
