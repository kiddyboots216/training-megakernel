"""The embedding route: embedding rows to the ranks that use them, and their gradients back.

Each rank owns ``arenas.VOCAB_ROWS_PER_RANK`` rows of the embedding. In the forward pass each rank
sends every owner the sorted IDs of the rows it needs, and the owner writes those rows into the
requester's route arena (``training_program.gather_embedding_rows``). In the backward pass each
rank sends every owner its FP32 gradient sums for those rows, and the owner adds them, in a fixed
rank order, into its optimizer gradient (``training_program.reduce_scatter_embedding_gradient``).
The rows move with plain stores into peers' route arenas, not through multicast.

This module holds the route arena's offsets, the layout of its status record, the epoch waits and
the failure recorder the kernel uses, and the arena's allocation for the build.
"""

from __future__ import annotations

from dataclasses import dataclass
import cutlass.cute as cute
import model
import torch
from model import SHAPE
from communication.memory_ops import (
    atomic_cas_gpu_u32,
    global_timer_ns,
    load_acquire_sys_u32,
    store_relaxed_gpu_u32,
)
from cutlass import Int32, Int64
from training_megakernel.route_layout import embedding_route_layout

# The compact layout: owners add routed gradient rows straight into their optimizer gradient, so
# the owner-gradient region holds only the lookup's two epoch vectors (forward request and
# response). The build records this layout in launch_abi.json and the host runtime rebuilds the
# same layout from it.
ROUTE_LAYOUT = embedding_route_layout(SHAPE.route_records, compact_owner_gradient=True)
ROUTE_RECORDS = ROUTE_LAYOUT.records_per_source_owner
ROUTE_PAYLOAD_ELEMENTS_PER_SOURCE = ROUTE_LAYOUT.payload_elements_per_source
ROUTE_PAYLOAD_OFFSET = ROUTE_LAYOUT.payload_offset
ROUTE_ROW_IDS_OFFSET = ROUTE_LAYOUT.row_ids_offset
ROUTE_ARRIVAL_OFFSET = ROUTE_LAYOUT.arrival_offset
ROUTE_COUNTS_OFFSET = ROUTE_LAYOUT.counts_offset
ROUTE_CONSUMED_OFFSET = ROUTE_LAYOUT.consumed_offset
ROUTE_FORWARD_REQUEST_OFFSET = ROUTE_LAYOUT.forward_request_offset
ROUTE_FORWARD_RESPONSE_OFFSET = ROUTE_LAYOUT.forward_response_offset
ROUTE_DONE_OFFSET = ROUTE_LAYOUT.done_offset
ROUTE_ARENA_BYTES = ROUTE_LAYOUT.arena_bytes

# The route's status record keeps the first failed wait: its kind, CTA, peer rank and the expected
# and observed epochs. The host requires it to be all zero at a checkpoint.
ROUTE_STATUS_KIND = 0
ROUTE_STATUS_CTA = 1
ROUTE_STATUS_PEER = 2
ROUTE_STATUS_EXPECTED = 3
ROUTE_STATUS_OBSERVED = 4
ROUTE_STATUS_WORDS = 5
ROUTE_STATUS_OK = 0
ROUTE_STATUS_REUSE_TIMEOUT = 1
ROUTE_STATUS_ARRIVAL_TIMEOUT = 2
ROUTE_STATUS_FUTURE_ARRIVAL = 3


@cute.jit
def _record_route_failure(
    status: cute.Tensor,
    kind: Int32,
    cta: Int32,
    peer: Int32,
    expected: Int32,
    observed: Int32,
):
    """Record a failed wait in ``status`` unless an earlier failure has already claimed it."""

    won = atomic_cas_gpu_u32(
        status.iterator.toint() + Int64(ROUTE_STATUS_KIND * 4),
        Int32(ROUTE_STATUS_OK),
        kind,
    )
    if won == Int32(ROUTE_STATUS_OK):
        _ = store_relaxed_gpu_u32(
            status.iterator.toint() + Int64(ROUTE_STATUS_CTA * 4), cta
        )
        _ = store_relaxed_gpu_u32(
            status.iterator.toint() + Int64(ROUTE_STATUS_PEER * 4), peer
        )
        _ = store_relaxed_gpu_u32(
            status.iterator.toint() + Int64(ROUTE_STATUS_EXPECTED * 4), expected
        )
        _ = store_relaxed_gpu_u32(
            status.iterator.toint() + Int64(ROUTE_STATUS_OBSERVED * 4), observed
        )


@cute.jit
def wait_route_epoch_at_least(
    address: Int64,
    expected: Int32,
    status: cute.Tensor,
    kind: Int32,
    cta: Int32,
    peer: Int32,
    timeout_ns: Int64,
):
    """Spin until the u32 epoch at ``address`` reaches ``expected``, reading it with acquire.

    ``address`` may be in a peer's arena. After ``timeout_ns`` the wait records a failure of
    ``kind`` and returns anyway.
    """

    observed = Int32(0)
    deadline = global_timer_ns() + timeout_ns
    waiting = Int32(1)
    while waiting == Int32(1):
        observed = load_acquire_sys_u32(address)
        if observed >= expected:
            waiting = Int32(0)
        elif global_timer_ns() >= deadline:
            _record_route_failure(status, kind, cta, peer, expected, observed)
            waiting = Int32(0)


@cute.jit
def wait_route_epoch_equal(
    address: Int64,
    expected: Int32,
    status: cute.Tensor,
    cta: Int32,
    peer: Int32,
    timeout_ns: Int64,
):
    """Spin until the u32 epoch at ``address`` equals ``expected``, reading it with acquire.

    The protocol keeps peers from publishing an epoch ahead of this rank, so a larger value is
    recorded as ``ROUTE_STATUS_FUTURE_ARRIVAL`` and a timeout as ``ROUTE_STATUS_ARRIVAL_TIMEOUT``.
    Either way the wait then returns.
    """

    observed = Int32(0)
    deadline = global_timer_ns() + timeout_ns
    waiting = Int32(1)
    while waiting == Int32(1):
        observed = load_acquire_sys_u32(address)
        if observed == expected:
            waiting = Int32(0)
        elif observed > expected:
            _record_route_failure(
                status,
                Int32(ROUTE_STATUS_FUTURE_ARRIVAL),
                cta,
                peer,
                expected,
                observed,
            )
            waiting = Int32(0)
        elif global_timer_ns() >= deadline:
            _record_route_failure(
                status,
                Int32(ROUTE_STATUS_ARRIVAL_TIMEOUT),
                cta,
                peer,
                expected,
                observed,
            )
            waiting = Int32(0)


@dataclass
class EmbeddingRouteArena:
    """One rank's route arena and the addresses of every rank's arena.

    ``peer_bases`` is the device table of the eight arena addresses the kernel stores into.
    """

    tensor: torch.Tensor
    peer_bases: torch.Tensor


def allocate_embedding_route_arena(device: torch.device) -> EmbeddingRouteArena:
    """Allocate the zeroed route arena and the table of every rank's arena address."""

    tensor = torch.zeros(ROUTE_ARENA_BYTES, dtype=torch.uint8, device=device)
    # The build's one GPU has no peers: the local arena stands in for every rank's, since the
    # compile needs only the address table's shape and dtype.
    peer_bases = torch.tensor(
        [tensor.data_ptr()] * model.WORLD,
        dtype=torch.int64,
        device=device,
    )
    return EmbeddingRouteArena(tensor=tensor, peer_bases=peer_bases)
