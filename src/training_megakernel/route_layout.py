"""Embedding-route capacity and byte layout shared by compile and runtime."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

EMBEDDING_ROUTE_WORLD_SIZE = 8
EMBEDDING_ROUTE_HIDDEN = 4_096
EMBEDDING_ROUTE_OWNER_ROWS = 18_992
EMBEDDING_ROUTE_CTAS = 132
EMBEDDING_ROUTE_THREADS = 256
EMBEDDING_ROUTE_GUARD_BYTES = 4_096
EMBEDDING_ROUTE_LIVE_CONTROL_BYTES = EMBEDDING_ROUTE_WORLD_SIZE * 4 * 2


def validate_embedding_route_records(value: int) -> int:
    """Validate one per-source/per-owner route-record capacity."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("embedding-route records must be an integer")
    if not 1 <= value <= EMBEDDING_ROUTE_OWNER_ROWS:
        raise ValueError(f"embedding-route records must be in [1, {EMBEDDING_ROUTE_OWNER_ROWS}]")
    return value


def _align_guard(value: int) -> int:
    alignment = EMBEDDING_ROUTE_GUARD_BYTES
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class EmbeddingRouteLayout:
    """Complete Phase-6 symmetric-arena layout for one record capacity."""

    records_per_source_owner: int
    compact_owner_gradient: bool
    payload_elements_per_source: int
    payload_offset: int
    payload_bytes: int
    payload_guard_offset: int
    row_ids_offset: int
    row_id_bytes: int
    row_id_guard_offset: int
    arrival_offset: int
    arrival_bytes: int
    arrival_guard_offset: int
    counts_offset: int
    count_bytes: int
    count_guard_offset: int
    consumed_offset: int
    consumed_bytes: int
    consumed_guard_offset: int
    owner_gradient_offset: int
    owner_gradient_bytes: int
    owner_gradient_guard_offset: int
    norm_partials_offset: int
    norm_partial_bytes: int
    norm_partial_guard_offset: int
    done_offset: int
    done_bytes: int
    done_guard_offset: int
    arena_bytes: int
    forward_request_offset: int
    forward_response_offset: int


@cache
def embedding_route_layout(
    records_per_source_owner: int,
    *,
    compact_owner_gradient: bool,
) -> EmbeddingRouteLayout:
    records = validate_embedding_route_records(records_per_source_owner)
    world = EMBEDDING_ROUTE_WORLD_SIZE
    hidden = EMBEDDING_ROUTE_HIDDEN
    guard = EMBEDDING_ROUTE_GUARD_BYTES
    fp32_bytes = 4

    payload_elements_per_source = records * hidden
    payload_offset = guard
    payload_bytes = world * payload_elements_per_source * fp32_bytes
    payload_guard_offset = payload_offset + payload_bytes
    row_ids_offset = _align_guard(payload_guard_offset + guard)
    row_id_bytes = world * records * 8
    row_id_guard_offset = row_ids_offset + row_id_bytes
    arrival_offset = _align_guard(row_id_guard_offset + guard)
    arrival_bytes = world * 4
    arrival_guard_offset = arrival_offset + arrival_bytes
    counts_offset = _align_guard(arrival_guard_offset + guard)
    count_bytes = world * 4
    count_guard_offset = counts_offset + count_bytes
    consumed_offset = _align_guard(count_guard_offset + guard)
    consumed_bytes = world * 4
    consumed_guard_offset = consumed_offset + consumed_bytes
    owner_gradient_offset = _align_guard(consumed_guard_offset + guard)
    owner_gradient_bytes = (
        EMBEDDING_ROUTE_LIVE_CONTROL_BYTES
        if compact_owner_gradient
        else EMBEDDING_ROUTE_OWNER_ROWS * hidden * fp32_bytes
    )
    owner_gradient_guard_offset = owner_gradient_offset + owner_gradient_bytes
    norm_partials_offset = _align_guard(owner_gradient_guard_offset + guard)
    norm_partial_bytes = EMBEDDING_ROUTE_CTAS * EMBEDDING_ROUTE_THREADS * fp32_bytes
    norm_partial_guard_offset = norm_partials_offset + norm_partial_bytes
    done_offset = _align_guard(norm_partial_guard_offset + guard)
    done_bytes = 4
    done_guard_offset = done_offset + done_bytes
    arena_bytes = _align_guard(done_guard_offset + guard)

    return EmbeddingRouteLayout(
        records_per_source_owner=records,
        compact_owner_gradient=compact_owner_gradient,
        payload_elements_per_source=payload_elements_per_source,
        payload_offset=payload_offset,
        payload_bytes=payload_bytes,
        payload_guard_offset=payload_guard_offset,
        row_ids_offset=row_ids_offset,
        row_id_bytes=row_id_bytes,
        row_id_guard_offset=row_id_guard_offset,
        arrival_offset=arrival_offset,
        arrival_bytes=arrival_bytes,
        arrival_guard_offset=arrival_guard_offset,
        counts_offset=counts_offset,
        count_bytes=count_bytes,
        count_guard_offset=count_guard_offset,
        consumed_offset=consumed_offset,
        consumed_bytes=consumed_bytes,
        consumed_guard_offset=consumed_guard_offset,
        owner_gradient_offset=owner_gradient_offset,
        owner_gradient_bytes=owner_gradient_bytes,
        owner_gradient_guard_offset=owner_gradient_guard_offset,
        norm_partials_offset=norm_partials_offset,
        norm_partial_bytes=norm_partial_bytes,
        norm_partial_guard_offset=norm_partial_guard_offset,
        done_offset=done_offset,
        done_bytes=done_bytes,
        done_guard_offset=done_guard_offset,
        arena_bytes=arena_bytes,
        forward_request_offset=owner_gradient_offset,
        forward_response_offset=owner_gradient_offset + world * 4,
    )


__all__ = (
    "EMBEDDING_ROUTE_LIVE_CONTROL_BYTES",
    "EMBEDDING_ROUTE_OWNER_ROWS",
    "EmbeddingRouteLayout",
    "embedding_route_layout",
    "validate_embedding_route_records",
)
