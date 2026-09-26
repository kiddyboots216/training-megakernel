"""Device-resident packed-token rings for multi-step megakernel calls."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .contract import WORLD_SIZE
from .geometry import Geometry
from .layout import EMBEDDING_OWNER_ROWS
from .route_layout import validate_embedding_route_records
from .workload import TokenWindow


def embedding_route(
    input_ids: torch.Tensor,
    *,
    sequence: int,
    embedding_route_records: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The embedding route of one window: padded unique IDs, inverse, owner offsets, counts."""

    route_capacity = validate_embedding_route_records(embedding_route_records)
    route_ids = input_ids.to(device="cpu", dtype=torch.int64)
    unique, inverse = torch.unique(route_ids, sorted=True, return_inverse=True)
    owners = torch.div(unique, EMBEDDING_OWNER_ROWS, rounding_mode="floor")
    counts64 = torch.bincount(owners, minlength=WORLD_SIZE)
    if counts64.numel() != WORLD_SIZE:
        raise ValueError("embedding route produced an owner outside WORLD8")
    max_records = int(counts64.max().item())
    if max_records > route_capacity:
        raise ValueError(
            f"embedding route needs {max_records} records, capacity is {route_capacity}"
        )
    padded = torch.full((sequence,), -1, dtype=torch.int64)
    padded[: unique.numel()].copy_(unique)
    offsets = torch.cumsum(counts64, dim=0) - counts64
    return (
        padded,
        inverse.to(dtype=torch.int32),
        offsets.to(dtype=torch.int32),
        counts64.to(dtype=torch.int32),
    )


@dataclass(frozen=True)
class TokenShardRing:
    """Precomputed inputs needed to advance the resident program at an optimizer boundary.

    Tensors are flattened intentionally.  The CuTe successor derives a
    one-window view from these same seven existing ABI arguments at each PC-0
    re-entry, avoiding copies and preserving the model body's pointer topology.
    """

    input_ids: torch.Tensor
    labels: torch.Tensor
    local_valid_tokens: torch.Tensor
    embedding_route_unique_ids: torch.Tensor
    embedding_route_inverse: torch.Tensor
    embedding_route_owner_offsets: torch.Tensor
    embedding_route_owner_counts: torch.Tensor
    slots: int
    segment_extents: tuple[int, ...]
    rotary_segment_extents: tuple[int, ...]
    geometry: Geometry

    @property
    def sequence(self) -> int:
        return self.geometry.sequence

    @property
    def embedding_route_records(self) -> int:
        return self.geometry.route_records

    @classmethod
    def from_workloads(
        cls,
        workloads: Sequence[TokenWindow],
        *,
        device: torch.device,
        geometry: Geometry,
    ) -> TokenShardRing:
        sequence = geometry.sequence
        route_capacity = validate_embedding_route_records(geometry.route_records)
        shards = tuple(workloads)
        if not shards:
            raise ValueError("a token shard ring must contain at least one window")
        extents = {row.input_ids.numel() for row in shards} | {
            row.labels.numel() for row in shards
        }
        if extents != {sequence}:
            raise ValueError(
                f"token windows have {sorted(extents)} rows; the bundle runs "
                f"{sequence} tokens per GPU"
            )
        packing = {(row.segment_extents, row.rotary_segment_extents) for row in shards}
        if len(packing) != 1:
            raise ValueError(
                "advancing-shard v1 requires identical packed and rotary geometry"
            )
        routes = tuple(
            embedding_route(
                row.input_ids,
                sequence=sequence,
                embedding_route_records=route_capacity,
            )
            for row in shards
        )

        def stacked(rows: Sequence[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
            return torch.stack(tuple(rows)).to(device=device, dtype=dtype).contiguous().view(-1)

        return cls(
            input_ids=stacked(tuple(row.input_ids for row in shards), torch.int32),
            labels=stacked(tuple(row.labels for row in shards), torch.int32),
            local_valid_tokens=torch.tensor(
                [row.local_valid_tokens for row in shards],
                dtype=torch.int32,
                device=device,
            ),
            embedding_route_unique_ids=stacked(
                tuple(route[0] for route in routes), torch.int64
            ),
            embedding_route_inverse=stacked(
                tuple(route[1] for route in routes), torch.int32
            ),
            embedding_route_owner_offsets=stacked(
                tuple(route[2] for route in routes), torch.int32
            ),
            embedding_route_owner_counts=stacked(
                tuple(route[3] for route in routes), torch.int32
            ),
            slots=len(shards),
            segment_extents=shards[0].segment_extents,
            rotary_segment_extents=shards[0].rotary_segment_extents,
            geometry=geometry,
        )

    @classmethod
    def meta(cls, *, slots: int, geometry: Geometry) -> TokenShardRing:
        if slots < 1:
            raise ValueError("token shard slots must be positive")
        validate_embedding_route_records(geometry.route_records)
        sequence = geometry.sequence
        device = torch.device("meta")
        return cls(
            input_ids=torch.empty(slots * sequence, dtype=torch.int32, device=device),
            labels=torch.empty(slots * sequence, dtype=torch.int32, device=device),
            local_valid_tokens=torch.empty(slots, dtype=torch.int32, device=device),
            embedding_route_unique_ids=torch.empty(
                slots * sequence, dtype=torch.int64, device=device
            ),
            embedding_route_inverse=torch.empty(
                slots * sequence, dtype=torch.int32, device=device
            ),
            embedding_route_owner_offsets=torch.empty(
                slots * WORLD_SIZE, dtype=torch.int32, device=device
            ),
            embedding_route_owner_counts=torch.empty(
                slots * WORLD_SIZE, dtype=torch.int32, device=device
            ),
            slots=slots,
            segment_extents=(),
            rotary_segment_extents=(),
            geometry=geometry,
        )

    def runtime_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.input_ids,
            self.labels,
            self.local_valid_tokens,
            self.embedding_route_unique_ids,
            self.embedding_route_inverse,
            self.embedding_route_owner_offsets,
            self.embedding_route_owner_counts,
        )

    def runtime_views(self) -> tuple[torch.Tensor, ...]:
        """Return ABI-shaped heads while retaining each full backing storage."""

        sequence = self.sequence
        return (
            self.input_ids[:sequence],
            self.labels[:sequence],
            self.local_valid_tokens[:1],
            self.embedding_route_unique_ids[:sequence],
            self.embedding_route_inverse[:sequence],
            self.embedding_route_owner_offsets[:WORLD_SIZE],
            self.embedding_route_owner_counts[:WORLD_SIZE],
        )


__all__ = ("TokenShardRing", "embedding_route")
