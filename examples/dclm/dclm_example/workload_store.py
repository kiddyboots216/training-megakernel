"""One-pass workload shards for long resident training runs.

The top-level manifest describes the shard set; each rank has one compact index
and one contiguous token file of fixed-size records, so a consumer reads and
materializes exactly one packed window at a time without retaining the dataset
token payload in memory.  The manifest records the tokens per GPU
(``sequence``); records are 4 bytes per token, and a shard set trains only a
bundle built for the same sequence.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from training_megakernel.contract import VOCAB
from training_megakernel.layout import PAD_TILE
from training_megakernel.shape import PACKED_SEGMENTS
from training_megakernel.workload import TokenWindow

SHARDED_WORKLOAD_MANIFEST = "manifest.json"
SHARDED_WORKLOAD_SCHEMA = "training_megakernel_shard_manifest_v1"
SHARDED_RANK_INDEX_SCHEMA = "training_megakernel_shard_rank_index_v1"
TOKEN_BYTES = 4
IGNORE_INDEX = -100


def rank_index_name(rank: int) -> str:
    return f"rank-{rank:05d}.index.json"


def rank_token_name(rank: int) -> str:
    return f"rank-{rank:05d}.tokens.i32"


def padding_segments(pad: int, segment_count: int) -> tuple[int, ...]:
    """Split ``pad`` rows into ``segment_count`` segments, the last a 64-row multiple."""

    if segment_count == 1:
        return (pad,)
    balanced_tail = (pad // segment_count // PAD_TILE) * PAD_TILE
    if balanced_tail < PAD_TILE:
        raise ValueError("ignored padding is too short for a 64-row aligned tail")
    prefix_count = segment_count - 1
    base, remainder = divmod(pad - balanced_tail, prefix_count)
    return tuple(base + (index < remainder) for index in range(prefix_count)) + (
        balanced_tail,
    )


def load_shard_manifest(
    root: str | Path, *, sequence: int, embedding_route_records: int
) -> dict[str, Any]:
    """Read a shard set's manifest and check it fits the bundle.

    ``sequence`` and ``embedding_route_records`` are the bundle's tokens per GPU
    and route capacity; shards prepared for a smaller capacity fit.
    """

    manifest = json.loads((Path(root) / SHARDED_WORKLOAD_MANIFEST).read_bytes())
    if manifest.get("schema") != SHARDED_WORKLOAD_SCHEMA:
        raise ValueError("unexpected sharded workload manifest schema")
    if manifest["sequence"] != sequence:
        raise ValueError(
            f"the DCLM shard set holds {manifest['sequence']} tokens per GPU; the "
            f"bundle needs shards prepared for {sequence} (prepare.py shards --bundle)"
        )
    prepared_capacity = manifest["embedding_route_records_per_source_owner"]
    if prepared_capacity > embedding_route_records:
        raise ValueError(
            "sharded workload was prepared for an embedding-route capacity of "
            f"{prepared_capacity}, above the bundle's {embedding_route_records}"
        )
    return manifest


@dataclass(frozen=True)
class ShardedWindowRecord:
    stream_index: int
    documents: tuple[int, ...]
    pad: int

    @property
    def geometry(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The packed segment extents and the rotary segment extents."""

        padding = padding_segments(self.pad, PACKED_SEGMENTS - len(self.documents))
        return self.documents + padding, self.documents + (self.pad,)


@dataclass(frozen=True)
class ShardedWorkloadSource:
    """One rank-local index over a bounded streaming token file."""

    rank: int
    stream_start: int
    stream_count: int
    token_path: Path
    records: tuple[ShardedWindowRecord, ...]
    sequence: int

    @classmethod
    def open(
        cls, root: str | Path, manifest: dict[str, Any], *, rank: int
    ) -> ShardedWorkloadSource:
        """Open one rank's view of a shard set whose manifest was loaded."""

        root_path = Path(root)
        sequence = manifest["sequence"]
        stream_start = manifest["stream_start"]
        stream_count = manifest["stream_count"]
        index = json.loads((root_path / rank_index_name(rank)).read_bytes())
        records = tuple(
            ShardedWindowRecord(
                stream_index=stream_start + offset,
                documents=tuple(int(value) for value in row["docs"]),
                pad=int(row["pad"]),
            )
            for offset, row in enumerate(index["windows"])
        )
        if len(records) != stream_count or any(
            sum(record.documents) + record.pad != sequence for record in records
        ):
            raise ValueError(f"rank {rank} index does not hold {stream_count} windows")
        token_path = root_path / rank_token_name(rank)
        if token_path.stat().st_size != stream_count * TOKEN_BYTES * sequence:
            raise ValueError(f"rank {rank} token file is not {stream_count} records")
        return cls(
            rank=rank,
            stream_start=stream_start,
            stream_count=stream_count,
            token_path=token_path,
            records=records,
            sequence=sequence,
        )

    @property
    def stream_stop(self) -> int:
        return self.stream_start + self.stream_count

    def _materialize(self, descriptor: int, record: ShardedWindowRecord) -> TokenWindow:
        size = TOKEN_BYTES * self.sequence
        payload = bytearray(size)
        view = memoryview(payload)
        offset = (record.stream_index - self.stream_start) * size
        completed = 0
        while completed < size:
            count = os.preadv(descriptor, [view[completed:]], offset + completed)
            if count <= 0:
                raise EOFError("rank token file ended inside a fixed-size record")
            completed += count
        view.release()
        input_ids = torch.frombuffer(payload, dtype=torch.int32)
        if int(input_ids.min()) < 0 or int(input_ids.max()) >= VOCAB:
            raise ValueError("packed token payload contains an out-of-vocabulary token")
        labels = torch.full((self.sequence,), IGNORE_INDEX, dtype=torch.int32)
        begin = 0
        for length in record.documents:
            labels[begin : begin + length - 1] = input_ids[begin + 1 : begin + length]
            begin += length
        segment_extents, rotary_segment_extents = record.geometry
        return TokenWindow(
            input_ids=input_ids,
            labels=labels,
            local_valid_tokens=sum(record.documents) - len(record.documents),
            segment_extents=segment_extents,
            rotary_segment_extents=rotary_segment_extents,
            stream_index=record.stream_index,
        )

    def iter_windows(self, *, start_stream: int, count: int) -> Iterator[TokenWindow]:
        """Yield exactly one rank-local window at a time."""

        if start_stream < self.stream_start or start_stream + count > self.stream_stop:
            raise ValueError(
                f"requested streams [{start_stream}, {start_stream + count}) outside "
                f"sharded range [{self.stream_start}, {self.stream_stop})"
            )
        begin = start_stream - self.stream_start
        records = self.records[begin : begin + count]
        if len({record.geometry for record in records}) != 1:
            raise ValueError(
                "sharded advancing source requires identical packed and rotary geometry"
            )

        def materialize() -> Iterator[TokenWindow]:
            descriptor = os.open(self.token_path, os.O_RDONLY)
            try:
                for record in records:
                    yield self._materialize(descriptor, record)
            finally:
                os.close(descriptor)

        return materialize()


__all__ = (
    "SHARDED_RANK_INDEX_SCHEMA",
    "SHARDED_WORKLOAD_MANIFEST",
    "SHARDED_WORKLOAD_SCHEMA",
    "TOKEN_BYTES",
    "ShardedWindowRecord",
    "ShardedWorkloadSource",
    "load_shard_manifest",
    "padding_segments",
    "rank_index_name",
    "rank_token_name",
)
