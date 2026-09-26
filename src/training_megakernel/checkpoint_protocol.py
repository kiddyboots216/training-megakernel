"""Layout of an in-launch rank checkpoint, shared by the image and the host.

A rank shard stores the FP32 optimizer masters, moments and hyperparameters;
the two BF16 mirrors are rebuilt from their masters on restore.  The stored
tensors' shapes depend on the build's depth, so every layout function takes the
optimizer's owner element count, O(D) from ``shape.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from . import resident_protocol as P
from .shape import HIDDEN


@dataclass(frozen=True)
class Surface:
    name: str
    dtype: Literal["float32", "bfloat16"]
    shape: tuple[int, ...]
    file_offset: int
    nbytes: int


@dataclass(frozen=True)
class Chunk:
    surface_index: int
    surface_name: str
    chunk_index: int
    source_byte_offset: int
    file_offset: int
    nbytes: int


@dataclass(frozen=True)
class DerivedSurface:
    """A persistent tensor reconstructed from one stored tensor."""

    name: str
    dtype: Literal["bfloat16"]
    shape: tuple[int, ...]
    derivation: Literal["BF16_RNE"]
    source: str


def _align(value: int) -> int:
    page = P.CHECKPOINT_PAGE_BYTES
    return (value + page - 1) // page * page


def compact_stored_surfaces(owner_elements: int) -> tuple[Surface, ...]:
    """The stored tensors of one rank shard, each page aligned, in device order."""

    owner = (owner_elements,)
    final = (HIDDEN,)
    specs = (
        ("parameter", owner),
        ("exp_avg", owner),
        ("exp_avg_sq", owner),
        ("final_parameter", final),
        ("final_exp_avg", final),
        ("final_exp_avg_sq", final),
        ("hyperparameters", (6,)),
    )
    offset = 0
    rows: list[Surface] = []
    for name, shape in specs:
        offset = _align(offset)
        nbytes = 4
        for extent in shape:
            nbytes *= extent
        rows.append(Surface(name, "float32", shape, offset, nbytes))
        offset += nbytes
    return tuple(rows)


def compact_derived_surfaces(owner_elements: int) -> tuple[DerivedSurface, ...]:
    owner = (owner_elements,)
    final = (HIDDEN,)
    return (
        DerivedSurface("bf16_parameter", "bfloat16", owner, "BF16_RNE", "parameter"),
        DerivedSurface(
            "final_bf16_parameter",
            "bfloat16",
            final,
            "BF16_RNE",
            "final_parameter",
        ),
    )


def rank_file_bytes(surfaces: tuple[Surface, ...]) -> int:
    return _align(surfaces[-1].file_offset + surfaces[-1].nbytes)


def chunks_for_surfaces(
    surfaces: tuple[Surface, ...], chunk_bytes: int = P.CHECKPOINT_CHUNK_BYTES
) -> tuple[Chunk, ...]:
    """The chunks of a rank shard in the order the device sends them."""

    result: list[Chunk] = []
    for surface_index, surface in enumerate(surfaces):
        source_offset = 0
        chunk_index = 0
        while source_offset < surface.nbytes:
            nbytes = min(chunk_bytes, surface.nbytes - source_offset)
            result.append(
                Chunk(
                    surface_index=surface_index,
                    surface_name=surface.name,
                    chunk_index=chunk_index,
                    source_byte_offset=source_offset,
                    file_offset=surface.file_offset + source_offset,
                    nbytes=nbytes,
                )
            )
            source_offset += nbytes
            chunk_index += 1
    return tuple(result)
