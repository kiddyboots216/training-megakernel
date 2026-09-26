"""Runtime-packing-derived FA4-forward task scheduling.

The production forward scheduler enumerates one document, one KV head, and one
query ``m_block`` at a time.  A static persistent grid assigns visit ordinal
``i`` to CTA ``i % grid_ctas``.  With ragged packed documents, the native
document/head/block order gives almost every CTA the same *number* of tasks but
can give a few CTAs much more causal K-loop work.

Every forward task owns a disjoint output/LSE tile.  The schedule therefore
keeps the task-local causal K-loop intact and changes only which persistent CTA
executes each whole task.  Tasks are assigned longest-processing-time first to
the least-loaded CTA, subject to the exact visit capacity that CTA has in the
native static-stride grid.  The resulting table is a deterministic permutation
derived only from runtime document lengths and fixed kernel geometry.  It uses
no clocks or profiles.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ForwardTask:
    ordinal: int
    document: int
    head: int
    m_block: int
    blocks_in_document: int
    tile_m: int
    tile_n: int

    @property
    def causal_loop_work(self) -> int:
        """Number of causal K tiles visited by this whole query tile."""

        return math.ceil(self.tile_m * (self.m_block + 1) / self.tile_n)


def _positive_ints(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    return result


def block_counts(document_lengths: Iterable[int], *, tile_m: int) -> tuple[int, ...]:
    lengths = _positive_ints(document_lengths, name="document_lengths")
    if tile_m <= 0:
        raise ValueError("tile_m must be positive")
    return tuple(math.ceil(length / tile_m) for length in lengths)


def native_tasks(
    document_lengths: Iterable[int],
    *,
    kv_heads: int,
    tile_m: int,
    tile_n: int,
) -> tuple[ForwardTask, ...]:
    """Enumerate the production document/head/m-block ordinal space."""

    if kv_heads <= 0:
        raise ValueError("kv_heads must be positive")
    if tile_n <= 0:
        raise ValueError("tile_n must be positive")
    counts = block_counts(document_lengths, tile_m=tile_m)
    tasks: list[ForwardTask] = []
    ordinal = 0
    for document, blocks in enumerate(counts):
        for head in range(kv_heads):
            for m_block in range(blocks):
                tasks.append(
                    ForwardTask(
                        ordinal=ordinal,
                        document=document,
                        head=head,
                        m_block=m_block,
                        blocks_in_document=blocks,
                        tile_m=tile_m,
                        tile_n=tile_n,
                    )
                )
                ordinal += 1
    return tuple(tasks)


def visit_capacities(task_count: int, *, grid_ctas: int) -> tuple[int, ...]:
    """Return the native number of static-stride visits owned by each CTA."""

    if task_count < 0:
        raise ValueError("task_count must be nonnegative")
    if grid_ctas <= 0:
        raise ValueError("grid_ctas must be positive")
    return tuple(len(range(cta, task_count, grid_ctas)) for cta in range(grid_ctas))


def build_lpt_schedule(
    document_lengths: Iterable[int],
    *,
    kv_heads: int,
    tile_m: int,
    tile_n: int,
    grid_ctas: int,
) -> tuple[int, ...]:
    """Return visit ordinal -> native task ordinal.

    Each CTA receives exactly ``len(range(cta, task_count, grid_ctas))`` tasks,
    so this permutation can replace the native ordinal decode without changing
    the persistent scheduler's visit count or termination law.  Heap and task
    tie-breakers are explicit to make the table reproducible across hosts and
    Python versions.
    """

    tasks = native_tasks(
        document_lengths,
        kv_heads=kv_heads,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    capacities = visit_capacities(len(tasks), grid_ctas=grid_ctas)
    assignments: list[list[int]] = [[] for _ in range(grid_ctas)]

    # (modeled load, assigned visit count, CTA) gives a deterministic LPT law.
    heap = [(0, 0, cta) for cta, capacity in enumerate(capacities) if capacity > 0]
    heapq.heapify(heap)
    for task_ordinal in sorted(
        range(len(tasks)),
        key=lambda ordinal: (-tasks[ordinal].causal_loop_work, ordinal),
    ):
        if not heap:
            raise AssertionError("LPT heap exhausted before all tasks were assigned")
        load, assigned, cta = heapq.heappop(heap)
        assignments[cta].append(task_ordinal)
        load += tasks[task_ordinal].causal_loop_work
        assigned += 1
        if assigned < capacities[cta]:
            heapq.heappush(heap, (load, assigned, cta))

    schedule: list[int | None] = [None] * len(tasks)
    for cta, task_ordinals in enumerate(assignments):
        if len(task_ordinals) != capacities[cta]:
            raise AssertionError("LPT assignment changed a CTA's visit capacity")
        for visit, task_ordinal in enumerate(task_ordinals):
            schedule[cta + visit * grid_ctas] = task_ordinal
    if any(task_ordinal is None for task_ordinal in schedule):
        raise AssertionError("LPT schedule contains an unfilled visit slot")
    result = tuple(int(task_ordinal) for task_ordinal in schedule)
    if sorted(result) != list(range(len(tasks))):
        raise AssertionError("LPT schedule is not a task permutation")
    return result


__all__ = (
    "ForwardTask",
    "block_counts",
    "build_lpt_schedule",
    "native_tasks",
    "visit_capacities",
)
