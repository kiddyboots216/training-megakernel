"""The dynamic persistent tile schedulers of the program's GEMM members, and their parameters.

A GEMM phase shares out its tiles through a queue: one Int32 ticket counter in global memory.
CTA b first runs tile b; each further tile comes from an atomic ticket.  The step's reset
(`resident_step`) returns every queue to 0.  `DynamicTileScheduler` serves
queues used once per step (the LM head's).  `EpochDynamicTileScheduler` serves a decoder
family's queue, which every layer's phase of that family reuses within a step, and
`EpochDynamicTileSchedulerWithoutCta0` is its qkv dX form, which leaves CTA 0 free.
`dynamic_scheduler_params` and `dynamic_scheduler_params_offset` switch a member's prepared
scheduler parameters to dynamic persistence on a queue.

Adapted from `TileScheduler.create` and `TileScheduler._fetch_next_work_idx` in
quack/tile_scheduler.py (Copyright (c) 2025, Tri Dao), from quack-kernels 0.6.0.  Quack is
distributed under the Apache License 2.0 (see THIRD_PARTY_NOTICES.md).

- `create` takes the first tile from the CTA's x index (the program's grid is 132 x 1 x 1,
  while Quack's persistent grid runs along z) and supports only dynamic persistence on a
  1x1x1 cluster.
- `_fetch_next_work_idx` is the dynamic branch with a relaxed `atomic_add` in place of Quack's
  wrapping increment, offset by the x grid size; `EpochDynamicTileScheduler` and its subclass
  also take the ticket modulo the phase's tile count.
"""

from __future__ import annotations

import dataclasses

import cutlass.cute as cute
from cutlass import Int32, const_expr
from quack.pipeline import PipelineStateWAdvance
from quack.tile_scheduler import PersistenceMode, TileScheduler
from model import PROGRAM_CTAS


# A queue state: the ticket counter.
SCHEDULER_STATE_WORDS = 1
# The CTAs that run qkv dX while CTA 0 reduces the q/k norm gradients.
CTAS_WITHOUT_CTA0 = PROGRAM_CTAS - 1


class DynamicTileScheduler(TileScheduler):
    """Quack's dynamic persistent scheduler on the program's grid, for a queue used once per step.

    `is_scheduler_warp` is part of Quack's `create` signature and unused here.
    """

    @staticmethod
    @cute.jit
    def create(
        params,
        sched_smem=None,
        scheduler_pipeline=None,
        is_scheduler_warp=False,
        *,
        loc=None,
        ip=None,
    ):
        assert const_expr(params.persistence_mode == PersistenceMode.DYNAMIC)
        assert const_expr(cute.size(params.cluster_shape_mnk, loc=loc, ip=ip) == 1)
        assert sched_smem is not None and scheduler_pipeline is not None
        stages = const_expr(cute.size(sched_smem, mode=[1]))
        return DynamicTileScheduler(
            Int32(cute.arch.block_idx()[0]),
            Int32(0),
            Int32(0),
            Int32(0),
            sched_smem,
            scheduler_pipeline,
            PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(0)),
            params,
            loc=loc,
            ip=ip,
        )

    @cute.jit
    def _fetch_next_work_idx(self, *, loc=None, ip=None):
        next_work_idx = Int32(0)
        if cute.arch.lane_idx() == 0:
            next_work_idx = Int32(cute.arch.grid_dim()[0]) + cute.arch.atomic_add(
                self.params.tile_count_semaphore,
                Int32(1),
                sem="relaxed",
                scope="gpu",
                loc=loc,
                ip=ip,
            )
        return cute.arch.shuffle_sync(next_work_idx, 0)


class EpochDynamicTileScheduler(DynamicTileScheduler):
    """The dynamic scheduler for a queue that every layer's phase of one family reuses.

    A ticket t gives tile `grid_dim.x + t % T`, T being the phase's tile count.  A phase takes
    T tickets: one per tile after the first 132, and one more per CTA that ran a tile, to find
    the queue empty.  So the counter, never reset between layers, is a multiple of T when each
    phase starts.
    """

    @staticmethod
    @cute.jit
    def create(
        params,
        sched_smem=None,
        scheduler_pipeline=None,
        is_scheduler_warp=False,
        *,
        loc=None,
        ip=None,
    ):
        assert const_expr(params.persistence_mode == PersistenceMode.DYNAMIC)
        assert const_expr(cute.size(params.cluster_shape_mnk, loc=loc, ip=ip) == 1)
        assert sched_smem is not None and scheduler_pipeline is not None
        stages = const_expr(cute.size(sched_smem, mode=[1]))
        return EpochDynamicTileScheduler(
            Int32(cute.arch.block_idx()[0]),
            Int32(0),
            Int32(0),
            Int32(0),
            sched_smem,
            scheduler_pipeline,
            PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(0)),
            params,
            loc=loc,
            ip=ip,
        )

    @cute.jit
    def _fetch_next_work_idx(self, *, loc=None, ip=None):
        next_work_idx = Int32(0)
        if cute.arch.lane_idx() == 0:
            raw_ticket = cute.arch.atomic_add(
                self.params.tile_count_semaphore,
                Int32(1),
                sem="relaxed",
                scope="gpu",
                loc=loc,
                ip=ip,
            )
            task_tiles = Int32(cute.size(self.params.problem_shape_ncluster_mnl))
            next_work_idx = Int32(cute.arch.grid_dim()[0]) + raw_ticket % task_tiles
        return cute.arch.shuffle_sync(next_work_idx, 0)


class EpochDynamicTileSchedulerWithoutCta0(
    EpochDynamicTileScheduler
):
    """`EpochDynamicTileScheduler` on CTAs 1..131, for qkv dX.

    CTA 0 reduces the q/k norm gradients meanwhile and never creates this scheduler.  CTA b
    first runs tile b - 1, and the ticketed tiles start at tile 131; each phase still takes T
    tickets.
    """

    @staticmethod
    @cute.jit
    def create(
        params,
        sched_smem=None,
        scheduler_pipeline=None,
        is_scheduler_warp=False,
        *,
        loc=None,
        ip=None,
    ):
        assert const_expr(params.persistence_mode == PersistenceMode.DYNAMIC)
        assert const_expr(cute.size(params.cluster_shape_mnk, loc=loc, ip=ip) == 1)
        assert sched_smem is not None and scheduler_pipeline is not None
        stages = const_expr(cute.size(sched_smem, mode=[1]))
        physical_bidx = Int32(cute.arch.block_idx()[0])
        return EpochDynamicTileSchedulerWithoutCta0(
            physical_bidx - Int32(1),
            Int32(0),
            Int32(0),
            Int32(0),
            sched_smem,
            scheduler_pipeline,
            PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(0)),
            params,
            loc=loc,
            ip=ip,
        )

    @cute.jit
    def _fetch_next_work_idx(self, *, loc=None, ip=None):
        next_work_idx = Int32(0)
        if cute.arch.lane_idx() == 0:
            raw_ticket = cute.arch.atomic_add(
                self.params.tile_count_semaphore,
                Int32(1),
                sem="relaxed",
                scope="gpu",
                loc=loc,
                ip=ip,
            )
            task_tiles = Int32(cute.size(self.params.problem_shape_ncluster_mnl))
            next_work_idx = Int32(CTAS_WITHOUT_CTA0) + raw_ticket % task_tiles
        return cute.arch.shuffle_sync(next_work_idx, 0)


@cute.jit
def dynamic_scheduler_params(params, scheduler_state: cute.Tensor):
    """The prepared scheduler parameters with dynamic persistence on `scheduler_state`."""

    return dataclasses.replace(
        params,
        tile_count_semaphore=scheduler_state.iterator,
        persistence_mode=PersistenceMode.DYNAMIC,
    )


@cute.jit
def dynamic_scheduler_params_offset(
    params,
    scheduler_state: cute.Tensor,
    word_offset: Int32,
):
    """`dynamic_scheduler_params` for the queue state `word_offset` words into the tensor.

    The head forward keeps one queue state per chunk in one tensor.
    """

    return dataclasses.replace(
        params,
        tile_count_semaphore=scheduler_state.iterator + word_offset,
        persistence_mode=PersistenceMode.DYNAMIC,
    )
