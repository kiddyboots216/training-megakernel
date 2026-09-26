"""Quack's RMSNorm forward and backward kernels as members of the program, on 128-thread shards.

Adapted from `RMSNorm.kernel` and `RMSNormBackward.kernel` in quack/rmsnorm.py (Copyright
(c) 2025, Wentao Guo, Ted Zadouri, Tri Dao), from quack-kernels 0.6.0.  Quack is distributed
under the Apache License 2.0 (see THIRD_PARTY_NOTICES.md).  `block_reduce`,
`block_or_cluster_reduce` and `row_reduce` are adapted from the functions of the same names in
quack/reduce.py (Copyright (c) 2025, Tri Dao), from the same release.

- `rmsnorm_forward_body` is the forward kernel as a persistent grid-stride body: a loop of
  `task_waves` waves over row tiles replaces the block-indexed single tile, and `is_even_N`
  is fixed True (the hidden size fills the tile, so no column predicate is built).
- `rmsnorm_backward_body` is the backward kernel with its constexpr ranges lowered to Python
  ranges.
- `block_reduce` takes its warp index and barrier from the shard instead of the CTA,
  `block_or_cluster_reduce` has no cluster branch (the shards never form clusters), and
  `row_reduce` passes the shard through.

Both bodies take shared memory from the program's page and run on one 128-thread shard of the
CTA (`shard`): the thread, warp and block indices, the grid size and the barriers they use are
the shard's.  Roles 1 and 2 host shards 0 and 1, so the grid has 2 x 132 shards, and
`use_rms_shard_page` gives each shard half of the page.  decoder_layer.run_row_phase runs the
layer norms this way, and training_program runs the final norm.
"""

from __future__ import annotations

from functools import partial
import operator
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr, pipeline
from cutlass.cute.nvgpu import cpasync
from quack import copy_utils, utils
from quack.reduce import warp_reduce
from quack.rmsnorm_config import RmsNormBwdConfig, RmsNormFwdConfig

import model
from program_smem import ProgramSmemAllocator


# The RMS bodies run as two independent 128-thread shards per CTA, on roles 1 and 2 (the
# 224-register roles); role 0 takes no part.
RMS_SHARD_THREADS = 128
RMS_SHARDS = 2
# ProgramSmemAllocator's default capacity, restored after an RMS phase; each shard gets half.
ALLOCATOR_PAGE_BYTES = 230400
RMS_SHARD_PAGE_BYTES = ALLOCATOR_PAGE_BYTES // RMS_SHARDS
# Named barriers of the two shards.  FA4's backward tile scheduler uses barrier 12, so the
# shards use 11 and 14.
RMS_SHARD_BARRIER_IDS = (11, 14)
# 128 threads per row, so a shard's tile is one row; no cluster.  The backward reloads dY from
# shared memory after the row reduction and uses a two-stage cp.async pipeline (no TMA).
RMS_FORWARD_CONFIG = RmsNormFwdConfig(RMS_SHARD_THREADS, 128, 1, None, False)
RMS_BACKWARD_CONFIG = RmsNormBwdConfig(
    RMS_SHARD_THREADS, 128, 1, "smem", None, False, 2
)
# The forward body's grid-stride waves (one row per shard per wave), and the rows of the
# backward's dW partial (one per shard of the grid).
RMS_FORWARD_WAVES = -(-model.SEQUENCE // (RMS_SHARDS * model.PROGRAM_CTAS))
RMS_PARTIAL_ROWS = RMS_SHARDS * model.PROGRAM_CTAS


def use_rms_shard_page(full_page, shard: int) -> None:
    """Point ProgramSmemAllocator at shard `shard`'s half of the page, while a body is traced."""

    # The page is the raw shared-memory pointer the kernel allocated.  Offset the pointer
    # rather than wrapping it in a tensor: the allocator applies each body's own layouts.
    ProgramSmemAllocator.page = full_page + shard * RMS_SHARD_PAGE_BYTES
    ProgramSmemAllocator.capacity_bytes = RMS_SHARD_PAGE_BYTES


def restore_full_page(full_page) -> None:
    """Point ProgramSmemAllocator back at the whole page."""

    ProgramSmemAllocator.page = full_page
    ProgramSmemAllocator.capacity_bytes = ALLOCATOR_PAGE_BYTES


# Shard s runs on role s + 1, and each CTA hosts two shards of the grid.  These return the
# shard's own indices and synchronize only the shard's threads: the CTA-wide barrier would
# also wait for the other shard and for role 0, which runs no RMS work.
def shard_thread_idx(shard: int):
    tidx, tidy, tidz = cute.arch.thread_idx()
    return (tidx - RMS_SHARD_THREADS * (shard + 1), tidy, tidz)


def shard_warp_idx(shard: int):
    return cute.arch.warp_idx() - 4 * (shard + 1)


def shard_block_idx(shard: int):
    bidx, bidy, bidz = cute.arch.block_idx()
    return (bidx * RMS_SHARDS + shard, bidy, bidz)


def shard_grid_dim():
    gdim, gdy, gdz = cute.arch.grid_dim()
    return (gdim * RMS_SHARDS, gdy, gdz)


def shard_barrier(shard: int):
    return cute.arch.barrier(
        barrier_id=RMS_SHARD_BARRIER_IDS[shard], number_of_threads=RMS_SHARD_THREADS
    )


# quack/reduce.py's block_reduce, block_or_cluster_reduce and row_reduce on one shard (the
# module docstring says what changed).
@cute.jit
def block_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    init_val: cute.Numeric = 0.0,
    dtype: cutlass.Constexpr = None,
    shard: cutlass.Constexpr[int] = 0,
) -> cute.Numeric:
    """Reduce across the shard's warps; reduction_buffer has shape
    (num_warps / warps_per_row, warps_per_row)."""
    lane_idx, warp_idx = cute.arch.lane_idx(), shard_warp_idx(shard)
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    shard_barrier(shard)
    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return warp_reduce(block_reduce_val, op, dtype=dtype)


@cute.jit
def block_or_cluster_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: Optional[cute.Pointer],
    phase: Optional[Int32] = None,
    init_val: cute.Numeric = 0.0,
    dtype: cutlass.Constexpr = None,
    shard: cutlass.Constexpr[int] = 0,
) -> cute.Numeric:
    """Block reduction on the shard; `mbar_ptr` must be None (the shards never form a cluster)."""
    assert mbar_ptr is None, "the RMS shards never run as a cluster"
    return block_reduce(
        val,
        op,
        reduction_buffer,
        init_val=init_val,
        dtype=dtype,
        shard=shard,
    )


@cute.jit
def row_reduce(
    x: cute.TensorSSA | cute.Numeric,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor] = None,
    mbar_ptr: Optional[cute.Pointer] = None,
    phase: Optional[Int32] = None,
    init_val: cute.Numeric = 0.0,
    hook_fn: Optional[Callable] = None,
    shard: cutlass.Constexpr[int] = 0,
) -> cute.Numeric:
    """reduction_buffer must have shape (num_warps / warps_per_row, (warps_per_row, cluster_n))"""
    if const_expr(isinstance(x, cute.TensorSSA)):
        val = x.reduce(op, init_val=init_val, reduction_profile=0)
    else:
        val = x
    warp_op = {
        cute.ReductionOp.ADD: operator.add,
        cute.ReductionOp.MAX: cute.arch.fmax if const_expr(x.dtype == Float32) else max,
        cute.ReductionOp.MIN: cute.arch.fmin if const_expr(x.dtype == Float32) else min,
        cute.ReductionOp.MUL: operator.mul,
    }[op]
    val = warp_reduce(
        val,
        warp_op,
        threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE),
        dtype=x.dtype,
    )
    if const_expr(hook_fn is not None):
        hook_fn()
    if const_expr(reduction_buffer is not None):
        warps_per_row, cluster_n = reduction_buffer.shape[1]
        assert cluster_n == 1 or mbar_ptr is not None, (
            "mbar_ptr must be provided for cluster reduction"
        )
        if const_expr(warps_per_row > 1 or cluster_n > 1):
            val = block_or_cluster_reduce(
                val,
                warp_op,
                reduction_buffer,
                mbar_ptr,
                phase=phase,
                init_val=init_val,
                dtype=x.dtype,
                shard=shard,
            )
    return val


@cute.jit
def rmsnorm_forward_body(
    self,
    mX: cute.Tensor,
    mW: Optional[cute.Tensor],
    mB: Optional[cute.Tensor],
    mRes: Optional[cute.Tensor],
    mO: cute.Tensor,
    mResO: Optional[cute.Tensor],
    mRstd: Optional[cute.Tensor],
    mMean: Optional[cute.Tensor],
    eps: Float32,
    tiler_mn: cute.Shape,
    tiled_copy: cute.TiledCopy,
    threads_per_row: cutlass.Constexpr[int],
    task_waves: cutlass.Constexpr[int],
    shard: cutlass.Constexpr[int],
):
    """Quack's `RMSNorm.kernel` on shard `shard`, over `task_waves` grid-stride waves of row
    tiles."""
    tidx, _, _ = shard_thread_idx(shard)
    bidx_start, _, bidz = shard_block_idx(shard)
    gdim, _, _ = shard_grid_dim()
    cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else shard_block_idx(shard)[1]
    tv_layout = tiled_copy.layout_tv_tiled
    smem = ProgramSmemAllocator()
    sX = smem.allocate_tensor(
        mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
    )
    if const_expr(mRes is not None):
        sRes = smem.allocate_tensor(
            mRes.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )
    reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)
    if const_expr(cute.rank(mX) == 3):
        mX, mW, mB, mRes, mO, mResO, mRstd, mMean = [
            mT[None, bidz, None] if const_expr(mT is not None) else None
            for mT in (mX, mW, mB, mRes, mO, mResO, mRstd, mMean)
        ]
    shape = (cute.size(mX, mode=[0]), cute.size(mX, mode=[1]))
    idX = cute.make_identity_tensor(shape)
    is_even_N = True
    for wave in cutlass.range(0, task_waves, 1, unroll=1):
        bidx = bidx_start + wave * gdim
        gX, gRes, gO, gResO, gRstd, gMean, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) if mT is not None else None
            for mT in (mX, mRes, mO, mResO, mRstd, mMean, idX)
        ]
        gW, gB = [
            cute.local_tile(mT, tiler_mn, (0, cluster_y)) if const_expr(mT is not None) else None
            for mT in (mW, mB)
        ]
        thr_copy_X = tiled_copy.get_slice(tidx)
        tXgW = thr_copy_X.partition_S(gW) if const_expr(mW is not None) else None
        tXgB = thr_copy_X.partition_S(gB) if const_expr(mB is not None) else None
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        if const_expr(mRes is not None):
            tXgRes = thr_copy_X.partition_S(gRes)
            tXsRes = thr_copy_X.partition_D(sRes)
        tXgO = thr_copy_X.partition_D(gO)
        if const_expr(mResO is not None):
            tXgResO = thr_copy_X.partition_D(gResO)
        tXrRstd = thr_copy_X.partition_D(gRstd) if const_expr(mRstd is not None) else None
        tXrMean = thr_copy_X.partition_D(gMean) if const_expr(mMean is not None) else None
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]
        tXrW = cute.make_rmem_tensor_like(tXgW) if const_expr(mW is not None) else None
        tXrB = cute.make_rmem_tensor_like(tXgB) if const_expr(mB is not None) else None
        tXrX, tXrO = [cute.make_rmem_tensor_like(t) for t in (tXgX, tXgO)]
        if const_expr(mRes is not None):
            tXrRes = cute.make_rmem_tensor_like(tXgRes)
        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)
        tXpX = (
            copy_utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if not is_even_N
            else None
        )
        copy = partial(copy_utils.copy, pred=tXpX)
        row = tXcX[0][0]
        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
            if const_expr(mRes is not None):
                copy(tXgRes, tXsRes, is_async=True)
        cute.arch.cp_async_commit_group()
        if const_expr(not self.delay_w_load):
            if const_expr(mW is not None):
                copy(tXgW, tXrW)
            if const_expr(mB is not None):
                copy(tXgB, tXrB)
        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)
        if const_expr(mRes is not None):
            cute.autovec_copy(tXsRes, tXrRes)
            x += tXrRes.load().to(cute.Float32)
        if const_expr(mResO is not None):
            tXrResO = cute.make_rmem_tensor_like(tXgResO)
            tXrResO.store(x.to(tXrResO.element_type))
            if row < shape[0]:
                copy(tXrResO, tXgResO)
        mean, rstd = (None, None)
        if const_expr(self.is_layernorm):
            sum_x = row_reduce(
                x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr + 0 if const_expr(self.cluster_n > 1) else None,
                init_val=0.0,
                hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
                shard=shard,
            )
            mean = sum_x / shape[1]
            if const_expr(mMean is not None):
                if (
                    tXcX[0][1] == 0
                    and row < shape[0]
                    and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
                ):
                    tXrMean[0] = mean
            if const_expr(self.reload_from == "smem"):
                cute.autovec_copy(tXsX, tXrX)
                x = tXrX.load().to(cute.Float32)
                if const_expr(mRes is not None):
                    cute.autovec_copy(tXsRes, tXrRes)
                    x += tXrRes.load().to(cute.Float32)
            elif const_expr(self.reload_from == "gmem"):
                copy(tXgX, tXrX)
                x = tXrX.load().to(cute.Float32)
                if const_expr(mRes is not None):
                    copy(tXgRes, tXrRes)
                    x += tXrRes.load().to(cute.Float32)
            x_centered = x - mean
            if const_expr(not is_even_N):
                tXrX_centered = cute.make_rmem_tensor_like(tXrX, Float32)
                tXrX_centered.store(x_centered)
                utils.fill_oob(tXrX_centered, tXpX, fill_value=Float32.zero)
                x_centered = tXrX_centered.load()
            sum_sq_x_sub_mean = row_reduce(
                x_centered * x_centered,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 1],
                mbar_ptr + 1 if const_expr(self.cluster_n > 1) else None,
                init_val=0.0,
                shard=shard,
            )
            rstd = cute.math.rsqrt(sum_sq_x_sub_mean / shape[1] + eps, fastmath=True)
        else:
            mean = const_expr(0.0)
            sum_sq_x = row_reduce(
                x * x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                init_val=0.0,
                hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
                shard=shard,
            )
            rstd = cute.math.rsqrt(sum_sq_x / shape[1] + eps, fastmath=True)
        if const_expr(mRstd is not None):
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXrRstd[0] = rstd
        if const_expr(self.delay_w_load):
            if const_expr(mW is not None):
                copy(tXgW, tXrW)
            if const_expr(mB is not None):
                copy(tXgB, tXrB)
        if const_expr(self.reload_from == "smem" or self.reload_from == "gmem"):
            if const_expr(self.reload_from == "smem"):
                cute.autovec_copy(tXsX, tXrX)
                if const_expr(mRes is not None):
                    cute.autovec_copy(tXsRes, tXrRes)
            else:
                copy(tXgX, tXrX)
                if const_expr(mRes is not None):
                    copy(tXgRes, tXrRes)
            x = tXrX.load().to(cute.Float32)
            if const_expr(mRes is not None):
                x += tXrRes.load().to(cute.Float32)
        x_hat = (x - mean) * rstd if const_expr(self.is_layernorm) else x * rstd
        y = x_hat
        if const_expr(mW is not None):
            y *= tXrW.load().to(cute.Float32)
        if const_expr(mB is not None):
            y += tXrB.load().to(cute.Float32)
        tXrO.store(y.to(tXrO.element_type))
        if row < shape[0]:
            copy(tXrO, tXgO)


@cute.jit
def rmsnorm_backward_body(
    self,
    mX: cute.Tensor,
    mW: Optional[cute.Tensor],
    mdO: cute.Tensor,
    mdResO: Optional[cute.Tensor],
    mRstd: cute.Tensor,
    mdX: cute.Tensor,
    mdW: Optional[cute.Tensor],
    mdB: Optional[cute.Tensor],
    mdRes: Optional[cute.Tensor],
    tma_atom_X: Optional[cute.CopyAtom],
    mX_tma: Optional[cute.Tensor],
    tma_atom_dO: Optional[cute.CopyAtom],
    mdO_tma: Optional[cute.Tensor],
    tiler_mn: cute.Shape,
    tiled_copy: cute.TiledCopy,
    threads_per_row: cutlass.Constexpr[int],
    shard: cutlass.Constexpr[int],
):
    """Quack's `RMSNormBackward.kernel` on shard `shard`; each shard writes one dW partial row
    (row `2 * blockIdx.x + shard` of `mdW`)."""
    tidx, _, _ = shard_thread_idx(shard)
    warp_id = cute.arch.make_warp_uniform(shard_warp_idx(shard))
    if const_expr(self.per_head):
        bidx_start, _, bidz = shard_block_idx(shard)
    else:
        bidx_start, _, _ = shard_block_idx(shard)
    gdim, _, _ = shard_grid_dim()
    cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else shard_block_idx(shard)[1]
    tv_layout = tiled_copy.layout_tv_tiled
    if const_expr(self.per_head):
        mX, mW, mdO, mdResO, mdX, mdW, mdB, mdRes = [
            mT[None, bidz, None] if const_expr(mT is not None) else None
            for mT in (mX, mW, mdO, mdResO, mdX, mdW, mdB, mdRes)
        ]
        mRstd = mRstd[None, bidz]
    shape = mX.shape
    M = shape[0]
    is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
    idX = cute.make_identity_tensor(shape)
    smem = ProgramSmemAllocator()
    USE_TMA = const_expr(self.USE_TMA)
    n_smem_stages = const_expr(self.config.smem_stages)
    smem_layout = cute.make_ordered_layout(
        (tiler_mn[0], tiler_mn[1], n_smem_stages), order=(1, 0, 2)
    )
    smem_align = const_expr(128 if USE_TMA else 16)
    sX = smem.allocate_tensor(mX.element_type, smem_layout, byte_alignment=smem_align)
    sdO = smem.allocate_tensor(mdO.element_type, smem_layout, byte_alignment=smem_align)
    reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
        smem, tv_layout, is_persistent=True
    )
    if const_expr(mbar_ptr is not None):
        mbar_full_ptr, mbar_empty_ptr = (mbar_ptr, mbar_ptr + 2)
    else:
        mbar_full_ptr, mbar_empty_ptr = (None, None)
    thr_copy_X = tiled_copy.get_slice(tidx)
    gX, gdO, gdResO, gdX, gdRes, cX = [
        cute.local_tile(mT, tiler_mn, (None, cluster_y)) if mT is not None else None
        for mT in (mX, mdO, mdResO, mdX, mdRes, idX)
    ]
    gW = cute.local_tile(mW, tiler_mn, (0, cluster_y)) if mW is not None else None
    gdW, gdB = [
        cute.local_tile(mT, (1, tiler_mn[1]), (bidx_start, cluster_y))
        if const_expr(mT is not None)
        else None
        for mT in (mdW, mdB)
    ]
    tXgX = thr_copy_X.partition_S(gX)
    tXsX = thr_copy_X.partition_D(sX)
    tXgdO = thr_copy_X.partition_S(gdO)
    tXsdO = thr_copy_X.partition_D(sdO)
    tXgdX = thr_copy_X.partition_D(gdX)
    if const_expr(mdResO is not None):
        tXgdResO = thr_copy_X.partition_S(gdResO)
    if const_expr(mdRes is not None):
        tXgdRes = thr_copy_X.partition_D(gdRes)
    tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None, None]
    tXrX, tXrdO, tXrdX = [
        cute.make_rmem_tensor_like(thr[None, None, None, 0]) for thr in (tXgX, tXgdO, tXgdX)
    ]
    tXrdResO = None
    if const_expr(mdResO is not None):
        tXrdResO = cute.make_rmem_tensor_like(tXgdResO[None, None, None, 0])
    tXrdRes = None
    if const_expr(mdRes is not None):
        tXrdRes = cute.make_rmem_tensor_like(tXgdRes[None, None, None, 0])
    tXpX = (
        None
        if is_even_N
        else copy_utils.predicate_k(thr_copy_X.partition_S(cX[None, None, 0]), limit=shape[1])
    )
    copy = partial(copy_utils.copy, pred=tXpX)
    tXgdW, tXrdW = (None, None)
    tXgdB, tXrdB = (None, None)
    if const_expr(mdW is not None):
        tXgdW = thr_copy_X.partition_S(gdW)
        tXrdW = cute.make_rmem_tensor_like(tXgdW, Float32)
    if const_expr(mdB is not None):
        tXgdB = thr_copy_X.partition_S(gdB)
        tXrdB = cute.make_rmem_tensor_like(tXgdB, Float32)
    num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
    NUM_PIPE_STAGES = const_expr(self.config.smem_stages)
    if const_expr(USE_TMA):
        tma_mbar_ptr = smem.allocate_array(Int64, num_elems=NUM_PIPE_STAGES * 2)
    self._initialize_cluster(tidx, mbar_ptr, num_warps, is_persistent=True)
    tXrW = None
    if const_expr(mW is not None):
        tXgW = thr_copy_X.partition_S(gW)
        tXrW = cute.make_rmem_tensor_like(tXgW)
        if const_expr(not is_even_N):
            tXrW.fill(0.0)
        copy(tXgW, tXrW)
        tXrW.store((tXrW.load().to(Float32) + Float32(0.0)).to(tXrW.element_type))
    if const_expr(self.cluster_n > 1):
        cute.arch.cluster_wait()
    producer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, NUM_PIPE_STAGES
    )
    consumer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, NUM_PIPE_STAGES
    )
    if const_expr(USE_TMA):
        num_threads_total = cute.size(tiled_copy)
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_threads_total)
        tma_bytes_x = const_expr(cute.size(tiler_mn) * mX.element_type.width // 8)
        tma_bytes_do = const_expr(cute.size(tiler_mn) * mdO.element_type.width // 8)
        tma_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=tma_mbar_ptr,
            num_stages=NUM_PIPE_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tma_bytes_x + tma_bytes_do,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        gX_tma = cute.local_tile(mX_tma, tiler_mn, (None, cluster_y))
        gdO_tma = cute.local_tile(mdO_tma, tiler_mn, (None, cluster_y))
        tXsX_tma, tXgX_tma = cpasync.tma_partition(
            tma_atom_X,
            0,
            cute.make_layout(1),
            cute.group_modes(sX, 0, 2),
            cute.group_modes(gX_tma, 0, 2),
        )
        tXsdO_tma, tXgdO_tma = cpasync.tma_partition(
            tma_atom_dO,
            0,
            cute.make_layout(1),
            cute.group_modes(sdO, 0, 2),
            cute.group_modes(gdO_tma, 0, 2),
        )
    if const_expr(mdW is not None):
        tXrdW.fill(0.0)
    if const_expr(mdB is not None):
        tXrdB.fill(0.0)
    stage = Int32(0)
    producer_phase = Int32(1)
    consumer_phase = Int32(0)
    next_wave_work_id = (NUM_PIPE_STAGES - 1) * gdim
    next_wave_row_id = next_wave_work_id * tiler_mn[0]
    M_ceil = cute.ceil_div(M, tiler_mn[0])
    if const_expr(USE_TMA):
        for prefetch_iter in range(const_expr(NUM_PIPE_STAGES - 1)):
            init_bidx = bidx_start + prefetch_iter * gdim
            if warp_id == 0:
                if init_bidx < M_ceil:
                    tma_pipeline.producer_acquire(producer_state)
                    pipe_bar = tma_pipeline.producer_get_barrier(producer_state)
                    cute.copy(
                        tma_atom_X,
                        tXgX_tma[None, init_bidx],
                        tXsX_tma[None, producer_state.index],
                        tma_bar_ptr=pipe_bar,
                    )
                    cute.copy(
                        tma_atom_dO,
                        tXgdO_tma[None, init_bidx],
                        tXsdO_tma[None, producer_state.index],
                        tma_bar_ptr=pipe_bar,
                    )
                    tma_pipeline.producer_commit(producer_state)
                    producer_state.advance()
    else:
        for prefetch_iter in range(const_expr(NUM_PIPE_STAGES - 1)):
            init_bidx = bidx_start + prefetch_iter * gdim
            init_row = tXcX[None, None, None, init_bidx][0][0]
            if init_row < M:
                copy(
                    tXgX[None, None, None, init_bidx],
                    tXsX[None, None, None, producer_state.index],
                    is_async=True,
                )
                copy(
                    tXgdO[None, None, None, init_bidx],
                    tXsdO[None, None, None, producer_state.index],
                    is_async=True,
                )
            elif const_expr(tiler_mn[0] > 1):
                utils.fill_oob(
                    tXsX[None, None, None, producer_state.index],
                    None,
                    fill_value=mX.element_type.zero,
                )
                utils.fill_oob(
                    tXsdO[None, None, None, producer_state.index],
                    None,
                    fill_value=mdO.element_type.zero,
                )
            cute.arch.cp_async_commit_group()
            producer_state.advance()
    for bidx in cutlass.range(bidx_start, cute.ceil_div(M, tiler_mn[0]), gdim):
        row = tXcX[None, None, None, bidx][0][0]
        if const_expr(USE_TMA):
            ahead_bidx = bidx + next_wave_work_id
            if warp_id == 0:
                if ahead_bidx < M_ceil:
                    tma_pipeline.producer_acquire(producer_state)
                    pipe_bar = tma_pipeline.producer_get_barrier(producer_state)
                    cute.copy(
                        tma_atom_X,
                        tXgX_tma[None, ahead_bidx],
                        tXsX_tma[None, producer_state.index],
                        tma_bar_ptr=pipe_bar,
                    )
                    cute.copy(
                        tma_atom_dO,
                        tXgdO_tma[None, ahead_bidx],
                        tXsdO_tma[None, producer_state.index],
                        tma_bar_ptr=pipe_bar,
                    )
                    tma_pipeline.producer_commit(producer_state)
                    producer_state.advance()
        else:
            ahead_bidx = bidx + next_wave_work_id
            if row + next_wave_row_id < M:
                copy(
                    tXgX[None, None, None, ahead_bidx],
                    tXsX[None, None, None, producer_state.index],
                    is_async=True,
                )
                copy(
                    tXgdO[None, None, None, ahead_bidx],
                    tXsdO[None, None, None, producer_state.index],
                    is_async=True,
                )
            elif const_expr(tiler_mn[0] > 1):
                utils.fill_oob(
                    tXsX[None, None, None, producer_state.index],
                    None,
                    fill_value=mX.element_type.zero,
                )
                utils.fill_oob(
                    tXsdO[None, None, None, producer_state.index],
                    None,
                    fill_value=mdO.element_type.zero,
                )
            cute.arch.cp_async_commit_group()
            producer_state.advance()
        rstd = cutlass.Float.zero
        if row < M or tiler_mn[0] == 1:
            rstd = mRstd[row]
        if const_expr(mdResO is not None):
            if row < M or tiler_mn[0] == 1:
                copy(tXgdResO[None, None, None, bidx], tXrdResO)
            elif tiler_mn[0] > 1:
                tXrdResO.fill(0.0)
        if const_expr(USE_TMA):
            tma_pipeline.consumer_wait(consumer_state)
        else:
            cute.arch.cp_async_wait_group(const_expr(NUM_PIPE_STAGES - 1))
        smem_stage = consumer_state.index
        cute.autovec_copy(tXsX[None, None, None, smem_stage], tXrX)
        x = tXrX.load().to(cute.Float32)
        cute.autovec_copy(tXsdO[None, None, None, smem_stage], tXrdO)
        dout = tXrdO.load().to(cute.Float32)
        x_hat = x * rstd
        wdy = dout
        if const_expr(mW is not None):
            wdy *= tXrW.load().to(Float32)
        if const_expr(self.cluster_n > 1):
            cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)
        mean_xhat_wdy = (
            row_reduce(
                x_hat * wdy,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, stage],
                mbar_full_ptr + stage if const_expr(self.cluster_n > 1) else None,
                phase=consumer_phase,
                init_val=0.0,
                shard=shard,
            )
            / shape[1]
        )
        if const_expr(self.cluster_n > 1):
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            lane_idx = cute.arch.lane_idx()
            if lane_idx < self.cluster_n:
                cute.arch.mbarrier_arrive(mbar_empty_ptr + stage, peer_cta_rank_in_cluster=lane_idx)
        if const_expr(self.reload_wdy == "smem"):
            cute.autovec_copy(tXsdO[None, None, None, smem_stage], tXrdO)
            dout = tXrdO.load().to(cute.Float32)
            wdy = dout
            if const_expr(mW is not None):
                wdy *= tXrW.load().to(Float32)
        if const_expr(self.reload_x == "smem"):
            cute.autovec_copy(tXsX[None, None, None, smem_stage], tXrX)
            x = tXrX.load().to(cute.Float32)
            x_hat = x * rstd
        dx = (wdy - x_hat * mean_xhat_wdy) * rstd
        if const_expr(mdResO is not None):
            dx += tXrdResO.load().to(cute.Float32)
        tXrdX.store(dx.to(tXrdX.element_type))
        if row < M or tiler_mn[0] == 1:
            copy(tXrdX, tXgdX[None, None, None, bidx])
        if const_expr(mdRes is not None):
            tXrdRes.store(dx.to(tXrdRes.element_type))
            if row < M or tiler_mn[0] == 1:
                copy(tXrdRes, tXgdRes[None, None, None, bidx])
        if const_expr(mdW is not None):
            tXrdW.store(tXrdW.load() + dout * x_hat)
        if const_expr(mdB is not None):
            tXrdB.store(tXrdB.load() + dout)
        if const_expr(USE_TMA):
            tma_pipeline.sync_object_empty.arrive(consumer_state.index, tma_pipeline.consumer_mask)
        consumer_state.advance()
        stage ^= 1
        if stage == 0:
            consumer_phase ^= 1
            producer_phase ^= 1
    if const_expr(tiler_mn[0] > 1):
        if const_expr(mdW is not None):
            sdW = cute.make_tensor(
                cute.recast_ptr(sX.iterator, dtype=cute.Float32),
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            )
            tXsdW = thr_copy_X.partition_D(sdW)
            shard_barrier(shard)
            row = tXcX[None, None, None, 0][0][0]
            if row > 0:
                cute.autovec_copy(tXrdW, tXsdW)
            shard_barrier(shard)
            if row == 0:
                for i in range(1, const_expr(tiler_mn[0])):
                    tXrdW_other = cute.make_rmem_tensor_like(tXrdW)
                    tXsdW_other = cute.make_tensor(tXsdW.iterator + i * sdW.stride[0], tXsdW.layout)
                    cute.autovec_copy(tXsdW_other, tXrdW_other)
                    tXrdW.store(tXrdW.load() + tXrdW_other.load())
                copy(tXrdW, tXgdW)
            shard_barrier(shard)
        if const_expr(mdB is not None):
            sdB = cute.make_tensor(
                cute.recast_ptr(sX.iterator, dtype=cute.Float32),
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            )
            tXsdB = thr_copy_X.partition_D(sdB)
            shard_barrier(shard)
            row = tXcX[None, None, None, 0][0][0]
            if row > 0:
                cute.autovec_copy(tXrdB, tXsdB)
            shard_barrier(shard)
            if row == 0:
                for i in range(1, const_expr(tiler_mn[0])):
                    tXrdB_other = cute.make_rmem_tensor_like(tXrdB)
                    tXsdB_other = cute.make_tensor(tXsdB.iterator + i * sdB.stride[0], tXsdB.layout)
                    cute.autovec_copy(tXsdB_other, tXrdB_other)
                    tXrdB.store(tXrdB.load() + tXrdB_other.load())
                copy(tXrdB, tXgdB)
    else:
        if const_expr(mdW is not None):
            copy(tXrdW, tXgdW)
        if const_expr(mdB is not None):
            copy(tXrdB, tXgdB)
    if const_expr(self.cluster_n > 1):
        stage ^= 1
        if stage == 0:
            producer_phase ^= 1
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)


# The names training_program.py imports the two bodies under.
resident_rms_forward_body_h4096 = rmsnorm_forward_body
resident_rms_backward_body_h4096 = rmsnorm_backward_body
