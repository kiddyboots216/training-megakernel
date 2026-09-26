"""FA4's backward postprocess as per-tile task bodies of the program.

Adapted from `FlashAttentionBackwardPostprocess.kernel` in
flash_attn/cute/flash_bwd_postprocess.py (FlashAttention 4, commit 890f238).  Copyright (c)
2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao;
FlashAttention is distributed under the BSD 3-Clause License (see THIRD_PARTY_NOTICES.md).

The kernel becomes a task body: it takes the tile (`m_block`, `head_idx`, `batch_idx`) as
arguments instead of asking a tile scheduler, declares its global tensors 16-byte aligned,
synchronizes its 256 threads on named barrier 2 instead of the whole CTA, takes shared memory
from the program's page, and maps the physical threads 128..383 (roles 1 and 2) that run it
onto FA4's logical 0..255.

- `postprocess_task_body` is that task body; the program runs it for dQ.
- `postprocess_task_body_two_outputs` has one more output, `mdQMirror`, which receives the
  same BF16 values in the same final copy; the program runs it for dV, mirrored into the
  packed dQKV rows.
- `postprocess_task_body_direct_load` reads the FP32 accumulator tile straight from global
  memory into registers through the S2R copy (`cute.copy` with the S2R atom over a global
  view laid out like the shared tile), instead of a global to shared copy plus barrier
  followed by a shared to register copy; everything after the register fragment is
  unchanged. The program runs it for dK.

The annotations are evaluated when the module loads, as in FA4's own module.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import Float32, Int32, const_expr
from cutlass.utils import LayoutEnum
from flash_attn.cute import utils
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from quack import layout_utils

from program_smem import ProgramSmemAllocator


@cute.jit
def postprocess_task_body(
    self,
    mdQaccum: cute.Tensor,
    mdQ: cute.Tensor,
    mCuSeqlensQ: Optional[cute.Tensor],
    mSeqUsedQ: Optional[cute.Tensor],
    scale: cutlass.Float32,
    tiled_mma: cute.TiledMma,
    dQ_swapAB: cutlass.Constexpr,
    sdQaccum_layout: cute.Layout,
    sdQ_layout: cute.ComposedLayout,
    g2s_tiled_copy_dQaccum: cute.TiledCopy,
    s2r_tiled_copy_dQaccum: cute.TiledCopy,
    gmem_tiled_copy_dQ: cute.TiledCopy,
    m_block: Int32,
    head_idx: Int32,
    batch_idx: Int32,
):
    mdQaccum = cute.make_tensor(
        cute.make_ptr(
            dtype=mdQaccum.element_type,
            value=mdQaccum.iterator.toint(),
            mem_space=mdQaccum.iterator.memspace,
            assumed_align=16,
        ),
        mdQaccum.layout,
    )
    mdQaccum = assume_tensor_aligned(mdQaccum)
    mdQ = cute.make_tensor(
        cute.make_ptr(
            dtype=mdQ.element_type,
            value=mdQ.iterator.toint(),
            mem_space=mdQ.iterator.memspace,
            assumed_align=16,
        ),
        mdQ.layout,
    )
    mdQ = assume_tensor_aligned(mdQ)
    smem = ProgramSmemAllocator()
    sdQaccum = smem.allocate_tensor(cutlass.Float32, sdQaccum_layout, byte_alignment=1024)
    sdQaccum_flat = cute.make_tensor(sdQaccum.iterator, cute.make_layout(cute.size(sdQaccum)))
    if const_expr(self.arch // 10 in [8, 9, 12]):
        sdQ = cute.make_tensor(cute.recast_ptr(sdQaccum.iterator, dtype=self.dtype), sdQ_layout)
    else:
        sdQ = cute.make_tensor(
            cute.recast_ptr(sdQaccum.iterator, sdQ_layout.inner, dtype=self.dtype), sdQ_layout.outer
        )[None, None, 0]
    sdQt = layout_utils.transpose_view(sdQ)
    tidx, _, _ = cute.arch.thread_idx()
    tidx = tidx - Int32(128)
    seqlen = SeqlenInfoQK.create(
        batch_idx,
        mdQ.shape[1],
        0,
        mCuSeqlensQ=mCuSeqlensQ,
        mCuSeqlensK=None,
        mSeqUsedQ=mSeqUsedQ,
        mSeqUsedK=None,
        tile_m=self.tile_m * self.cluster_size,
    )
    if const_expr(not seqlen.has_cu_seqlens_q):
        mdQ_cur = mdQ[batch_idx, None, head_idx, None]
        mdQaccum_cur = mdQaccum[batch_idx, head_idx, None]
        head_dim = mdQ.shape[3]
    else:
        padded_offset_q = seqlen.padded_offset_q
        mdQ_cur = cute.domain_offset((seqlen.offset_q, 0), mdQ[None, head_idx, None])
        mdQaccum_cur = cute.domain_offset(
            (padded_offset_q * self.tile_hdim,), mdQaccum[head_idx, None]
        )
        head_dim = mdQ.shape[2]
        mdQaccum_cur_ptr = cute.make_ptr(
            dtype=mdQaccum_cur.element_type,
            value=mdQaccum_cur.iterator.toint(),
            mem_space=mdQaccum_cur.iterator.memspace,
            assumed_align=mdQaccum.iterator.alignment,
        )
        mdQaccum_cur = cute.make_tensor(mdQaccum_cur_ptr, mdQaccum_cur.layout)
    gdQaccum = cute.local_tile(mdQaccum_cur, (self.tile_m * self.tile_hdim,), (m_block,))
    gdQ = cute.local_tile(mdQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
    seqlen_q = seqlen.seqlen_q
    seqlen_q_rounded = cute.round_up(seqlen_q, self.tile_m)
    if const_expr(self.arch // 10 in [10, 11] and self.use_2cta_instrs):
        num_reduce_threads = self.num_threads
        thr_mma_dsk = tiled_mma.get_slice(tidx)
        dQacc_shape = thr_mma_dsk.partition_shape_C((self.tile_m, self.tile_hdim))
        tdQtdQ = thr_mma_dsk.make_fragment_C(dQacc_shape)
        tdQtdQ = cute.make_tensor(tdQtdQ.iterator, tdQtdQ.layout)
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol)), Float32
        )
        tiled_tmem_ld = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ)
        thr_tmem_ld = tiled_tmem_ld.get_slice(tidx)
        cdQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
        tdQcdQ = thr_mma_dsk.partition_C(cdQ)
        tdQcdQ_tensor = cute.make_tensor(tdQcdQ.iterator, tdQcdQ.layout)
        tdQrdQ = thr_tmem_ld.partition_D(tdQcdQ_tensor)
        tiled_copy_accum = s2r_tiled_copy_dQaccum
        g2s_thr_copy = tiled_copy_accum.get_slice(tidx)
        tdQrdQ_fp32 = cute.make_rmem_tensor(tdQrdQ.shape, cutlass.Float32)
        tdQrdQ_s2r = cute.make_tensor(tdQrdQ_fp32.iterator, tdQrdQ_fp32.shape)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, self.dtype, cutlass.Float32, tiled_tmem_ld
        )
        r2s_tiled_copy = cute.make_tiled_copy(
            smem_copy_atom,
            layout_tv=tiled_tmem_ld.layout_dst_tv_tiled,
            tiler_mn=tiled_tmem_ld.tiler_mn,
        )
        tdQsdQ_r2s = thr_tmem_ld.partition_D(thr_mma_dsk.partition_C(sdQ))
        tdQrdQ_r2s = cute.make_rmem_tensor(tdQsdQ_r2s.shape, self.dtype)
        num_stages = cute.size(tdQrdQ_fp32, mode=[1])
        stage_stride = self.dQ_reduce_ncol
        row_groups = 2
        assert num_stages % row_groups == 0
        assert num_reduce_threads % row_groups == 0
        stage_groups = num_stages // row_groups
        threads_per_row_group = num_reduce_threads // row_groups
        stage_loads = tuple(((row_group, row_group) for row_group in range(row_groups)))
        stage_iters = tuple(
            ((row_group, row_group * threads_per_row_group) for row_group in range(row_groups))
        )
        s2r_lane = tidx % threads_per_row_group
        s2r_buf = tidx // threads_per_row_group
        gdQaccum_layout_g2s = cute.make_layout(
            shape=(self.tile_m * self.dQ_reduce_ncol, 1), stride=(1, 0)
        )
        sdQaccum_g2s = g2s_thr_copy.partition_D(sdQaccum)
        for stage_group in cutlass.range_constexpr(stage_groups):
            for stage_offset, smem_buf in stage_loads:
                stage_idx = stage_group + stage_offset * stage_groups
                gdQaccum_stage = cute.local_tile(
                    gdQaccum, (self.tile_m * self.dQ_reduce_ncol,), (stage_idx,)
                )
                gdQaccum_stage_g2s = cute.make_tensor(gdQaccum_stage.iterator, gdQaccum_layout_g2s)
                tdQgdQ = g2s_thr_copy.partition_S(gdQaccum_stage_g2s)
                cute.copy(g2s_thr_copy, tdQgdQ[None, None, 0], sdQaccum_g2s[None, None, smem_buf])
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=2, number_of_threads=256)
            for stage_offset, lane_offset in stage_iters:
                stage_idx = stage_group + stage_offset * stage_groups
                s2r_src_tidx = s2r_lane + lane_offset
                s2r_thr_copy = tiled_copy_accum.get_slice(s2r_src_tidx)
                sdQaccum_src = s2r_thr_copy.partition_S(sdQaccum)[None, None, s2r_buf]
                tdQrdQ_s2r_cpy = tdQrdQ_s2r[None, stage_idx, None, None]
                tdQrdQ_r2s_cpy = cute.make_tensor(
                    tdQrdQ_s2r_cpy.iterator, cute.make_layout(sdQaccum_src.shape)
                )
                cute.copy(s2r_thr_copy, sdQaccum_src, tdQrdQ_r2s_cpy)
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(barrier_id=2, number_of_threads=256)
                stage_lo = stage_idx % stage_stride
                stage_hi = stage_idx // stage_stride
                tdQrdQ_r2s_cpy = cute.make_tensor(
                    cute.recast_ptr(tdQrdQ_r2s_cpy.iterator),
                    tdQrdQ_r2s[(None, 0), (stage_lo, stage_hi), 0, 0].shape,
                )
                dQ_vec = tdQrdQ_r2s_cpy.load() * scale
                tdQrdQ_r2s[(None, 0), (stage_lo, stage_hi), 0, 0].store(dQ_vec.to(self.dtype))
        cute.copy(r2s_tiled_copy, tdQrdQ_r2s[None, None, None, 0], tdQsdQ_r2s[None, None, None, 0])
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(barrier_id=2, number_of_threads=256)
    else:
        g2s_thr_copy_dQaccum = g2s_tiled_copy_dQaccum.get_slice(tidx)
        tdQgdQaccum = g2s_thr_copy_dQaccum.partition_S(gdQaccum)
        tdQsdQaccumg2s = g2s_thr_copy_dQaccum.partition_D(sdQaccum_flat)
        cute.copy(g2s_tiled_copy_dQaccum, tdQgdQaccum, tdQsdQaccumg2s)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier(barrier_id=2, number_of_threads=256)
        s2r_thr_copy_dQaccum = s2r_tiled_copy_dQaccum.get_slice(tidx)
        tdQsdQaccum = s2r_thr_copy_dQaccum.partition_S(sdQaccum)
        tile_shape = (self.tile_m, self.tile_hdim)
        acc = None
        tiled_copy_t2r = None
        if const_expr(self.arch // 10 in [8, 9, 12]):
            acc_shape = tiled_mma.partition_shape_C(
                tile_shape if const_expr(not dQ_swapAB) else tile_shape[::-1]
            )
            acc = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
            assert cute.size(acc) == cute.size(tdQsdQaccum)
        else:
            thr_mma = tiled_mma.get_slice(0)
            dQacc_shape = tiled_mma.partition_shape_C((self.tile_m, self.tile_hdim))
            tdQtdQ = tiled_mma.make_fragment_C(dQacc_shape)
            tdQcdQ = thr_mma.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_hdim)))
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol)), Float32
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ)
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tdQrdQ_t2r_shape = thr_copy_t2r.partition_D(tdQcdQ).shape
            acc = cute.make_rmem_tensor(tdQrdQ_t2r_shape, Float32)
        tdQrdQaccum = cute.make_tensor(acc.iterator, cute.make_layout(tdQsdQaccum.shape))
        cute.autovec_copy(tdQsdQaccum, tdQrdQaccum)
        rdQ = cute.make_fragment_like(acc, self.dtype)
        rdQ.store((acc.load() * scale).to(self.dtype))
        cute.arch.barrier(barrier_id=2, number_of_threads=256)
        if const_expr(self.arch // 10 in [8, 9, 12]):
            copy_atom_r2s_dQ = utils.get_smem_store_atom(
                self.arch, self.dtype, transpose=self.dQ_swapAB
            )
            tiled_copy_r2s_dQ = cute.make_tiled_copy_C(copy_atom_r2s_dQ, tiled_mma)
        else:
            thr_layout_r2s_dQ = cute.make_layout((self.num_threads, 1))
            val_layout_r2s_dQ = cute.make_layout((1, 128 // self.dtype.width))
            copy_atom_r2s_dQ = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=128
            )
            tiled_copy_r2s_dQ = cute.make_tiled_copy_tv(
                copy_atom_r2s_dQ, thr_layout_r2s_dQ, val_layout_r2s_dQ
            )
        thr_copy_r2s_dQ = tiled_copy_r2s_dQ.get_slice(tidx)
        cdQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
        if const_expr(self.arch // 10 in [8, 9, 12]):
            taccdQrdQ = thr_copy_r2s_dQ.retile(rdQ)
        else:
            taccdQcdQ_shape = thr_copy_r2s_dQ.partition_S(cdQ).shape
            taccdQrdQ = cute.make_tensor(rdQ.iterator, taccdQcdQ_shape)
        taccdQsdQ = thr_copy_r2s_dQ.partition_D(sdQ if const_expr(not self.dQ_swapAB) else sdQt)
        cute.copy(thr_copy_r2s_dQ, taccdQrdQ, taccdQsdQ)
    cute.arch.barrier(barrier_id=2, number_of_threads=256)
    gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_slice(tidx)
    tdQgdQ = gmem_thr_copy_dQ.partition_S(gdQ)
    tdQsdQ = gmem_thr_copy_dQ.partition_D(sdQ)
    tdQrdQ = cute.make_fragment_like(tdQsdQ, self.dtype)
    cute.autovec_copy(tdQsdQ, tdQrdQ)
    tdQcdQ = gmem_thr_copy_dQ.partition_S(cdQ)
    tdQpdQ = utils.predicate_k(tdQcdQ, limit=head_dim)
    for rest_m in cutlass.range(cute.size(tdQrdQ.shape[1]), unroll_full=True):
        if tdQcdQ[0, rest_m, 0][0] < seqlen_q - m_block * self.tile_m:
            cute.copy(
                gmem_tiled_copy_dQ,
                tdQrdQ[None, rest_m, None],
                tdQgdQ[None, rest_m, None],
                pred=tdQpdQ[None, rest_m, None],
            )


@cute.jit
def postprocess_task_body_two_outputs(
    self,
    mdQaccum: cute.Tensor,
    mdQ: cute.Tensor,
    mdQMirror: cute.Tensor,
    mCuSeqlensQ: Optional[cute.Tensor],
    mSeqUsedQ: Optional[cute.Tensor],
    scale: cutlass.Float32,
    tiled_mma: cute.TiledMma,
    dQ_swapAB: cutlass.Constexpr,
    sdQaccum_layout: cute.Layout,
    sdQ_layout: cute.ComposedLayout,
    g2s_tiled_copy_dQaccum: cute.TiledCopy,
    s2r_tiled_copy_dQaccum: cute.TiledCopy,
    gmem_tiled_copy_dQ: cute.TiledCopy,
    m_block: Int32,
    head_idx: Int32,
    batch_idx: Int32,
):
    mdQaccum = cute.make_tensor(
        cute.make_ptr(
            dtype=mdQaccum.element_type,
            value=mdQaccum.iterator.toint(),
            mem_space=mdQaccum.iterator.memspace,
            assumed_align=16,
        ),
        mdQaccum.layout,
    )
    mdQaccum = assume_tensor_aligned(mdQaccum)
    mdQ = cute.make_tensor(
        cute.make_ptr(
            dtype=mdQ.element_type,
            value=mdQ.iterator.toint(),
            mem_space=mdQ.iterator.memspace,
            assumed_align=16,
        ),
        mdQ.layout,
    )
    mdQMirror = cute.make_tensor(
        cute.make_ptr(
            dtype=mdQMirror.element_type,
            value=mdQMirror.iterator.toint(),
            mem_space=mdQMirror.iterator.memspace,
            assumed_align=16,
        ),
        mdQMirror.layout,
    )
    mdQ = assume_tensor_aligned(mdQ)
    mdQMirror = assume_tensor_aligned(mdQMirror)
    smem = ProgramSmemAllocator()
    sdQaccum = smem.allocate_tensor(cutlass.Float32, sdQaccum_layout, byte_alignment=1024)
    sdQaccum_flat = cute.make_tensor(sdQaccum.iterator, cute.make_layout(cute.size(sdQaccum)))
    if const_expr(self.arch // 10 in [8, 9, 12]):
        sdQ = cute.make_tensor(cute.recast_ptr(sdQaccum.iterator, dtype=self.dtype), sdQ_layout)
    else:
        sdQ = cute.make_tensor(
            cute.recast_ptr(sdQaccum.iterator, sdQ_layout.inner, dtype=self.dtype), sdQ_layout.outer
        )[None, None, 0]
    sdQt = layout_utils.transpose_view(sdQ)
    tidx, _, _ = cute.arch.thread_idx()
    tidx = tidx - Int32(128)
    seqlen = SeqlenInfoQK.create(
        batch_idx,
        mdQ.shape[1],
        0,
        mCuSeqlensQ=mCuSeqlensQ,
        mCuSeqlensK=None,
        mSeqUsedQ=mSeqUsedQ,
        mSeqUsedK=None,
        tile_m=self.tile_m * self.cluster_size,
    )
    if const_expr(not seqlen.has_cu_seqlens_q):
        mdQ_cur = mdQ[batch_idx, None, head_idx, None]
        mdQMirror_cur = mdQMirror[batch_idx, None, head_idx, None]
        mdQaccum_cur = mdQaccum[batch_idx, head_idx, None]
        head_dim = mdQ.shape[3]
    else:
        padded_offset_q = seqlen.padded_offset_q
        mdQ_cur = cute.domain_offset((seqlen.offset_q, 0), mdQ[None, head_idx, None])
        mdQMirror_cur = cute.domain_offset((seqlen.offset_q, 0), mdQMirror[None, head_idx, None])
        mdQaccum_cur = cute.domain_offset(
            (padded_offset_q * self.tile_hdim,), mdQaccum[head_idx, None]
        )
        head_dim = mdQ.shape[2]
        mdQaccum_cur_ptr = cute.make_ptr(
            dtype=mdQaccum_cur.element_type,
            value=mdQaccum_cur.iterator.toint(),
            mem_space=mdQaccum_cur.iterator.memspace,
            assumed_align=mdQaccum.iterator.alignment,
        )
        mdQaccum_cur = cute.make_tensor(mdQaccum_cur_ptr, mdQaccum_cur.layout)
    gdQaccum = cute.local_tile(mdQaccum_cur, (self.tile_m * self.tile_hdim,), (m_block,))
    gdQ = cute.local_tile(mdQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
    gdQMirror = cute.local_tile(mdQMirror_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
    seqlen_q = seqlen.seqlen_q
    seqlen_q_rounded = cute.round_up(seqlen_q, self.tile_m)
    if const_expr(self.arch // 10 in [10, 11] and self.use_2cta_instrs):
        num_reduce_threads = self.num_threads
        thr_mma_dsk = tiled_mma.get_slice(tidx)
        dQacc_shape = thr_mma_dsk.partition_shape_C((self.tile_m, self.tile_hdim))
        tdQtdQ = thr_mma_dsk.make_fragment_C(dQacc_shape)
        tdQtdQ = cute.make_tensor(tdQtdQ.iterator, tdQtdQ.layout)
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol)), Float32
        )
        tiled_tmem_ld = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ)
        thr_tmem_ld = tiled_tmem_ld.get_slice(tidx)
        cdQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
        tdQcdQ = thr_mma_dsk.partition_C(cdQ)
        tdQcdQ_tensor = cute.make_tensor(tdQcdQ.iterator, tdQcdQ.layout)
        tdQrdQ = thr_tmem_ld.partition_D(tdQcdQ_tensor)
        tiled_copy_accum = s2r_tiled_copy_dQaccum
        g2s_thr_copy = tiled_copy_accum.get_slice(tidx)
        tdQrdQ_fp32 = cute.make_rmem_tensor(tdQrdQ.shape, cutlass.Float32)
        tdQrdQ_s2r = cute.make_tensor(tdQrdQ_fp32.iterator, tdQrdQ_fp32.shape)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, self.dtype, cutlass.Float32, tiled_tmem_ld
        )
        r2s_tiled_copy = cute.make_tiled_copy(
            smem_copy_atom,
            layout_tv=tiled_tmem_ld.layout_dst_tv_tiled,
            tiler_mn=tiled_tmem_ld.tiler_mn,
        )
        tdQsdQ_r2s = thr_tmem_ld.partition_D(thr_mma_dsk.partition_C(sdQ))
        tdQrdQ_r2s = cute.make_rmem_tensor(tdQsdQ_r2s.shape, self.dtype)
        num_stages = cute.size(tdQrdQ_fp32, mode=[1])
        stage_stride = self.dQ_reduce_ncol
        row_groups = 2
        assert num_stages % row_groups == 0
        assert num_reduce_threads % row_groups == 0
        stage_groups = num_stages // row_groups
        threads_per_row_group = num_reduce_threads // row_groups
        stage_loads = tuple(((row_group, row_group) for row_group in range(row_groups)))
        stage_iters = tuple(
            ((row_group, row_group * threads_per_row_group) for row_group in range(row_groups))
        )
        s2r_lane = tidx % threads_per_row_group
        s2r_buf = tidx // threads_per_row_group
        gdQaccum_layout_g2s = cute.make_layout(
            shape=(self.tile_m * self.dQ_reduce_ncol, 1), stride=(1, 0)
        )
        sdQaccum_g2s = g2s_thr_copy.partition_D(sdQaccum)
        for stage_group in cutlass.range_constexpr(stage_groups):
            for stage_offset, smem_buf in stage_loads:
                stage_idx = stage_group + stage_offset * stage_groups
                gdQaccum_stage = cute.local_tile(
                    gdQaccum, (self.tile_m * self.dQ_reduce_ncol,), (stage_idx,)
                )
                gdQaccum_stage_g2s = cute.make_tensor(gdQaccum_stage.iterator, gdQaccum_layout_g2s)
                tdQgdQ = g2s_thr_copy.partition_S(gdQaccum_stage_g2s)
                cute.copy(g2s_thr_copy, tdQgdQ[None, None, 0], sdQaccum_g2s[None, None, smem_buf])
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=2, number_of_threads=256)
            for stage_offset, lane_offset in stage_iters:
                stage_idx = stage_group + stage_offset * stage_groups
                s2r_src_tidx = s2r_lane + lane_offset
                s2r_thr_copy = tiled_copy_accum.get_slice(s2r_src_tidx)
                sdQaccum_src = s2r_thr_copy.partition_S(sdQaccum)[None, None, s2r_buf]
                tdQrdQ_s2r_cpy = tdQrdQ_s2r[None, stage_idx, None, None]
                tdQrdQ_r2s_cpy = cute.make_tensor(
                    tdQrdQ_s2r_cpy.iterator, cute.make_layout(sdQaccum_src.shape)
                )
                cute.copy(s2r_thr_copy, sdQaccum_src, tdQrdQ_r2s_cpy)
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(barrier_id=2, number_of_threads=256)
                stage_lo = stage_idx % stage_stride
                stage_hi = stage_idx // stage_stride
                tdQrdQ_r2s_cpy = cute.make_tensor(
                    cute.recast_ptr(tdQrdQ_r2s_cpy.iterator),
                    tdQrdQ_r2s[(None, 0), (stage_lo, stage_hi), 0, 0].shape,
                )
                dQ_vec = tdQrdQ_r2s_cpy.load() * scale
                tdQrdQ_r2s[(None, 0), (stage_lo, stage_hi), 0, 0].store(dQ_vec.to(self.dtype))
        cute.copy(r2s_tiled_copy, tdQrdQ_r2s[None, None, None, 0], tdQsdQ_r2s[None, None, None, 0])
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(barrier_id=2, number_of_threads=256)
    else:
        g2s_thr_copy_dQaccum = g2s_tiled_copy_dQaccum.get_slice(tidx)
        tdQgdQaccum = g2s_thr_copy_dQaccum.partition_S(gdQaccum)
        tdQsdQaccumg2s = g2s_thr_copy_dQaccum.partition_D(sdQaccum_flat)
        cute.copy(g2s_tiled_copy_dQaccum, tdQgdQaccum, tdQsdQaccumg2s)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier(barrier_id=2, number_of_threads=256)
        s2r_thr_copy_dQaccum = s2r_tiled_copy_dQaccum.get_slice(tidx)
        tdQsdQaccum = s2r_thr_copy_dQaccum.partition_S(sdQaccum)
        tile_shape = (self.tile_m, self.tile_hdim)
        acc = None
        tiled_copy_t2r = None
        if const_expr(self.arch // 10 in [8, 9, 12]):
            acc_shape = tiled_mma.partition_shape_C(
                tile_shape if const_expr(not dQ_swapAB) else tile_shape[::-1]
            )
            acc = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
            assert cute.size(acc) == cute.size(tdQsdQaccum)
        else:
            thr_mma = tiled_mma.get_slice(0)
            dQacc_shape = tiled_mma.partition_shape_C((self.tile_m, self.tile_hdim))
            tdQtdQ = tiled_mma.make_fragment_C(dQacc_shape)
            tdQcdQ = thr_mma.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_hdim)))
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol)), Float32
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ)
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tdQrdQ_t2r_shape = thr_copy_t2r.partition_D(tdQcdQ).shape
            acc = cute.make_rmem_tensor(tdQrdQ_t2r_shape, Float32)
        tdQrdQaccum = cute.make_tensor(acc.iterator, cute.make_layout(tdQsdQaccum.shape))
        cute.autovec_copy(tdQsdQaccum, tdQrdQaccum)
        rdQ = cute.make_fragment_like(acc, self.dtype)
        rdQ.store((acc.load() * scale).to(self.dtype))
        cute.arch.barrier(barrier_id=2, number_of_threads=256)
        if const_expr(self.arch // 10 in [8, 9, 12]):
            copy_atom_r2s_dQ = utils.get_smem_store_atom(
                self.arch, self.dtype, transpose=self.dQ_swapAB
            )
            tiled_copy_r2s_dQ = cute.make_tiled_copy_C(copy_atom_r2s_dQ, tiled_mma)
        else:
            thr_layout_r2s_dQ = cute.make_layout((self.num_threads, 1))
            val_layout_r2s_dQ = cute.make_layout((1, 128 // self.dtype.width))
            copy_atom_r2s_dQ = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=128
            )
            tiled_copy_r2s_dQ = cute.make_tiled_copy_tv(
                copy_atom_r2s_dQ, thr_layout_r2s_dQ, val_layout_r2s_dQ
            )
        thr_copy_r2s_dQ = tiled_copy_r2s_dQ.get_slice(tidx)
        cdQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
        if const_expr(self.arch // 10 in [8, 9, 12]):
            taccdQrdQ = thr_copy_r2s_dQ.retile(rdQ)
        else:
            taccdQcdQ_shape = thr_copy_r2s_dQ.partition_S(cdQ).shape
            taccdQrdQ = cute.make_tensor(rdQ.iterator, taccdQcdQ_shape)
        taccdQsdQ = thr_copy_r2s_dQ.partition_D(sdQ if const_expr(not self.dQ_swapAB) else sdQt)
        cute.copy(thr_copy_r2s_dQ, taccdQrdQ, taccdQsdQ)
    cute.arch.barrier(barrier_id=2, number_of_threads=256)
    gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_slice(tidx)
    tdQgdQ = gmem_thr_copy_dQ.partition_S(gdQ)
    tdQgdQMirror = gmem_thr_copy_dQ.partition_S(gdQMirror)
    tdQsdQ = gmem_thr_copy_dQ.partition_D(sdQ)
    tdQrdQ = cute.make_fragment_like(tdQsdQ, self.dtype)
    cute.autovec_copy(tdQsdQ, tdQrdQ)
    tdQcdQ = gmem_thr_copy_dQ.partition_S(cdQ)
    tdQpdQ = utils.predicate_k(tdQcdQ, limit=head_dim)
    for rest_m in cutlass.range(cute.size(tdQrdQ.shape[1]), unroll_full=True):
        if tdQcdQ[0, rest_m, 0][0] < seqlen_q - m_block * self.tile_m:
            cute.copy(
                gmem_tiled_copy_dQ,
                tdQrdQ[None, rest_m, None],
                tdQgdQ[None, rest_m, None],
                pred=tdQpdQ[None, rest_m, None],
            )
            cute.copy(
                gmem_tiled_copy_dQ,
                tdQrdQ[None, rest_m, None],
                tdQgdQMirror[None, rest_m, None],
                pred=tdQpdQ[None, rest_m, None],
            )


# Derived from flash_attn/cute/flash_bwd_postprocess.py in FlashAttention 4
# (github.com/Dao-AILab/flash-attention, commit 890f238), which is
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
# Pradeep Ramani, Tri Dao, and distributed under the BSD 3-Clause License.
@cute.jit
def postprocess_task_body_direct_load(self, mdQaccum: cute.Tensor, mdQ: cute.Tensor, mCuSeqlensQ: Optional[cute.Tensor], mSeqUsedQ: Optional[cute.Tensor], scale: cutlass.Float32, tiled_mma: cute.TiledMma, dQ_swapAB: cutlass.Constexpr, sdQaccum_layout: cute.Layout, sdQ_layout: cute.ComposedLayout, g2s_tiled_copy_dQaccum: cute.TiledCopy, s2r_tiled_copy_dQaccum: cute.TiledCopy, gmem_tiled_copy_dQ: cute.TiledCopy, m_block: Int32, head_idx: Int32, batch_idx: Int32):
    mdQaccum = cute.make_tensor(cute.make_ptr(dtype=mdQaccum.element_type, value=mdQaccum.iterator.toint(), mem_space=mdQaccum.iterator.memspace, assumed_align=16), mdQaccum.layout)
    mdQaccum = assume_tensor_aligned(mdQaccum)
    mdQ = cute.make_tensor(cute.make_ptr(dtype=mdQ.element_type, value=mdQ.iterator.toint(), mem_space=mdQ.iterator.memspace, assumed_align=16), mdQ.layout)
    mdQ = assume_tensor_aligned(mdQ)
    smem = ProgramSmemAllocator()
    sdQaccum = smem.allocate_tensor(cutlass.Float32, sdQaccum_layout, byte_alignment=1024)
    sdQaccum_flat = cute.make_tensor(sdQaccum.iterator, cute.make_layout(cute.size(sdQaccum)))
    if const_expr(self.arch // 10 in [8, 9, 12]):
        sdQ = cute.make_tensor(cute.recast_ptr(sdQaccum.iterator, dtype=self.dtype), sdQ_layout)
    else:
        sdQ = cute.make_tensor(cute.recast_ptr(sdQaccum.iterator, sdQ_layout.inner, dtype=self.dtype), sdQ_layout.outer)[None, None, 0]
    sdQt = layout_utils.transpose_view(sdQ)
    tidx, _, _ = cute.arch.thread_idx()
    tidx = tidx - Int32(128)
    seqlen = SeqlenInfoQK.create(batch_idx, mdQ.shape[1], 0, mCuSeqlensQ=mCuSeqlensQ, mCuSeqlensK=None, mSeqUsedQ=mSeqUsedQ, mSeqUsedK=None, tile_m=self.tile_m * self.cluster_size)
    if const_expr(not seqlen.has_cu_seqlens_q):
        mdQ_cur = mdQ[batch_idx, None, head_idx, None]
        mdQaccum_cur = mdQaccum[batch_idx, head_idx, None]
        head_dim = mdQ.shape[3]
    else:
        padded_offset_q = seqlen.padded_offset_q
        mdQ_cur = cute.domain_offset((seqlen.offset_q, 0), mdQ[None, head_idx, None])
        mdQaccum_cur = cute.domain_offset((padded_offset_q * self.tile_hdim,), mdQaccum[head_idx, None])
        head_dim = mdQ.shape[2]
        mdQaccum_cur_ptr = cute.make_ptr(dtype=mdQaccum_cur.element_type, value=mdQaccum_cur.iterator.toint(), mem_space=mdQaccum_cur.iterator.memspace, assumed_align=mdQaccum.iterator.alignment)
        mdQaccum_cur = cute.make_tensor(mdQaccum_cur_ptr, mdQaccum_cur.layout)
    gdQaccum = cute.local_tile(mdQaccum_cur, (self.tile_m * self.tile_hdim,), (m_block,))
    gdQ = cute.local_tile(mdQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
    seqlen_q = seqlen.seqlen_q
    seqlen_q_rounded = cute.round_up(seqlen_q, self.tile_m)
    if const_expr(self.arch // 10 in [10, 11] and self.use_2cta_instrs):
        num_reduce_threads = self.num_threads
        thr_mma_dsk = tiled_mma.get_slice(tidx)
        dQacc_shape = thr_mma_dsk.partition_shape_C((self.tile_m, self.tile_hdim))
        tdQtdQ = thr_mma_dsk.make_fragment_C(dQacc_shape)
        tdQtdQ = cute.make_tensor(tdQtdQ.iterator, tdQtdQ.layout)
        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol)), Float32)
        tiled_tmem_ld = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ)
        thr_tmem_ld = tiled_tmem_ld.get_slice(tidx)
        cdQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
        tdQcdQ = thr_mma_dsk.partition_C(cdQ)
        tdQcdQ_tensor = cute.make_tensor(tdQcdQ.iterator, tdQcdQ.layout)
        tdQrdQ = thr_tmem_ld.partition_D(tdQcdQ_tensor)
        tiled_copy_accum = s2r_tiled_copy_dQaccum
        g2s_thr_copy = tiled_copy_accum.get_slice(tidx)
        tdQrdQ_fp32 = cute.make_rmem_tensor(tdQrdQ.shape, cutlass.Float32)
        tdQrdQ_s2r = cute.make_tensor(tdQrdQ_fp32.iterator, tdQrdQ_fp32.shape)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(LayoutEnum.ROW_MAJOR, self.dtype, cutlass.Float32, tiled_tmem_ld)
        r2s_tiled_copy = cute.make_tiled_copy(smem_copy_atom, layout_tv=tiled_tmem_ld.layout_dst_tv_tiled, tiler_mn=tiled_tmem_ld.tiler_mn)
        tdQsdQ_r2s = thr_tmem_ld.partition_D(thr_mma_dsk.partition_C(sdQ))
        tdQrdQ_r2s = cute.make_rmem_tensor(tdQsdQ_r2s.shape, self.dtype)
        num_stages = cute.size(tdQrdQ_fp32, mode=[1])
        stage_stride = self.dQ_reduce_ncol
        row_groups = 2
        assert num_stages % row_groups == 0
        assert num_reduce_threads % row_groups == 0
        stage_groups = num_stages // row_groups
        threads_per_row_group = num_reduce_threads // row_groups
        stage_loads = tuple(((row_group, row_group) for row_group in range(row_groups)))
        stage_iters = tuple(((row_group, row_group * threads_per_row_group) for row_group in range(row_groups)))
        s2r_lane = tidx % threads_per_row_group
        s2r_buf = tidx // threads_per_row_group
        gdQaccum_layout_g2s = cute.make_layout(shape=(self.tile_m * self.dQ_reduce_ncol, 1), stride=(1, 0))
        sdQaccum_g2s = g2s_thr_copy.partition_D(sdQaccum)
        for stage_group in cutlass.range_constexpr(stage_groups):
            for stage_offset, smem_buf in stage_loads:
                stage_idx = stage_group + stage_offset * stage_groups
                gdQaccum_stage = cute.local_tile(gdQaccum, (self.tile_m * self.dQ_reduce_ncol,), (stage_idx,))
                gdQaccum_stage_g2s = cute.make_tensor(gdQaccum_stage.iterator, gdQaccum_layout_g2s)
                tdQgdQ = g2s_thr_copy.partition_S(gdQaccum_stage_g2s)
                cute.copy(g2s_thr_copy, tdQgdQ[None, None, 0], sdQaccum_g2s[None, None, smem_buf])
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=2, number_of_threads=256)
            for stage_offset, lane_offset in stage_iters:
                stage_idx = stage_group + stage_offset * stage_groups
                s2r_src_tidx = s2r_lane + lane_offset
                s2r_thr_copy = tiled_copy_accum.get_slice(s2r_src_tidx)
                sdQaccum_src = s2r_thr_copy.partition_S(sdQaccum)[None, None, s2r_buf]
                tdQrdQ_s2r_cpy = tdQrdQ_s2r[None, stage_idx, None, None]
                tdQrdQ_r2s_cpy = cute.make_tensor(tdQrdQ_s2r_cpy.iterator, cute.make_layout(sdQaccum_src.shape))
                cute.copy(s2r_thr_copy, sdQaccum_src, tdQrdQ_r2s_cpy)
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(barrier_id=2, number_of_threads=256)
                stage_lo = stage_idx % stage_stride
                stage_hi = stage_idx // stage_stride
                tdQrdQ_r2s_cpy = cute.make_tensor(cute.recast_ptr(tdQrdQ_r2s_cpy.iterator), tdQrdQ_r2s[(None, 0), (stage_lo, stage_hi), 0, 0].shape)
                dQ_vec = tdQrdQ_r2s_cpy.load() * scale
                tdQrdQ_r2s[(None, 0), (stage_lo, stage_hi), 0, 0].store(dQ_vec.to(self.dtype))
        cute.copy(r2s_tiled_copy, tdQrdQ_r2s[None, None, None, 0], tdQsdQ_r2s[None, None, None, 0])
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(barrier_id=2, number_of_threads=256)
    else:
        s2r_thr_copy_dQaccum = s2r_tiled_copy_dQaccum.get_slice(tidx)
        gdQaccum_canonical = cute.make_tensor(gdQaccum.iterator, sdQaccum_layout)
        tdQgdQaccum_canonical = s2r_thr_copy_dQaccum.partition_S(gdQaccum_canonical)
        tile_shape = (self.tile_m, self.tile_hdim)
        acc = None
        tiled_copy_t2r = None
        if const_expr(self.arch // 10 in [8, 9, 12]):
            acc_shape = tiled_mma.partition_shape_C(tile_shape if const_expr(not dQ_swapAB) else tile_shape[::-1])
            acc = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
            assert cute.size(acc) == cute.size(tdQgdQaccum_canonical)
        else:
            thr_mma = tiled_mma.get_slice(0)
            dQacc_shape = tiled_mma.partition_shape_C((self.tile_m, self.tile_hdim))
            tdQtdQ = tiled_mma.make_fragment_C(dQacc_shape)
            tdQcdQ = thr_mma.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_hdim)))
            tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol)), Float32)
            tiled_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ)
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tdQrdQ_t2r_shape = thr_copy_t2r.partition_D(tdQcdQ).shape
            acc = cute.make_rmem_tensor(tdQrdQ_t2r_shape, Float32)
        tdQrdQaccum = cute.make_tensor(acc.iterator, cute.make_layout(tdQgdQaccum_canonical.shape))
        cute.copy(s2r_thr_copy_dQaccum, tdQgdQaccum_canonical, tdQrdQaccum)
        rdQ = cute.make_fragment_like(acc, self.dtype)
        rdQ.store((acc.load() * scale).to(self.dtype))
        cute.arch.barrier(barrier_id=2, number_of_threads=256)
        if const_expr(self.arch // 10 in [8, 9, 12]):
            copy_atom_r2s_dQ = utils.get_smem_store_atom(self.arch, self.dtype, transpose=self.dQ_swapAB)
            tiled_copy_r2s_dQ = cute.make_tiled_copy_C(copy_atom_r2s_dQ, tiled_mma)
        else:
            thr_layout_r2s_dQ = cute.make_layout((self.num_threads, 1))
            val_layout_r2s_dQ = cute.make_layout((1, 128 // self.dtype.width))
            copy_atom_r2s_dQ = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=128)
            tiled_copy_r2s_dQ = cute.make_tiled_copy_tv(copy_atom_r2s_dQ, thr_layout_r2s_dQ, val_layout_r2s_dQ)
        thr_copy_r2s_dQ = tiled_copy_r2s_dQ.get_slice(tidx)
        cdQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
        if const_expr(self.arch // 10 in [8, 9, 12]):
            taccdQrdQ = thr_copy_r2s_dQ.retile(rdQ)
        else:
            taccdQcdQ_shape = thr_copy_r2s_dQ.partition_S(cdQ).shape
            taccdQrdQ = cute.make_tensor(rdQ.iterator, taccdQcdQ_shape)
        taccdQsdQ = thr_copy_r2s_dQ.partition_D(sdQ if const_expr(not self.dQ_swapAB) else sdQt)
        cute.copy(thr_copy_r2s_dQ, taccdQrdQ, taccdQsdQ)
    cute.arch.barrier(barrier_id=2, number_of_threads=256)
    gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_slice(tidx)
    tdQgdQ = gmem_thr_copy_dQ.partition_S(gdQ)
    tdQsdQ = gmem_thr_copy_dQ.partition_D(sdQ)
    tdQrdQ = cute.make_fragment_like(tdQsdQ, self.dtype)
    cute.autovec_copy(tdQsdQ, tdQrdQ)
    tdQcdQ = gmem_thr_copy_dQ.partition_S(cdQ)
    tdQpdQ = utils.predicate_k(tdQcdQ, limit=head_dim)
    for rest_m in cutlass.range(cute.size(tdQrdQ.shape[1]), unroll_full=True):
        if tdQcdQ[0, rest_m, 0][0] < seqlen_q - m_block * self.tile_m:
            cute.copy(gmem_tiled_copy_dQ, tdQrdQ[None, rest_m, None], tdQgdQ[None, rest_m, None], pred=tdQpdQ[None, rest_m, None])
