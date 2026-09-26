"""FA4's forward kernel, split for the program's roles.

Adapted from `FlashAttentionForwardSm90` in flash_attn/cute/flash_fwd_sm90.py
(FlashAttention 4, commit 890f238).  Copyright (c) 2025, Jay Shah, Ganesh Bikshandi,
Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao; FlashAttention is distributed under
the BSD 3-Clause License (see THIRD_PARTY_NOTICES.md).

- `forward_setup` and `forward_role` are `kernel`, split at its first warp-role branch
  (`warp_idx < 4`): the setup every warp runs returns the state the role bodies share, and
  the role body picks its branch by the constexpr role (`physical_warpgroup`) instead.  The
  `setmaxregister` requests are gone (the program sets one register split for all members),
  and shared memory comes from the program's page (`ProgramSmemAllocator`).
- `persistent_forward_mma` is `mma` with `softmax.reset()` at the top of the tile loop and
  the Q-pipeline release moved after `self.epilogue`, which stages O in Q's shared memory,
  so one CTA can run many tiles.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
import flash_attn.cute.pipeline as pipeline_custom
from cutlass import Float32, Int32, const_expr, pipeline
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline.helpers import pipeline_init_arrive, pipeline_init_wait
from flash_attn.cute import utils
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparse_utils import consume_block_sparse_loads
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.softmax import Softmax
from flash_attn.cute.utils import AuxData
from quack import layout_utils, sm90_utils
from quack.cute_dsl_utils import ParamsBase

from program_smem import ProgramSmemAllocator


@cute.jit
def forward_setup(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    mLSE: Optional[cute.Tensor],
    mCuSeqlensQ: Optional[cute.Tensor],
    mCuSeqlensK: Optional[cute.Tensor],
    mSeqUsedQ: Optional[cute.Tensor],
    mSeqUsedK: Optional[cute.Tensor],
    mPageTable: Optional[cute.Tensor],
    tma_atom_Q: Optional[cute.CopyAtom],
    tma_atom_K: Optional[cute.CopyAtom],
    tma_atom_V: Optional[cute.CopyAtom],
    tma_atom_O: Optional[cute.CopyAtom],
    softmax_scale_log2: Float32,
    softmax_scale: Optional[Float32],
    window_size_left: Optional[Int32],
    window_size_right: Optional[Int32],
    learnable_sink: Optional[cute.Tensor],
    blocksparse_tensors: Optional[BlockSparseTensors],
    sQ_layout: cute.ComposedLayout,
    sK_layout: cute.ComposedLayout,
    sV_layout: cute.ComposedLayout,
    sO_layout: cute.ComposedLayout,
    sP_layout: cute.ComposedLayout | None,
    gmem_tiled_copy_Q: cute.TiledCopy,
    gmem_tiled_copy_K: cute.TiledCopy,
    gmem_tiled_copy_V: cute.TiledCopy,
    gmem_tiled_copy_O: cute.TiledCopy,
    tiled_mma_qk: cute.TiledMma,
    tiled_mma_pv: cute.TiledMma,
    tile_sched_params: ParamsBase,
    TileScheduler: cutlass.Constexpr[Callable],
    SharedStorage: cutlass.Constexpr[Callable],
    aux_data: AuxData = AuxData(),
    fastdiv_mods=None,
):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
            if const_expr(tma_atom is not None):
                cpasync.prefetch_descriptor(tma_atom)
    smem = ProgramSmemAllocator()
    storage = smem.allocate(SharedStorage)
    mbar_ptr_Q = storage.mbar_ptr_Q.data_ptr()
    ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
    tma_warp = ThreadCooperativeGroup(1)
    load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
    mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
    if const_expr(self.use_tma_Q):
        pipeline_q = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=mbar_ptr_Q,
            num_stages=1,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=self.tma_copy_bytes["Q"],
            defer_sync=True,
        )
    else:
        pipeline_q = pipeline_custom.PipelineCpAsync.create(
            barrier_storage=mbar_ptr_Q,
            num_stages=1,
            producer_group=load_threads,
            consumer_group=mma_warps,
            defer_sync=True,
            elect_one_release=True,
            syncwarp_before_release=False,
        )
    if const_expr(self.use_tma_KV):
        pipeline_k = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_stages,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=self.tma_copy_bytes["K"],
            defer_sync=True,
        )
        pipeline_v = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_V.data_ptr(),
            num_stages=self.num_stages,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=self.tma_copy_bytes["V"],
            defer_sync=True,
        )
    else:
        pipeline_k = pipeline_custom.PipelineCpAsync.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_stages,
            producer_group=load_threads,
            consumer_group=mma_warps,
            defer_sync=True,
            elect_one_release=True,
            syncwarp_before_release=False,
        )
        pipeline_v = pipeline_custom.PipelineCpAsync.create(
            barrier_storage=storage.mbar_ptr_V.data_ptr(),
            num_stages=self.num_stages,
            producer_group=load_threads,
            consumer_group=mma_warps,
            defer_sync=True,
            elect_one_release=True,
            syncwarp_before_release=False,
        )
    pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
    sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
    sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
    if const_expr(not self.Q_in_regs):
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
    else:
        sV = storage.sQ.get_tensor(sV_layout.outer, swizzle=sV_layout.inner, dtype=mV.element_type)
    sVt = layout_utils.transpose_view(sV)
    sP = None
    if const_expr(sP_layout is not None):
        sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
    sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)
    block_info = BlockInfo(
        self.tile_m,
        self.tile_n,
        self.is_causal,
        self.is_local,
        False,
        window_size_left,
        window_size_right,
        qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
    )
    SeqlenInfoCls = partial(
        SeqlenInfoQK.create,
        seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
        seqlen_k_static=mK.shape[0]
        if const_expr(mPageTable is None)
        else mK.shape[0] * mPageTable.shape[1],
        mCuSeqlensQ=mCuSeqlensQ,
        mCuSeqlensK=mCuSeqlensK,
        mSeqUsedQ=mSeqUsedQ,
        mSeqUsedK=mSeqUsedK,
        mCuTotalMBlocks=blocksparse_tensors.cu_total_m_blocks
        if blocksparse_tensors is not None
        else None,
        mCuBlockIdxOffsets=blocksparse_tensors.cu_block_idx_offsets
        if blocksparse_tensors is not None
        else None,
    )
    AttentionMaskCls = partial(
        AttentionMask,
        self.tile_m,
        self.tile_n,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
    )
    TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)
    pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
    return (
        warp_idx,
        storage,
        pipeline_q,
        pipeline_k,
        pipeline_v,
        sQ,
        sK,
        sV,
        sVt,
        sP,
        sO,
        block_info,
        SeqlenInfoCls,
        AttentionMaskCls,
        TileSchedulerCls,
    )


@cute.jit
def forward_role(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    mLSE: Optional[cute.Tensor],
    mCuSeqlensQ: Optional[cute.Tensor],
    mCuSeqlensK: Optional[cute.Tensor],
    mSeqUsedQ: Optional[cute.Tensor],
    mSeqUsedK: Optional[cute.Tensor],
    mPageTable: Optional[cute.Tensor],
    tma_atom_Q: Optional[cute.CopyAtom],
    tma_atom_K: Optional[cute.CopyAtom],
    tma_atom_V: Optional[cute.CopyAtom],
    tma_atom_O: Optional[cute.CopyAtom],
    softmax_scale_log2: Float32,
    softmax_scale: Optional[Float32],
    window_size_left: Optional[Int32],
    window_size_right: Optional[Int32],
    learnable_sink: Optional[cute.Tensor],
    blocksparse_tensors: Optional[BlockSparseTensors],
    sQ_layout: cute.ComposedLayout,
    sK_layout: cute.ComposedLayout,
    sV_layout: cute.ComposedLayout,
    sO_layout: cute.ComposedLayout,
    sP_layout: cute.ComposedLayout | None,
    gmem_tiled_copy_Q: cute.TiledCopy,
    gmem_tiled_copy_K: cute.TiledCopy,
    gmem_tiled_copy_V: cute.TiledCopy,
    gmem_tiled_copy_O: cute.TiledCopy,
    tiled_mma_qk: cute.TiledMma,
    tiled_mma_pv: cute.TiledMma,
    tile_sched_params: ParamsBase,
    TileScheduler: cutlass.Constexpr[Callable],
    SharedStorage: cutlass.Constexpr[Callable],
    aux_data: AuxData = AuxData(),
    fastdiv_mods=None,
    warp_idx=None,
    storage=None,
    pipeline_q=None,
    pipeline_k=None,
    pipeline_v=None,
    sQ=None,
    sK=None,
    sV=None,
    sVt=None,
    sP=None,
    sO=None,
    block_info=None,
    SeqlenInfoCls=None,
    AttentionMaskCls=None,
    TileSchedulerCls=None,
    physical_warpgroup: cutlass.Constexpr[int] = 0,
):
    if const_expr(physical_warpgroup == 0):
        self.load(
            mQ,
            mK,
            mV,
            sQ,
            sK,
            sV,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            pipeline_k,
            pipeline_v,
            pipeline_q,
            gmem_tiled_copy_Q,
            mPageTable,
            blocksparse_tensors,
            block_info,
            SeqlenInfoCls,
            TileSchedulerCls,
        )
    else:
        tidx, _, _ = cute.arch.thread_idx()
        tidx = tidx - 128
        self.mma(
            tiled_mma_qk,
            tiled_mma_pv,
            mO,
            mLSE,
            sQ,
            sK,
            sVt,
            sP,
            sO,
            learnable_sink,
            pipeline_k,
            pipeline_v,
            pipeline_q,
            gmem_tiled_copy_O,
            tma_atom_O,
            tidx,
            softmax_scale_log2,
            softmax_scale,
            block_info,
            SeqlenInfoCls,
            AttentionMaskCls,
            TileSchedulerCls,
            blocksparse_tensors,
            aux_data,
            fastdiv_mods,
        )


@cute.jit
def persistent_forward_mma(
    self,
    tiled_mma_qk: cute.TiledMma,
    tiled_mma_pv: cute.TiledMma,
    mO: cute.Tensor,
    mLSE: Optional[cute.Tensor],
    sQ: cute.Tensor,
    sK: cute.Tensor,
    sVt: cute.Tensor,
    sP: Optional[cute.Tensor],
    sO: cute.Tensor,
    learnable_sink: Optional[cute.Tensor],
    pipeline_k: pipeline.PipelineAsync,
    pipeline_v: pipeline.PipelineAsync,
    pipeline_q: pipeline.PipelineAsync,
    gmem_tiled_copy_O: cute.TiledCopy,
    tma_atom_O: Optional[cute.CopyAtom],
    tidx: Int32,
    softmax_scale_log2: Float32,
    softmax_scale: Optional[Float32],
    block_info: BlockInfo,
    SeqlenInfoCls: Callable,
    AttentionMaskCls: Callable,
    TileSchedulerCls: Callable,
    blocksparse_tensors: Optional[BlockSparseTensors],
    aux_data: AuxData = AuxData(),
    fastdiv_mods=None,
):
    aux_tensors = aux_data.tensors
    warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
    warp_group_thread_layout = cute.make_layout(
        self.num_wg_mma, stride=self.num_threads_per_warp_group
    )
    thr_mma_qk = tiled_mma_qk.get_slice(tidx)
    wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
    wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
    _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
        wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ, sK
    )
    mma_qk_fn = partial(
        sm90_utils.gemm_zero_init, tiled_mma_qk, (self.tile_m, self.tile_n), tSrQ, tSrK
    )
    acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
        wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), sP, sVt
    )
    mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)
    smem_copy_atom_P = utils.get_smem_store_atom(self.arch.major * 10 + self.arch.minor, self.dtype)
    smem_thr_copy_P = cute.make_tiled_copy_C(smem_copy_atom_P, tiled_mma_qk).get_slice(tidx)
    tPsP = smem_thr_copy_P.partition_D(sP) if const_expr(sP is not None) else None
    smem_copy_params = SimpleNamespace(smem_thr_copy_P=smem_thr_copy_P, tPsP=tPsP)
    self.mma_init()
    q_consumer_phase = Int32(0)
    kv_consumer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, self.num_stages
    )
    tile_scheduler = TileSchedulerCls()
    work_tile = tile_scheduler.initial_work_tile_info()
    softmax = Softmax.create(
        softmax_scale_log2, num_rows=acc_O.shape[0][0] * acc_O.shape[1], softmax_scale=softmax_scale
    )
    scores_scale = None
    if const_expr(self.rescale_O_before_gemm):
        scores_scale = cute.make_rmem_tensor_like(softmax.row_max, Float32)
    mma_one_n_block_all = partial(
        self.mma_one_n_block_intrawg_overlap
        if const_expr(self.intra_wg_overlap)
        else self.mma_one_n_block,
        mma_qk_fn=mma_qk_fn,
        pipeline_k=pipeline_k,
        pipeline_v=pipeline_v,
        acc_O=acc_O,
        tOrP=tOrP,
        smem_copy_params=smem_copy_params,
        check_inf=True,
        scores_scale=scores_scale,
    )
    process_first_half_block = partial(
        self.first_half_block_overlap,
        mma_qk_fn=mma_qk_fn,
        pipeline_k=pipeline_k,
        tOrP=tOrP,
        smem_copy_params=smem_copy_params,
        scores_scale=scores_scale,
        softmax=softmax,
        acc_O=acc_O,
    )
    process_last_half_block = partial(
        self.last_half_block_overlap,
        pipeline_v=pipeline_v,
        mma_pv_fn=mma_pv_fn,
        scores_scale=scores_scale,
        softmax=softmax,
        acc_O=acc_O,
    )
    while work_tile.is_valid_tile:
        softmax.reset()
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx
        seqlen = SeqlenInfoCls(batch_idx)
        recompute_fastdiv_mods_q = cutlass.const_expr(
            aux_tensors is not None and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
        )
        recompute_fastdiv_mods_k = cutlass.const_expr(
            aux_tensors is not None and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
        )
        if cutlass.const_expr(fastdiv_mods is not None):
            seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
            fastdiv_mods = (
                seqlen_q_divmod
                if not recompute_fastdiv_mods_q
                else FastDivmodDivisor(seqlen.seqlen_q),
                seqlen_k_divmod
                if not recompute_fastdiv_mods_k
                else FastDivmodDivisor(seqlen.seqlen_k),
            )
        mask = AttentionMaskCls(seqlen)
        mask_fn = partial(
            mask.apply_mask,
            batch_idx=batch_idx,
            head_idx=head_idx,
            m_block=m_block,
            thr_mma=thr_mma_qk,
            mask_causal=self.is_causal,
            mask_local=self.is_local,
            aux_data=aux_data,
            fastdiv_mods=fastdiv_mods,
        )
        score_mod_fn = None
        if const_expr(self.score_mod is not None):
            score_mod_fn = partial(
                self.apply_score_mod,
                thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                softmax_scale=softmax_scale,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )
        mma_one_n_block = partial(
            mma_one_n_block_all, seqlen=seqlen, softmax=softmax, score_mod_fn=score_mod_fn
        )
        n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
        pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)
        O_should_accumulate = False
        if const_expr(not self.use_block_sparsity):
            if const_expr(self.intra_wg_overlap):
                kv_consumer_state = process_first_half_block(
                    n_block=n_block_max - 1,
                    seqlen=seqlen,
                    kv_consumer_state=kv_consumer_state,
                    mask_fn=partial(mask_fn, mask_mod=self.mask_mod),
                    score_mod_fn=score_mod_fn,
                    is_first_block=True,
                )
            else:
                self.warp_scheduler_barrier_sync()
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=n_block_max - 1,
                    seqlen=seqlen,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=True),
                    is_first_n_block=True,
                    mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True),
                )
                O_should_accumulate = True
            n_block_max -= 1
            if const_expr(self.is_causal or self.is_local):
                n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                    seqlen, m_block, n_block_min
                )
                for n_tile in cutlass.range(n_block_max - n_block_min_causal_local_mask, unroll=1):
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=n_block_max - 1 - n_tile,
                        seqlen=seqlen,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False),
                    )
                    O_should_accumulate = True
                n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
            n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                seqlen, m_block, n_block_min
            )
            for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=n_block_max - 1 - n_tile,
                    seqlen=seqlen,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False),
                )
                O_should_accumulate = True
            if const_expr(self.is_local and block_info.window_size_left is not None):
                n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=n_block_max - 1 - n_tile,
                        seqlen=seqlen,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False),
                    )
                    O_should_accumulate = True
            if const_expr(self.intra_wg_overlap):
                kv_consumer_state = process_last_half_block(
                    kv_consumer_state=kv_consumer_state, zero_init=not O_should_accumulate
                )
                O_should_accumulate = True
            else:
                self.warp_scheduler_barrier_arrive()
        else:
            kv_consumer_state, O_should_accumulate, processed_any = consume_block_sparse_loads(
                blocksparse_tensors,
                batch_idx,
                head_idx,
                m_block,
                seqlen,
                kv_consumer_state,
                mma_pv_fn,
                mma_one_n_block,
                process_first_half_block,
                process_last_half_block,
                mask_fn,
                score_mod_fn,
                O_should_accumulate,
                self.mask_mod,
                fastdiv_mods,
                self.intra_wg_overlap,
                self.warp_scheduler_barrier_sync,
                self.warp_scheduler_barrier_arrive,
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                self.q_subtile_factor,
            )
            if not processed_any:
                softmax.reset()
                acc_O.fill(0.0)
        q_consumer_phase ^= 1
        sink_val = None
        if const_expr(learnable_sink is not None):
            if const_expr(not self.pack_gqa):
                sink_val = Float32(learnable_sink[head_idx])
            else:
                sink_val = cute.make_rmem_tensor_like(softmax.row_max, Float32)
                cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
                tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
                for r in cutlass.range(cute.size(sink_val), unroll_full=True):
                    row = m_block * self.tile_m + tScS_mn[r][0]
                    q_head_idx = row % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                    sink_val[r] = Float32(learnable_sink[q_head_idx])
        row_scale = softmax.finalize(sink_val=sink_val)
        softmax.rescale_O(acc_O, row_scale)
        self.epilogue(
            acc_O,
            softmax.row_sum,
            mO,
            mLSE,
            sO,
            seqlen,
            gmem_tiled_copy_O,
            tma_atom_O,
            tiled_mma_pv,
            tidx,
            m_block,
            head_idx,
            batch_idx,
        )
        pipeline_q.consumer_release_w_index(0)
        tile_scheduler.advance_to_next_work()
        work_tile = tile_scheduler.get_current_work()
