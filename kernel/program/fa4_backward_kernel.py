"""FA4's backward kernel, split for the program's roles.

Adapted from `FlashAttentionBackwardSm90.kernel` in flash_attn/cute/flash_bwd_sm90.py
(FlashAttention 4, commit 890f238).  Copyright (c) 2025, Jay Shah, Ganesh Bikshandi,
Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao; FlashAttention is distributed under
the BSD 3-Clause License (see THIRD_PARTY_NOTICES.md).

`kernel` is split at its first warp-role branch (`warp_idx < 4`) into the setup every warp
runs (`backward_setup`, which returns the state the role bodies share) and the
role body (`backward_role`), which picks its branch by the constexpr role
(`physical_warpgroup`).  The `setmaxregister` requests are gone (the program sets one register
split for all members), shared memory comes from the program's page, and
`attention.SlotSeqlenInfoQK` replaces `SeqlenInfoQK`, so the kernel finds its workspace rows
(the log2 LSE, dPsum and the dQ, dK and dV accumulators) through the program's slot table.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync
from flash_attn.cute import pipeline
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.utils import AuxData
from quack.cute_dsl_utils import ParamsBase

from attention import SlotSeqlenInfoQK
from program_smem import ProgramSmemAllocator


@cute.jit
def backward_setup(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    tma_atom_Q: cute.CopyAtom,
    tma_atom_K: cute.CopyAtom,
    tma_atom_V: cute.CopyAtom,
    tma_atom_dO: cute.CopyAtom,
    tma_atom_dK: cute.CopyAtom,
    tma_atom_dV: cute.CopyAtom,
    mLSE: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mCuSeqlensQ: Optional[cute.Tensor],
    mCuSeqlensK: Optional[cute.Tensor],
    mSeqUsedQ: Optional[cute.Tensor],
    mSeqUsedK: Optional[cute.Tensor],
    sQ_layout: cute.ComposedLayout,
    sK_layout: cute.ComposedLayout,
    sV_layout: cute.ComposedLayout,
    sPdS_layout: cute.ComposedLayout,
    sdO_layout: cute.ComposedLayout,
    sdQaccum_layout: cute.Layout,
    r2s_tiled_copy_dQaccum: cute.TiledCopy,
    tiled_mma_SdP: cute.TiledMma,
    tiled_mma_dK: cute.TiledMma,
    tiled_mma_dV: cute.TiledMma,
    tiled_mma_dQ: cute.TiledMma,
    softmax_scale_log2,
    softmax_scale,
    tile_sched_params: ParamsBase,
    TileScheduler: cutlass.Constexpr[Callable],
    SharedStorage: cutlass.Constexpr[Callable],
    aux_data: AuxData = AuxData(),
    fastdiv_mods=(None, None),
    blocksparse_tensors: Optional[BlockSparseTensors] = None,
    qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
    mdQ_semaphore: Optional[cute.Tensor] = None,
    mdK_semaphore: Optional[cute.Tensor] = None,
    mdV_semaphore: Optional[cute.Tensor] = None,
    window_size_left: Optional[Int32] = None,
    window_size_right: Optional[Int32] = None,
):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        for atom in [tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_dO, tma_atom_dK, tma_atom_dV]:
            if const_expr(atom is not None):
                cpasync.prefetch_descriptor(atom)
    smem = ProgramSmemAllocator()
    storage = smem.allocate(SharedStorage)
    pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
    pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
        cutlass.pipeline.Agent.Thread, self.num_mma_threads // cute.arch.WARP_SIZE
    )
    pipeline_Q = pipeline.PipelineTmaAsync.create(
        barrier_storage=storage.mbar_ptr_Q.data_ptr(),
        num_stages=self.Q_stage,
        producer_group=pipeline_producer_group,
        consumer_group=pipeline_consumer_group,
        tx_count=self.tma_copy_bytes["Q"] + self.tma_copy_bytes["LSE"],
        defer_sync=True,
    )
    pipeline_dO = pipeline.PipelineTmaAsync.create(
        barrier_storage=storage.mbar_ptr_dO.data_ptr(),
        num_stages=self.dO_stage,
        producer_group=pipeline_producer_group,
        consumer_group=pipeline_consumer_group,
        tx_count=self.tma_copy_bytes["dO"] + self.tma_copy_bytes["dPsum"],
        defer_sync=False,
    )
    sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
    sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
    sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
    sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
    sP = None
    if const_expr(not self.mma_dkv_is_rs):
        sP = storage.sP.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
    sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
    sLSE = storage.sLSE.get_tensor(
        cute.make_layout((self.tile_m, self.Q_stage), stride=(1, cute.round_up(self.tile_m, 64)))
    )
    sdPsum = storage.sdPsum.get_tensor(
        cute.make_layout((self.tile_m, self.dO_stage), stride=(1, cute.round_up(self.tile_m, 64)))
    )
    sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)
    block_info = BlockInfo(
        self.tile_m,
        self.tile_n,
        self.is_causal,
        self.is_local,
        False,
        window_size_left,
        window_size_right,
        qhead_per_kvhead_packgqa=1,
    )
    SeqlenInfoCls = partial(
        SlotSeqlenInfoQK.create,
        seqlen_q_static=mQ.shape[0],
        seqlen_k_static=mK.shape[0],
        mCuSeqlensQ=mCuSeqlensQ,
        mCuSeqlensK=mCuSeqlensK,
        mSeqUsedQ=mSeqUsedQ,
        mSeqUsedK=mSeqUsedK,
        tile_m=self.tile_m,
        tile_n=self.tile_n,
    )
    AttentionMaskCls = partial(
        AttentionMask,
        self.tile_m,
        self.tile_n,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        swap_AB=self.SdP_swapAB,
    )
    TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)
    return (
        warp_idx,
        storage,
        pipeline_Q,
        pipeline_dO,
        sQ,
        sdO,
        sK,
        sV,
        sP,
        sdS,
        sLSE,
        sdPsum,
        sdQaccum,
        block_info,
        SeqlenInfoCls,
        AttentionMaskCls,
        TileSchedulerCls,
    )


@cute.jit
def backward_role(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    tma_atom_Q: cute.CopyAtom,
    tma_atom_K: cute.CopyAtom,
    tma_atom_V: cute.CopyAtom,
    tma_atom_dO: cute.CopyAtom,
    tma_atom_dK: cute.CopyAtom,
    tma_atom_dV: cute.CopyAtom,
    mLSE: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mCuSeqlensQ: Optional[cute.Tensor],
    mCuSeqlensK: Optional[cute.Tensor],
    mSeqUsedQ: Optional[cute.Tensor],
    mSeqUsedK: Optional[cute.Tensor],
    sQ_layout: cute.ComposedLayout,
    sK_layout: cute.ComposedLayout,
    sV_layout: cute.ComposedLayout,
    sPdS_layout: cute.ComposedLayout,
    sdO_layout: cute.ComposedLayout,
    sdQaccum_layout: cute.Layout,
    r2s_tiled_copy_dQaccum: cute.TiledCopy,
    tiled_mma_SdP: cute.TiledMma,
    tiled_mma_dK: cute.TiledMma,
    tiled_mma_dV: cute.TiledMma,
    tiled_mma_dQ: cute.TiledMma,
    softmax_scale_log2,
    softmax_scale,
    tile_sched_params: ParamsBase,
    TileScheduler: cutlass.Constexpr[Callable],
    SharedStorage: cutlass.Constexpr[Callable],
    aux_data: AuxData = AuxData(),
    fastdiv_mods=(None, None),
    blocksparse_tensors: Optional[BlockSparseTensors] = None,
    qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
    mdQ_semaphore: Optional[cute.Tensor] = None,
    mdK_semaphore: Optional[cute.Tensor] = None,
    mdV_semaphore: Optional[cute.Tensor] = None,
    window_size_left: Optional[Int32] = None,
    window_size_right: Optional[Int32] = None,
    warp_idx=None,
    storage=None,
    pipeline_Q=None,
    pipeline_dO=None,
    sQ=None,
    sdO=None,
    sK=None,
    sV=None,
    sP=None,
    sdS=None,
    sLSE=None,
    sdPsum=None,
    sdQaccum=None,
    block_info=None,
    SeqlenInfoCls=None,
    AttentionMaskCls=None,
    TileSchedulerCls=None,
    physical_warpgroup: cutlass.Constexpr[int] = 0,
):
    if const_expr(physical_warpgroup == 0):
        if warp_idx == 0:
            self.load(
                mQ,
                mK,
                mV,
                mdO,
                mLSE,
                mdPsum,
                sQ,
                sK,
                sV,
                sdO,
                sLSE,
                sdPsum,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_dO,
                pipeline_Q,
                pipeline_dO,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                qhead_per_kvhead_divmod,
            )
        if warp_idx == 1:
            self.dQaccum_store(
                mdQaccum,
                sdQaccum,
                block_info,
                TileSchedulerCls,
                SeqlenInfoCls,
                blocksparse_tensors,
                mdQ_semaphore,
            )
    else:
        tidx, _, _ = cute.arch.thread_idx()
        tidx = tidx - 128
        mma_args = (
            tiled_mma_SdP,
            tiled_mma_dK,
            tiled_mma_dV,
            tiled_mma_dQ,
            mdK,
            mdV,
            mdK_semaphore,
            mdV_semaphore,
            mdQaccum,
            sQ,
            sK,
            sV,
            sdO,
            sP,
            sdS,
            sLSE,
            sdPsum,
            sdQaccum,
            pipeline_Q,
            pipeline_dO,
            tidx,
            tma_atom_dK,
            tma_atom_dV,
            r2s_tiled_copy_dQaccum,
            softmax_scale_log2,
            softmax_scale,
            block_info,
            SeqlenInfoCls,
            AttentionMaskCls,
            TileSchedulerCls,
            aux_data,
            fastdiv_mods,
            blocksparse_tensors,
            qhead_per_kvhead_divmod,
        )
        if const_expr(self.num_wg_dQ == self.num_wg_mma):
            self.mma(*mma_args, is_dQ_wg=True)
        else:
            warp_idx_in_mma = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - 4
            if warp_idx_in_mma < 4:
                self.mma(*mma_args, is_dQ_wg=True)
            else:
                self.mma(*mma_args, is_dQ_wg=False)
