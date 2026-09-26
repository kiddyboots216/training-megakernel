"""FA4's forward `__call__`, on the program's forward scheduler.

Adapted from `FlashAttentionForwardSm90.__call__` in flash_attn/cute/flash_fwd_sm90.py
(FlashAttention 4, commit 890f238).  Copyright (c) 2025, Jay Shah, Ganesh Bikshandi,
Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao; FlashAttention is distributed under
the BSD 3-Clause License (see THIRD_PARTY_NOTICES.md).

The body's only change is the scheduler: `attention.AttentionForwardScheduler` replaces
`SingleTileVarlenScheduler`.  `forward_call_setup`, at the end of the module, applies
`cute.jit`; the function's own source carries no decorator, so the DSL's AST preprocessing
leaves its code unchanged and its Python `if` statements run as plain Python while the
program traces it.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.base_dsl.arch import Arch
from cutlass.cute.nvgpu import cpasync
from cutlass.utils.layout import LayoutEnum
from flash_attn.cute import utils
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.pack_gqa import make_packgqa_tiled_tma_atom, pack_gqa_layout
from flash_attn.cute.tile_scheduler import (
    SingleTileLPTScheduler,
    SingleTileScheduler,
    TileSchedulerArguments,
)
from flash_attn.cute.utils import AuxData
from quack import copy_utils, layout_utils, sm90_utils

from attention import AttentionForwardScheduler


def _forward_call_setup(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    mLSE: Optional[cute.Tensor],
    softmax_scale: Float32,
    mCuSeqlensQ: Optional[cute.Tensor] = None,
    mCuSeqlensK: Optional[cute.Tensor] = None,
    mSeqUsedQ: Optional[cute.Tensor] = None,
    mSeqUsedK: Optional[cute.Tensor] = None,
    mPageTable: Optional[cute.Tensor] = None,
    window_size_left: Int32 | int | None = None,
    window_size_right: Int32 | int | None = None,
    learnable_sink: Optional[cute.Tensor] = None,
    blocksparse_tensors: Optional[BlockSparseTensors] = None,
    aux_data: AuxData = AuxData(),
    stream: cuda.CUstream = None,
):
    """Configures FA4's forward kernel for these tensors and calls `self.kernel`.

    The program's member records the kernel arguments there, and the launch does nothing. With
    `mCuSeqlensQ`, Q and O are (total_q, heads, head_dim) and the LSE is (heads, total_q).
    """
    self._check_type(
        *(
            t.element_type if t is not None else None
            for t in (mQ, mK, mV, mO, mLSE, mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK)
        )
    )
    self.varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None
    mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
    QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
    mQ, mO = [layout_utils.select(t, QO_layout_transpose) for t in (mQ, mO)]
    KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
    mK, mV = [layout_utils.select(t, KV_layout_transpose) for t in (mK, mV)]
    LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
    mLSE = layout_utils.select(mLSE, LSE_layout_transpose) if const_expr(mLSE is not None) else None
    tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
    self.num_mma_threads = tiled_mma_qk.size
    self.num_threads_per_warp_group = 128
    self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
    assert self.num_wg_mma in [1, 2, 3]
    self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)
    self.num_producer_threads = 32
    self.num_Q_load_threads = self.num_threads_per_warp_group
    self.num_epilogue_threads = self.num_mma_threads
    self.num_mma_regs, self.num_producer_regs = {1: (256, 56), 2: (240, 24), 3: (160, 32)}[
        self.num_wg_mma
    ]
    self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
    self.use_scheduler_barrier = (
        self.num_wg_mma >= 2 and self.tile_hdim <= 128
        if const_expr(self.intra_wg_overlap)
        else self.num_wg_mma == 2
    )
    self.use_tma_Q = self.arch >= Arch.sm_90 and (
        not (self.pack_gqa and self.tile_m % self.qhead_per_kvhead != 0)
    )
    self.use_tma_O = self.use_tma_Q
    if const_expr(self.num_wg_mma == 2 and (not self.use_tma_Q or not self.use_tma_KV)):
        self.num_mma_regs, self.num_producer_regs = (224, 40)
    self.rescale_O_before_gemm = self.tile_hdimv > 128 and self.intra_wg_overlap
    self._setup_attributes()
    self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
        sm90_utils.make_smem_layout(mX.element_type, LayoutEnum.ROW_MAJOR, shape, stage)
        for mX, shape, stage in [
            (mQ, (self.tile_m, self.tile_hdim), None),
            (mK, (self.tile_n, self.tile_hdim), self.num_stages),
            (mV, (self.tile_n, self.tile_hdimv), self.num_stages),
            (mO, (self.tile_m, self.tile_hdimv), None),
        ]
    ]
    self.sP_layout = None
    if const_expr(not self.mma_pv_is_rs):
        self.sP_layout = sm90_utils.make_smem_layout(
            mV.element_type, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_n)
        )
    SharedStorage = self._get_shared_storage_cls()
    mQ_og, mO_og = (mQ, mO)
    if const_expr(self.pack_gqa):
        nheads_kv = mK.shape[2]
        mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
        mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
        if const_expr(mLSE is not None):
            mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)
    gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
    gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()
    gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
    self.tma_copy_bytes = {
        name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
        for name, mX, layout in [
            ("Q", mQ, self.sQ_layout),
            ("K", mK, self.sK_layout),
            ("V", mV, self.sV_layout),
        ]
    }
    make_tiled_tma_atom_fn = (
        partial(make_packgqa_tiled_tma_atom, qhead_per_kvhead=self.qhead_per_kvhead, head_idx=2)
        if const_expr(self.pack_gqa)
        else cpasync.make_tiled_tma_atom
    )
    tma_atom_Q, tma_tensor_Q = (None, None)
    if const_expr(self.use_tma_Q):
        tma_atom_Q, tma_tensor_Q = make_tiled_tma_atom_fn(
            gmem_tiled_copy_Q,
            mQ_og if const_expr(self.pack_gqa) else mQ,
            self.sQ_layout,
            (self.tile_m, self.tile_hdim),
        )
    tma_atom_K, tma_tensor_K = (None, None)
    tma_atom_V, tma_tensor_V = (None, None)
    if const_expr(self.use_tma_KV):
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
            1,
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
            1,
        )
    tma_atom_O, tma_tensor_O = (None, None)
    if const_expr(self.use_tma_O):
        mO_tma = mO_og if const_expr(self.pack_gqa) else mO
        if const_expr(self.varlen_q):
            mO_tma = copy_utils.create_ragged_tensor_for_tma(mO_tma, ragged_dim=0, ptr_shift=True)
        tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
            gmem_tiled_copy_O, mO_tma, self.sO_layout, (self.tile_m, self.tile_hdimv)
        )
    if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
        TileScheduler = AttentionForwardScheduler
    else:
        TileScheduler = (
            SingleTileScheduler
            if const_expr(not self.is_causal or self.is_local)
            else SingleTileLPTScheduler
        )
    tile_sched_args = TileSchedulerArguments(
        cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
        cute.size(mQ.shape[2]),
        cute.size(mQ.shape[3])
        if const_expr(mCuSeqlensQ is None)
        else cute.size(mCuSeqlensQ.shape[0] - 1),
        1,
        cute.size(mK.shape[0])
        if const_expr(mPageTable is None)
        else mK.shape[0] * mPageTable.shape[1],
        mQ.shape[1],
        mV.shape[1],
        total_q=cute.size(mQ.shape[0])
        if const_expr(mCuSeqlensQ is not None)
        else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
        tile_shape_mn=(self.tile_m, self.tile_n),
        mCuSeqlensQ=mCuSeqlensQ,
        mSeqUsedQ=mSeqUsedQ,
        qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        element_size=self.dtype.width // 8,
        is_persistent=False,
        lpt=self.is_causal or self.is_local,
    )
    tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
    grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
    softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
        softmax_scale, self.score_mod
    )
    window_size_left = Int32(window_size_left) if window_size_left is not None else None
    window_size_right = Int32(window_size_right) if window_size_right is not None else None
    fastdiv_mods = utils.compute_fastdiv_mods(
        mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_data.tensors, mPageTable
    )
    self.kernel(
        tma_tensor_Q if const_expr(self.use_tma_Q) else mQ,
        tma_tensor_K if const_expr(self.use_tma_KV) else mK,
        tma_tensor_V if const_expr(self.use_tma_KV) else mV,
        tma_tensor_O if const_expr(self.use_tma_O) else mO,
        mLSE,
        mCuSeqlensQ,
        mCuSeqlensK,
        mSeqUsedQ,
        mSeqUsedK,
        mPageTable,
        tma_atom_Q,
        tma_atom_K,
        tma_atom_V,
        tma_atom_O,
        softmax_scale_log2,
        softmax_scale,
        window_size_left,
        window_size_right,
        learnable_sink,
        blocksparse_tensors,
        self.sQ_layout,
        self.sK_layout,
        self.sV_layout,
        self.sO_layout,
        self.sP_layout,
        self.gmem_tiled_copy_Q,
        self.gmem_tiled_copy_K,
        self.gmem_tiled_copy_V,
        self.gmem_tiled_copy_O,
        tiled_mma_qk,
        tiled_mma_pv,
        tile_sched_params,
        TileScheduler,
        SharedStorage,
        aux_data,
        fastdiv_mods,
    ).launch(grid=grid_dim, block=[self.num_threads, 1, 1], stream=stream, min_blocks_per_mp=1)


# The DSL's preprocessor looks the function up under its own name in these globals and, as
# its source has no decorator, returns it unchanged. So that name must stay bound to the plain
# function, and the traced entry point gets a separate name.
forward_call_setup = cute.jit(preprocess=True)(_forward_call_setup)
