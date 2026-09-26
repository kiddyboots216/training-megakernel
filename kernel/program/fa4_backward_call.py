"""FA4's backward `__call__`, on the program's backward scheduler.

Adapted from `FlashAttentionBackwardSm90.__call__` in flash_attn/cute/flash_bwd_sm90.py
(FlashAttention 4, commit 890f238).  Copyright (c) 2025, Jay Shah, Ganesh Bikshandi,
Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao; FlashAttention is distributed under
the BSD 3-Clause License (see THIRD_PARTY_NOTICES.md).

The body's only change is the scheduler: `attention.AttentionBackwardScheduler` replaces
`SingleTileVarlenScheduler`.  `backward_call_setup`, at the end of the module, applies
`cute.jit`; the function's own source carries no decorator, so the DSL's AST preprocessing
leaves its code unchanged and its Python `if` statements run as plain Python while the
program traces it.  In the program the member's `kernel` records the kernel arguments and the
launch does nothing.
"""

from __future__ import annotations

import math
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.tile_scheduler import (
    SingleTileLPTBwdScheduler,
    SingleTileScheduler,
    TileSchedulerArguments,
)
from flash_attn.cute.utils import AuxData
from quack import copy_utils, layout_utils

from attention import AttentionBackwardScheduler


def _backward_call_setup(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSE: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    softmax_scale: Float32,
    mCuSeqlensQ: Optional[cute.Tensor] = None,
    mCuSeqlensK: Optional[cute.Tensor] = None,
    mSeqUsedQ: Optional[cute.Tensor] = None,
    mSeqUsedK: Optional[cute.Tensor] = None,
    window_size_left: Int32 | int | None = None,
    window_size_right: Int32 | int | None = None,
    mdQ_semaphore: Optional[cute.Tensor] = None,
    mdK_semaphore: Optional[cute.Tensor] = None,
    mdV_semaphore: Optional[cute.Tensor] = None,
    aux_data: AuxData = AuxData(),
    blocksparse_tensors: Optional[BlockSparseTensors] = None,
    stream: cuda.CUstream = None,
):
    self.varlen_k = mCuSeqlensK is not None or mSeqUsedK is not None
    self._check_type(
        *(
            t.element_type if t is not None else None
            for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
        )
    )
    self.is_varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None
    mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV = [
        assume_tensor_aligned(t) for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
    ]

    def _qkv_transpose(t):
        return layout_utils.select(t, [1, 3, 2, 0] if cute.rank(t.shape) == 4 else [0, 2, 1])

    mQ, mK, mV, mdO = [_qkv_transpose(t) for t in (mQ, mK, mV, mdO)]
    if const_expr(self.qhead_per_kvhead == 1):
        mdK, mdV = [_qkv_transpose(t) for t in (mdK, mdV)]
    else:
        accum_transpose = [2, 1, 0] if cute.rank(mdK.shape) == 3 else [1, 0]
        mdK, mdV = [layout_utils.select(t, accum_transpose) for t in (mdK, mdV)]
    LSE_dPsum_dQaccum_transpose = [2, 1, 0] if cute.rank(mLSE.shape) == 3 else [1, 0]
    mLSE, mdPsum, mdQaccum = [
        layout_utils.select(t, LSE_dPsum_dQaccum_transpose) for t in (mLSE, mdPsum, mdQaccum)
    ]
    tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ = self._get_tiled_mma()
    if const_expr(self.deterministic):
        assert mdQ_semaphore is not None
        mdQ_semaphore = layout_utils.select(mdQ_semaphore, mode=[2, 3, 1, 0])
    if const_expr(self.deterministic and self.qhead_per_kvhead > 1):
        assert mdK_semaphore is not None
        assert mdV_semaphore is not None
        mdK_semaphore, mdV_semaphore = [
            layout_utils.select(t, mode=[2, 3, 1, 0]) for t in (mdK_semaphore, mdV_semaphore)
        ]
    else:
        mdK_semaphore = None
        mdV_semaphore = None
    self.num_mma_threads = tiled_mma_SdP.size
    assert self.num_mma_threads + 128 == self.num_threads
    self.num_threads_per_warp_group = 128
    self.num_producer_threads = 32
    REG_LIMIT = 504 if self.num_wg_mma == 2 else 512
    if const_expr(self.num_wg_mma == 2):
        if const_expr(self.num_wg_dQ == 1):
            self.num_mma_regs_wg0 = 256
            self.num_mma_regs_wg1 = 224
        else:
            self.num_mma_regs_wg0 = 240
            self.num_mma_regs_wg1 = 240
        self.num_mma_regs = self.num_mma_regs_wg0
        self.num_producer_regs = 24
        assert self.num_mma_regs_wg0 + self.num_mma_regs_wg1 + self.num_producer_regs <= REG_LIMIT
    else:
        self.num_mma_regs_wg0 = 160
        self.num_mma_regs_wg1 = 160
        self.num_mma_regs = 160
        self.num_producer_regs = 32
        assert self.num_mma_regs_wg0 * self.num_wg_mma + self.num_producer_regs <= REG_LIMIT
    self._setup_attributes()
    SharedStorage = self._get_shared_storage_cls()
    self.tma_copy_bytes = {
        name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
        for name, mX, layout in [
            ("Q", mQ, self.sQ_layout),
            ("K", mK, self.sK_layout),
            ("V", mV, self.sV_layout),
            ("dO", mdO, self.sdO_layout),
        ]
    }
    self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
    self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
    self.tma_copy_bytes["dQ"] = self.tile_m * self.tile_hdim * Float32.width // 8 // self.num_wg_dQ
    self.tma_copy_bytes["dKacc"] = self.tile_n * self.tile_hdim * Float32.width // 8
    self.tma_copy_bytes["dVacc"] = self.tile_n * self.tile_hdimv * Float32.width // 8
    tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mQ,
        cute.select(self.sQ_layout, mode=[0, 1]),
        (self.tile_m, self.tile_hdim),
    )
    tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mK,
        cute.select(self.sK_layout, mode=[0, 1]),
        (self.tile_n, self.tile_hdim),
    )
    tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mV,
        cute.select(self.sV_layout, mode=[0, 1]),
        (self.tile_n, self.tile_hdimv),
    )
    tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mdO,
        cute.select(self.sdO_layout, mode=[0, 1]),
        (self.tile_m, self.tile_hdimv),
    )
    if const_expr(self.qhead_per_kvhead == 1):
        mdK_tma = (
            copy_utils.create_ragged_tensor_for_tma(mdK, ragged_dim=0, ptr_shift=True)
            if self.varlen_k
            else mdK
        )
        mdV_tma = (
            copy_utils.create_ragged_tensor_for_tma(mdV, ragged_dim=0, ptr_shift=True)
            if self.varlen_k
            else mdV
        )
        tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mdK_tma,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mdV_tma,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
        )
    else:
        tma_atom_dK = tma_atom_dV = tma_tensor_dK = tma_tensor_dV = None
    if const_expr(mCuSeqlensK is not None or mSeqUsedK is not None):
        TileScheduler = AttentionBackwardScheduler
    elif const_expr(self.deterministic):
        TileScheduler = SingleTileLPTBwdScheduler
    else:
        TileScheduler = SingleTileScheduler
    self.spt = (self.is_causal or self.is_local) and self.deterministic
    tile_sched_args = TileSchedulerArguments(
        cute.ceil_div(cute.size(mK.shape[0]), self.tile_n),
        cute.size(mQ.shape[2]),
        cute.size(mK.shape[3])
        if const_expr(mCuSeqlensK is None)
        else cute.size(mCuSeqlensK.shape[0] - 1),
        1,
        cute.size(mQ.shape[0]),
        mQ.shape[1],
        mV.shape[1],
        total_q=cute.size(mK.shape[0])
        if const_expr(mCuSeqlensK is not None)
        else cute.size(mK.shape[0]) * cute.size(mK.shape[3]),
        tile_shape_mn=(self.tile_n, self.tile_m),
        mCuSeqlensQ=mCuSeqlensK,
        mSeqUsedQ=mSeqUsedK,
        qhead_per_kvhead_packgqa=1,
        element_size=self.dtype.width // 8,
        is_persistent=False,
        lpt=self.spt,
        head_swizzle=self.deterministic,
    )
    tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
    grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
    LOG2_E = math.log2(math.e)
    if const_expr(self.score_mod is None):
        softmax_scale_log2 = softmax_scale * LOG2_E
    else:
        softmax_scale_log2 = LOG2_E
    fastdiv_mods = None
    if const_expr(aux_data.tensors is not None):
        seqlen_q = cute.size(mQ.shape[0])
        seqlen_k = cute.size(mK.shape[0])
        seqlen_q_divmod = FastDivmodDivisor(seqlen_q)
        seqlen_k_divmod = FastDivmodDivisor(seqlen_k)
        fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)
    qhead_per_kvhead_divmod = None
    if const_expr(self.qhead_per_kvhead > 1):
        qhead_per_kvhead_divmod = FastDivmodDivisor(self.qhead_per_kvhead)
    self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
    if const_expr(window_size_left is not None):
        window_size_left = Int32(window_size_left)
    if const_expr(window_size_right is not None):
        window_size_right = Int32(window_size_right)
    self.kernel(
        tma_tensor_Q,
        tma_tensor_K,
        tma_tensor_V,
        tma_tensor_dO,
        tma_tensor_dK if const_expr(self.qhead_per_kvhead == 1) else mdK,
        tma_tensor_dV if const_expr(self.qhead_per_kvhead == 1) else mdV,
        tma_atom_Q,
        tma_atom_K,
        tma_atom_V,
        tma_atom_dO,
        tma_atom_dK,
        tma_atom_dV,
        mLSE,
        mdPsum,
        mdQaccum,
        mCuSeqlensQ,
        mCuSeqlensK,
        mSeqUsedQ,
        mSeqUsedK,
        self.sQ_layout,
        self.sK_layout,
        self.sV_layout,
        self.sPdS_layout,
        self.sdO_layout,
        self.sdQaccum_layout,
        self.r2s_tiled_copy_dQaccum,
        tiled_mma_SdP,
        tiled_mma_dK,
        tiled_mma_dV,
        tiled_mma_dQ,
        softmax_scale_log2,
        softmax_scale,
        tile_sched_params,
        TileScheduler,
        SharedStorage,
        aux_data,
        fastdiv_mods,
        blocksparse_tensors,
        qhead_per_kvhead_divmod,
        mdQ_semaphore,
        mdK_semaphore,
        mdV_semaphore,
        window_size_left,
        window_size_right,
    ).launch(
        grid=grid_dim,
        block=[self.num_threads, 1, 1],
        stream=stream,
        min_blocks_per_mp=1,
        use_pdl=True,
    )


# The DSL's preprocessor looks the function up under its own name in these globals and, as
# its source has no decorator, returns it unchanged. So that name must stay bound to the plain
# function, and the traced entry point gets a separate name.
backward_call_setup = cute.jit(preprocess=True)(_backward_call_setup)
