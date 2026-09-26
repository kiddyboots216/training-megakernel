"""Quack's SM90 GEMM kernel and epilogue, as members of the program.

Adapted from `GemmSm90.kernel` in quack/gemm_sm90.py (Copyright (c) 2025-2026, QuACK team)
and `GemmBase.epilogue` in quack/gemm_base.py (Copyright (c) 2026, Tri Dao), from
quack-kernels 0.6.0.  Quack is distributed under the Apache License 2.0 (see THIRD_PARTY_NOTICES.md).

- `gemm_body` is `kernel` without its two `setmaxregister` requests, so it
  runs in the register split the program already holds; shared memory comes from the
  program's page, and the pipelines' barrier storage is passed explicitly.  Its warp-role
  tests read `cute.arch.warp_idx` and `thread_idx`, which the caller relabels for the role
  (`gemm_role_rotation`).
- `gemm_body_for_warpgroup` is `kernel` with its two warp-role branches chosen by the
  constexpr physical warpgroup, no `setmaxregister` requests, the program's page, and the
  pipelines' barrier storage passed explicitly.
- `down_dx_bf16_epilogue` is `epilogue` for the down-projection dX: it rounds
  each accumulator subtile to BF16 through the first auxiliary output's shared-memory stage
  and reloads it before the SwiGLU-backward visit, acquiring the store buffer before that
  round trip instead of after the visit.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Optional, Tuple

import cutlass
import cutlass.cute as cute
import quack.sm90_utils as quack_sm90_utils
from cutlass import Boolean, Int32, const_expr, pipeline
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline.helpers import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.smem_allocator import SmemPartition
from quack import copy_utils
from quack.pipeline import make_pipeline_state
from quack.rounding import RoundingMode, epilogue_sr_seed
from quack.varlen_utils import VarlenManager

from program_smem import ProgramSmemAllocator


@cute.jit
def gemm_body(
    self,
    tiled_mma: cute.TiledMma,
    tma_atom_a: Optional[cute.CopyAtom],
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    tma_atom_d: Optional[cute.CopyAtom],
    mD_mnl: Optional[cute.Tensor],
    tma_atom_c: Optional[cute.CopyAtom],
    mC_mnl: Optional[cute.Tensor],
    epilogue_params,
    varlen_params: VarlenManager.Params,
    cluster_layout_mnk: cute.Layout,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    epi_smem_layout: cute.ComposedLayout,
    epi_c_smem_layout: cute.ComposedLayout,
    tile_sched_params,
    TileSchedulerCls: cutlass.Constexpr[Callable],
):
    """Quack's persistent GEMM for the role the caller is tracing.

    Takes the 18 arguments a member's `prepare_*` method returns, with the caller's scheduler
    parameters and class as the last two.  Quack's role tests read `cute.arch.warp_idx` and
    `thread_idx`; the caller relabels them (`gemm_role_rotation.push_role_rotation`) so that
    role 0 runs the A/B producer and roles 1 and 2 the two MMA warpgroups.
    """
    from cutlass.cute.experimental import iket

    varlen_m = const_expr(varlen_params.cu_seqlens_m is not None)
    varlen_k = const_expr(varlen_params.cu_seqlens_k is not None)
    assert not (varlen_m and varlen_k)
    if const_expr(self.gather_A):
        assert varlen_m or varlen_k
    has_D = const_expr(mD_mnl is not None)
    has_C = const_expr(mC_mnl is not None)
    has_epi_load = const_expr(self.epi_c_stage > 0)
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == self.ab_load_warp_id:
        for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_d, tma_atom_c):
            if const_expr(tma_atom is not None):
                cpasync.prefetch_descriptor(tma_atom)
    smem = ProgramSmemAllocator()
    storage = smem.allocate(self.shared_storage)
    ab_pipeline = self.make_ab_pipeline(
        tiled_mma=tiled_mma,
        cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)),
        ab_pipeline_mbar_ptr=storage.ab_pipeline_array_ptr.data_ptr(),
    )
    epi_pipeline = None
    if const_expr(has_epi_load):
        epi_pipeline = self.make_epi_pipeline(
            tx_count=self.epi_load_bytes_per_stage,
            epi_pipeline_mbar_ptr=storage.epi_pipeline_array_ptr.data_ptr(),
        )
    sched_pipeline = None
    sched_data = None
    if const_expr(self.is_persistent):
        sched_pipeline = self.make_sched_pipeline(
            cluster_layout_mnk,
            varlen_k=varlen_k,
            sched_pipeline_mbar_ptr=storage.sched_pipeline_array_ptr.data_ptr(),
        )
        sched_data = smem.allocate_tensor(
            Int32,
            cute.make_layout((4, self.sched_stage)),
            byte_alignment=16,
            partition=SmemPartition.RESERVED,
        )
    pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk[:-1], is_relaxed=True)
    sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
    sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
    sD = None
    if const_expr(has_D):
        sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
    sC = None
    if const_expr(has_C):
        sC = storage.sC.get_tensor(epi_c_smem_layout.outer, swizzle=epi_c_smem_layout.inner)
    epi_smem_tensors = self.epi_get_smem_tensors(epilogue_params, storage)
    varlen_manager = VarlenManager.create(
        varlen_params,
        len_m_static=Int32(
            cute.size(mA_mkl, mode=[0])
            if varlen_k or varlen_params.mAIdx is None
            else varlen_params.mAIdx.shape[0]
        ),
        len_k_static=Int32(cute.size(mA_mkl, mode=[1])),
    )
    TileSchedulerCls = partial(
        TileSchedulerCls.create, tile_sched_params, sched_data, sched_pipeline
    )
    pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])
    if warp_idx >= self.ab_load_warp_id:
        if (
            warp_idx >= self.ab_load_warp_id
            and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps
        ):
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            a_tma_multicast = {"cluster_shape": self.cluster_shape_mnk[:2], "multicast_dim": "M"}
            b_tma_multicast = {"cluster_shape": self.cluster_shape_mnk[:2], "multicast_dim": "N"}
            is_scheduler_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
            if const_expr(cute.size(cluster_layout_mnk) > 1):
                is_scheduler_warp = is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            ab_producer_state = make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            while work_tile.is_valid_tile:
                iket.range_push("tma_load")
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                copy_A, prefetch_A = (None, None)
                if const_expr(not self.gather_A):
                    mA_mk = varlen_manager.offset_batch_A(mA_mkl, batch_idx)
                    gA_mk = cute.local_tile(
                        mA_mk,
                        cute.select(self.cta_tile_shape_mnk, [0, 2]),
                        (tile_coord_mnkl[0], None),
                    )
                    copy_A = copy_utils.tma_get_block_copy_fn(
                        tma_atom_a, src_tensor=gA_mk, dst_tensor=sA, tma_multicast=a_tma_multicast
                    )
                else:
                    copy_A, prefetch_A = self._make_gather_A_copy(
                        mA_mkl, sA, varlen_manager, tile_coord_mnkl, batch_idx
                    )
                gB_nk = cute.local_tile(
                    varlen_manager.offset_batch_B(mB_nkl, batch_idx),
                    cute.select(self.cta_tile_shape_mnk, [1, 2]),
                    (tile_coord_mnkl[1], None),
                )
                copy_B = copy_utils.tma_get_block_copy_fn(
                    tma_atom_b, src_tensor=gB_nk, dst_tensor=sB, tma_multicast=b_tma_multicast
                )
                len_k = varlen_manager.len_k(batch_idx)
                k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                if const_expr(not self.gather_A):
                    ab_producer_state = self.load_tma(
                        ab_pipeline, ab_producer_state, [copy_A, copy_B], k_tile_cnt
                    )
                else:
                    ab_producer_state = self.load_AB_gather_A(
                        ab_pipeline,
                        ab_producer_state,
                        copy_A,
                        prefetch_A,
                        copy_B,
                        k_tile_cnt,
                        varlen_m=varlen_m,
                    )
                iket.range_pop()
                tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                work_tile = tile_scheduler.get_current_work()
            if const_expr(self.pingpong and (not varlen_k)):
                if is_scheduler_warp:
                    tile_scheduler.write_work_tile_to_smem(work_tile)
                work_tile = tile_scheduler.get_current_work()
            if warp_idx == self.ab_load_warp_id:
                ab_pipeline.producer_tail(ab_producer_state)
            if is_scheduler_warp:
                tile_scheduler.producer_tail()
    if warp_idx < self.ab_load_warp_id:
        is_tma_warp = Boolean(
            not self.pingpong
            and warp_idx == 0
            or (self.pingpong and (warp_idx == 0 or warp_idx == 4))
        )
        tidx, _, _ = cute.arch.thread_idx()
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        if const_expr(self.pingpong):
            tidx = tidx % self.num_threads_per_warp_group
        warp_group_thread_layout = cute.make_layout(
            self.mma_warp_groups if const_expr(not self.pingpong) else 1,
            stride=self.num_threads_per_warp_group,
        )
        thr_mma = tiled_mma.get_slice(
            warp_group_thread_layout(warp_group_idx if not self.pingpong else 0)
        )
        acc, tCrA, tCrB = quack_sm90_utils.partition_fragment_ABC(
            thr_mma, self.cta_tile_shape_mnk, sA, sB
        )
        acc_slow = None
        if const_expr(self.fp8_slow_accum):
            acc_slow = cute.make_rmem_tensor(acc.shape, self.acc_dtype)
        mma_fn = partial(quack_sm90_utils.gemm_w_idx, tiled_mma, acc, tCrA, tCrB)
        if const_expr(self.pingpong):
            if warp_group_idx == 0:
                self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")
        k_tile_cnt_static = cute.ceil_div(cute.size(mA_mkl, mode=[1]), self.cta_tile_shape_mnk[2])
        c_tile_cnt = cute.size(self.epi_tile_shape)
        ab_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
        epi_store_pipeline = self.make_epi_store_pipeline()
        epi_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.epi_c_stage)
        epi_producer_state = make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.epi_c_stage
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        if const_expr(self.pingpong):
            if warp_idx >= 4:
                epi_read_state.advance_iters(c_tile_cnt)
                epi_producer_state.advance_iters(c_tile_cnt)
                if const_expr(not varlen_k):
                    ab_read_state.advance_iters(k_tile_cnt_static)
                else:
                    len_k = varlen_manager.len_k(batch_idx=work_tile.tile_idx[3])
                    k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                    ab_read_state.advance_iters(k_tile_cnt)
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
        while work_tile.is_valid_tile:
            tile_coord_mnkl = work_tile.tile_idx
            batch_idx = tile_coord_mnkl[3]
            len_k = varlen_manager.len_k(batch_idx)
            k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
            if const_expr(self.pingpong):
                self.pingpong_barrier_sync(warp_group_idx, stage="mma")
            iket.range_push("mma")
            ab_read_state = self.mma(
                ab_pipeline, ab_read_state, mma_fn, acc, acc_slow, k_tile_cnt, warp_group_idx
            )
            if const_expr(varlen_k):
                if k_tile_cnt == 0:
                    acc.fill(0.0)
            iket.range_pop()
            if const_expr(self.pingpong):
                self.pingpong_barrier_sync(warp_group_idx, "epi")
            iket.range_push("epilogue")
            copy_D = None
            if const_expr(has_D):
                copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                    tma_atom_d,
                    varlen_manager.offset_batch_epi(mD_mnl, batch_idx),
                    self.cta_tile_shape_mnk[:2],
                    self.epi_tile,
                    sD,
                    tile_coord_mnkl,
                )
            copy_C = None
            if const_expr(has_C):
                copy_C_fn, _, _ = self.epilog_gmem_copy_and_partition(
                    tma_atom_c,
                    varlen_manager.offset_batch_epi(mC_mnl, batch_idx),
                    self.cta_tile_shape_mnk[:2],
                    self.epi_tile,
                    sC,
                    tile_coord_mnkl,
                )
                copy_C = copy_utils.tma_producer_copy_fn(copy_C_fn, epi_pipeline)
            if const_expr(has_epi_load):
                tile_load_copy_fns = self.epi_tile_load_g2s_copy_fns(
                    epilogue_params, epi_smem_tensors, tile_coord_mnkl, varlen_manager, epi_pipeline
                )
                copy_C = copy_utils.chain_tma_producer_copy_fns((copy_C, *tile_load_copy_fns))
            d_dtype_for_layout = self.d_dtype if self.d_dtype is not None else cutlass.BFloat16
            tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                tiled_mma, self.d_layout, d_dtype_for_layout, sD, tidx
            )
            tRS_rAcc = self.epi_retile_acc(acc, tRS_rD, tiled_copy_r2s)
            load_acc_subtile = partial(self.epi_load_acc_subtile, tRS_rAcc)
            if const_expr(has_C):
                tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC = self.epilog_smem_load_and_partition(
                    tiled_mma, self.c_layout, self.c_dtype, sC, tRS_rD.layout, tidx
                )
            else:
                tiled_copy_s2r, tSR_sC, tRS_rC, tSR_rC = (None, None, None, None)
            self.epi_visit_acc(epilogue_params, acc, tiled_mma, tile_coord_mnkl, tidx)
            epi_read_state, epi_producer_state = self.epilogue(
                epilogue_params,
                epi_smem_tensors,
                epi_pipeline,
                epi_store_pipeline,
                epi_read_state,
                epi_producer_state,
                self.epi_tile,
                load_acc_subtile,
                tRS_rD,
                tRS_rC,
                None,
                tiled_copy_r2s,
                tRS_sD,
                tiled_copy_s2r,
                tSR_rC,
                tSR_sC,
                copy_D,
                copy_C,
                tile_coord_mnkl,
                varlen_manager,
                self.epilogue_barrier,
                tile_scheduler,
                tidx,
                is_tma_warp,
            )
            if const_expr(self.pingpong):
                if is_tma_warp:
                    epi_store_pipeline.producer_tail()
                self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")
            iket.range_pop()
            if const_expr(not self.pingpong):
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
            else:
                epi_read_state.advance_iters(c_tile_cnt)
                epi_producer_state.advance_iters(c_tile_cnt)
                if const_expr(not varlen_k):
                    ab_read_state.advance_iters(k_tile_cnt_static)
                    tile_scheduler.advance_to_next_work(advance_count=self.mma_warp_groups)
                    work_tile = tile_scheduler.get_current_work()
                else:
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                    if work_tile.is_valid_tile:
                        len_k = varlen_manager.len_k(batch_idx=work_tile.tile_idx[3])
                        k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                        ab_read_state.advance_iters(k_tile_cnt)
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        if const_expr(not self.pingpong):
            if is_tma_warp:
                epi_store_pipeline.producer_tail()


@cute.jit
def gemm_body_for_warpgroup(
    self,
    tiled_mma: cute.TiledMma,
    tma_atom_a: Optional[cute.CopyAtom],
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    tma_atom_d: Optional[cute.CopyAtom],
    mD_mnl: Optional[cute.Tensor],
    tma_atom_c: Optional[cute.CopyAtom],
    mC_mnl: Optional[cute.Tensor],
    epilogue_params,
    varlen_params: VarlenManager.Params,
    cluster_layout_mnk: cute.Layout,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    epi_smem_layout: cute.ComposedLayout,
    epi_c_smem_layout: cute.ComposedLayout,
    tile_sched_params,
    TileSchedulerCls: cutlass.Constexpr[Callable],
    physical_warpgroup: cutlass.Constexpr[int],
):
    """`gemm_body` in Quack's own warp layout, for the constexpr `physical_warpgroup`.

    Warpgroup 2 runs the A/B producer and warpgroups 0 and 1 the MMA warpgroups, as in Quack;
    the warp tests inside each branch read the physical warp index.  The LM head runs its
    GEMMs this way, under its own register split.
    """
    from cutlass.cute.experimental import iket

    varlen_m = const_expr(varlen_params.cu_seqlens_m is not None)
    varlen_k = const_expr(varlen_params.cu_seqlens_k is not None)
    assert not (varlen_m and varlen_k)
    if const_expr(self.gather_A):
        assert varlen_m or varlen_k
    has_D = const_expr(mD_mnl is not None)
    has_C = const_expr(mC_mnl is not None)
    has_epi_load = const_expr(self.epi_c_stage > 0)
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == self.ab_load_warp_id:
        for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_d, tma_atom_c):
            if const_expr(tma_atom is not None):
                cpasync.prefetch_descriptor(tma_atom)
    smem = ProgramSmemAllocator()
    storage = smem.allocate(self.shared_storage)
    ab_pipeline = self.make_ab_pipeline(
        tiled_mma=tiled_mma,
        cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)),
        ab_pipeline_mbar_ptr=storage.ab_pipeline_array_ptr.data_ptr(),
    )
    epi_pipeline = None
    if const_expr(has_epi_load):
        epi_pipeline = self.make_epi_pipeline(
            tx_count=self.epi_load_bytes_per_stage,
            epi_pipeline_mbar_ptr=storage.epi_pipeline_array_ptr.data_ptr(),
        )
    sched_pipeline = None
    sched_data = None
    if const_expr(self.is_persistent):
        sched_pipeline = self.make_sched_pipeline(
            cluster_layout_mnk,
            varlen_k=varlen_k,
            sched_pipeline_mbar_ptr=storage.sched_pipeline_array_ptr.data_ptr(),
        )
        sched_data = smem.allocate_tensor(
            Int32,
            cute.make_layout((4, self.sched_stage)),
            byte_alignment=16,
            partition=SmemPartition.RESERVED,
        )
    pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk[:-1], is_relaxed=True)
    sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
    sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
    sD = None
    if const_expr(has_D):
        sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
    sC = None
    if const_expr(has_C):
        sC = storage.sC.get_tensor(epi_c_smem_layout.outer, swizzle=epi_c_smem_layout.inner)
    epi_smem_tensors = self.epi_get_smem_tensors(epilogue_params, storage)
    varlen_manager = VarlenManager.create(
        varlen_params,
        len_m_static=Int32(
            cute.size(mA_mkl, mode=[0])
            if varlen_k or varlen_params.mAIdx is None
            else varlen_params.mAIdx.shape[0]
        ),
        len_k_static=Int32(cute.size(mA_mkl, mode=[1])),
    )
    TileSchedulerCls = partial(
        TileSchedulerCls.create, tile_sched_params, sched_data, sched_pipeline
    )
    pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])
    if const_expr(physical_warpgroup == 2):
        if (
            warp_idx >= self.ab_load_warp_id
            and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps
        ):
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            a_tma_multicast = {"cluster_shape": self.cluster_shape_mnk[:2], "multicast_dim": "M"}
            b_tma_multicast = {"cluster_shape": self.cluster_shape_mnk[:2], "multicast_dim": "N"}
            is_scheduler_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
            if const_expr(cute.size(cluster_layout_mnk) > 1):
                is_scheduler_warp = is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            ab_producer_state = make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            while work_tile.is_valid_tile:
                iket.range_push("tma_load")
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                copy_A, prefetch_A = (None, None)
                if const_expr(not self.gather_A):
                    mA_mk = varlen_manager.offset_batch_A(mA_mkl, batch_idx)
                    gA_mk = cute.local_tile(
                        mA_mk,
                        cute.select(self.cta_tile_shape_mnk, [0, 2]),
                        (tile_coord_mnkl[0], None),
                    )
                    copy_A = copy_utils.tma_get_block_copy_fn(
                        tma_atom_a, src_tensor=gA_mk, dst_tensor=sA, tma_multicast=a_tma_multicast
                    )
                else:
                    copy_A, prefetch_A = self._make_gather_A_copy(
                        mA_mkl, sA, varlen_manager, tile_coord_mnkl, batch_idx
                    )
                gB_nk = cute.local_tile(
                    varlen_manager.offset_batch_B(mB_nkl, batch_idx),
                    cute.select(self.cta_tile_shape_mnk, [1, 2]),
                    (tile_coord_mnkl[1], None),
                )
                copy_B = copy_utils.tma_get_block_copy_fn(
                    tma_atom_b, src_tensor=gB_nk, dst_tensor=sB, tma_multicast=b_tma_multicast
                )
                len_k = varlen_manager.len_k(batch_idx)
                k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                if const_expr(not self.gather_A):
                    ab_producer_state = self.load_tma(
                        ab_pipeline, ab_producer_state, [copy_A, copy_B], k_tile_cnt
                    )
                else:
                    ab_producer_state = self.load_AB_gather_A(
                        ab_pipeline,
                        ab_producer_state,
                        copy_A,
                        prefetch_A,
                        copy_B,
                        k_tile_cnt,
                        varlen_m=varlen_m,
                    )
                iket.range_pop()
                tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                work_tile = tile_scheduler.get_current_work()
            if const_expr(self.pingpong and (not varlen_k)):
                if is_scheduler_warp:
                    tile_scheduler.write_work_tile_to_smem(work_tile)
                work_tile = tile_scheduler.get_current_work()
            if warp_idx == self.ab_load_warp_id:
                ab_pipeline.producer_tail(ab_producer_state)
            if is_scheduler_warp:
                tile_scheduler.producer_tail()
    if const_expr(physical_warpgroup < 2):
        is_tma_warp = Boolean(
            not self.pingpong
            and warp_idx == 0
            or (self.pingpong and (warp_idx == 0 or warp_idx == 4))
        )
        tidx, _, _ = cute.arch.thread_idx()
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        if const_expr(self.pingpong):
            tidx = tidx % self.num_threads_per_warp_group
        warp_group_thread_layout = cute.make_layout(
            self.mma_warp_groups if const_expr(not self.pingpong) else 1,
            stride=self.num_threads_per_warp_group,
        )
        thr_mma = tiled_mma.get_slice(
            warp_group_thread_layout(warp_group_idx if not self.pingpong else 0)
        )
        acc, tCrA, tCrB = quack_sm90_utils.partition_fragment_ABC(
            thr_mma, self.cta_tile_shape_mnk, sA, sB
        )
        acc_slow = None
        if const_expr(self.fp8_slow_accum):
            acc_slow = cute.make_rmem_tensor(acc.shape, self.acc_dtype)
        mma_fn = partial(quack_sm90_utils.gemm_w_idx, tiled_mma, acc, tCrA, tCrB)
        if const_expr(self.pingpong):
            if warp_group_idx == 0:
                self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")
        k_tile_cnt_static = cute.ceil_div(cute.size(mA_mkl, mode=[1]), self.cta_tile_shape_mnk[2])
        c_tile_cnt = cute.size(self.epi_tile_shape)
        ab_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
        epi_store_pipeline = self.make_epi_store_pipeline()
        epi_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.epi_c_stage)
        epi_producer_state = make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.epi_c_stage
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        if const_expr(self.pingpong):
            if warp_idx >= 4:
                epi_read_state.advance_iters(c_tile_cnt)
                epi_producer_state.advance_iters(c_tile_cnt)
                if const_expr(not varlen_k):
                    ab_read_state.advance_iters(k_tile_cnt_static)
                else:
                    len_k = varlen_manager.len_k(batch_idx=work_tile.tile_idx[3])
                    k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                    ab_read_state.advance_iters(k_tile_cnt)
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
        while work_tile.is_valid_tile:
            tile_coord_mnkl = work_tile.tile_idx
            batch_idx = tile_coord_mnkl[3]
            len_k = varlen_manager.len_k(batch_idx)
            k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
            if const_expr(self.pingpong):
                self.pingpong_barrier_sync(warp_group_idx, stage="mma")
            iket.range_push("mma")
            ab_read_state = self.mma(
                ab_pipeline, ab_read_state, mma_fn, acc, acc_slow, k_tile_cnt, warp_group_idx
            )
            if const_expr(varlen_k):
                if k_tile_cnt == 0:
                    acc.fill(0.0)
            iket.range_pop()
            if const_expr(self.pingpong):
                self.pingpong_barrier_sync(warp_group_idx, "epi")
            iket.range_push("epilogue")
            copy_D = None
            if const_expr(has_D):
                copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                    tma_atom_d,
                    varlen_manager.offset_batch_epi(mD_mnl, batch_idx),
                    self.cta_tile_shape_mnk[:2],
                    self.epi_tile,
                    sD,
                    tile_coord_mnkl,
                )
            copy_C = None
            if const_expr(has_C):
                copy_C_fn, _, _ = self.epilog_gmem_copy_and_partition(
                    tma_atom_c,
                    varlen_manager.offset_batch_epi(mC_mnl, batch_idx),
                    self.cta_tile_shape_mnk[:2],
                    self.epi_tile,
                    sC,
                    tile_coord_mnkl,
                )
                copy_C = copy_utils.tma_producer_copy_fn(copy_C_fn, epi_pipeline)
            if const_expr(has_epi_load):
                tile_load_copy_fns = self.epi_tile_load_g2s_copy_fns(
                    epilogue_params, epi_smem_tensors, tile_coord_mnkl, varlen_manager, epi_pipeline
                )
                copy_C = copy_utils.chain_tma_producer_copy_fns((copy_C, *tile_load_copy_fns))
            d_dtype_for_layout = self.d_dtype if self.d_dtype is not None else cutlass.BFloat16
            tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                tiled_mma, self.d_layout, d_dtype_for_layout, sD, tidx
            )
            tRS_rAcc = self.epi_retile_acc(acc, tRS_rD, tiled_copy_r2s)
            load_acc_subtile = partial(self.epi_load_acc_subtile, tRS_rAcc)
            if const_expr(has_C):
                tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC = self.epilog_smem_load_and_partition(
                    tiled_mma, self.c_layout, self.c_dtype, sC, tRS_rD.layout, tidx
                )
            else:
                tiled_copy_s2r, tSR_sC, tRS_rC, tSR_rC = (None, None, None, None)
            self.epi_visit_acc(epilogue_params, acc, tiled_mma, tile_coord_mnkl, tidx)
            epi_read_state, epi_producer_state = self.epilogue(
                epilogue_params,
                epi_smem_tensors,
                epi_pipeline,
                epi_store_pipeline,
                epi_read_state,
                epi_producer_state,
                self.epi_tile,
                load_acc_subtile,
                tRS_rD,
                tRS_rC,
                None,
                tiled_copy_r2s,
                tRS_sD,
                tiled_copy_s2r,
                tSR_rC,
                tSR_sC,
                copy_D,
                copy_C,
                tile_coord_mnkl,
                varlen_manager,
                self.epilogue_barrier,
                tile_scheduler,
                tidx,
                is_tma_warp,
            )
            if const_expr(self.pingpong):
                if is_tma_warp:
                    epi_store_pipeline.producer_tail()
                self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")
            iket.range_pop()
            if const_expr(not self.pingpong):
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
            else:
                epi_read_state.advance_iters(c_tile_cnt)
                epi_producer_state.advance_iters(c_tile_cnt)
                if const_expr(not varlen_k):
                    ab_read_state.advance_iters(k_tile_cnt_static)
                    tile_scheduler.advance_to_next_work(advance_count=self.mma_warp_groups)
                    work_tile = tile_scheduler.get_current_work()
                else:
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                    if work_tile.is_valid_tile:
                        len_k = varlen_manager.len_k(batch_idx=work_tile.tile_idx[3])
                        k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                        ab_read_state.advance_iters(k_tile_cnt)
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        if const_expr(not self.pingpong):
            if is_tma_warp:
                epi_store_pipeline.producer_tail()


@cute.jit
def down_dx_bf16_epilogue(
    self,
    params: EpilogueParams,  # noqa: F821 - Quack's annotation, never evaluated
    epi_smem_tensors: Dict[str, cute.Tensor],
    epi_pipeline: Optional[cutlass.pipeline.PipelineAsync],
    epi_store_pipeline: Optional[cutlass.pipeline.PipelineAsync],
    epi_read_state: Optional[cutlass.pipeline.PipelineState],
    epi_producer_state: Optional[cutlass.pipeline.PipelineState],
    epi_tile: cute.Tile,
    load_acc_subtile: Callable,
    tRS_rD: cute.Tensor,
    tRS_rC: Optional[cute.Tensor],
    tiled_copy_t2r: Optional[cute.TiledCopy],
    tiled_copy_r2s: cute.TiledCopy,
    tRS_sD: cute.Tensor,
    tiled_copy_s2r: Optional[cute.ThrCopy],
    tSR_rC: Optional[cute.Tensor],
    tSR_sC: Optional[cute.Tensor],
    copy_D: Optional[Callable],
    copy_C: Optional[Callable],
    tile_coord_mnkl: cute.Coord,
    varlen_manager: VarlenManager,
    epilogue_barrier: cutlass.pipeline.NamedBarrier,
    tile_scheduler,
    tidx: Int32,
    is_tma_warp: cutlass.Boolean,
) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState]:
    """Quack's `epilogue`, with each accumulator subtile rounded to BF16 before the visit.

    The subtile goes as BF16 into the first auxiliary output's (`mDGateHalf`) shared-memory
    stage and is read back, so the visit starts from BF16 values loaded from shared memory.
    The subtile's outputs are later stored from the same stage index, so the store buffer is
    acquired before the round trip rather than after the visit.  The member must have two
    auxiliary outputs.
    """
    has_C = const_expr(tRS_rC is not None)
    has_epi_load = const_expr(self.epi_c_stage > 0)
    has_D = const_expr(copy_D is not None)
    use_tma_epi = const_expr(epi_store_pipeline is not None)
    use_tma_c = const_expr(epi_pipeline is not None)
    inline_epi_load = const_expr(copy_C is not None)
    use_stochastic_rounding = const_expr(
        self.rounding_mode == RoundingMode.RS
        and self.acc_dtype == cutlass.Float32
        and (self.d_dtype == cutlass.BFloat16)
    )
    aux_out_ctxs = self.epi_setup_aux_out(
        params,
        epi_smem_tensors,
        tiled_copy_r2s,
        tiled_copy_t2r,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    )
    assert len(aux_out_ctxs) == 2, "down-dX lifetime cut requires two aux outputs"
    tiled_copy_dy_r2s = aux_out_ctxs[0][0]
    tRS_sDy = aux_out_ctxs[0][1]
    sDy = epi_smem_tensors["mDGateHalf"]
    copy_atom_dy_s2r = copy_utils.sm90_get_smem_load_op(self.aux_out_layout, self.aux_out_dtype)
    tiled_copy_dy_s2r = cute.make_tiled_copy_S(copy_atom_dy_s2r, tiled_copy_dy_r2s)
    thr_copy_dy_s2r = tiled_copy_dy_s2r.get_slice(tidx)
    tSR_sDy = thr_copy_dy_s2r.partition_S(sDy)
    tRS_rDy = cute.make_rmem_tensor_like(tRS_rD, self.aux_out_dtype)
    tSR_rDy = thr_copy_dy_s2r.retile(tRS_rDy)
    epi_tile_shape = cute.zipped_divide(
        cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile
    ).shape[1]
    epi_tile_layout = cute.make_ordered_layout(
        epi_tile_shape, order=(0, 1) if const_expr(self.epi_m_major) else (1, 0)
    )
    epi_tile_num = cute.size(epi_tile_shape)
    num_prev_subtiles = tile_scheduler.num_tiles_executed * epi_tile_num
    epi_tensors = self.epi_begin(
        params,
        epi_smem_tensors,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        epilogue_barrier,
        tidx,
        tRS_rD.layout,
    )
    if const_expr(inline_epi_load):
        for epi_idx in cutlass.range(min(epi_tile_num, self.epi_c_stage), unroll=1):
            epi_coord_C = epi_tile_layout.get_hier_coord(epi_idx)
            if const_expr(use_tma_c):
                if is_tma_warp:
                    epi_pipeline.producer_acquire(epi_producer_state)
                    copy_C(src_idx=epi_coord_C, producer_state=epi_producer_state)
                    epi_pipeline.producer_commit(epi_producer_state)
                epi_producer_state.advance()
            else:
                copy_C(src_idx=epi_coord_C, dst_idx=epi_idx % self.epi_c_stage)
        if const_expr(use_tma_c):
            epilogue_barrier.arrive_and_wait()
    for epi_idx in cutlass.range_constexpr(epi_tile_num):
        epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
        load_acc_subtile(tRS_rD, epi_coord)
        epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
        if const_expr(use_tma_epi):
            if is_tma_warp:
                epi_store_pipeline.producer_acquire()
        else:
            epilogue_barrier.arrive_and_wait()
        if const_expr(use_tma_epi):
            epilogue_barrier.arrive_and_wait()
        tRS_rDy.store(tRS_rD.load().to(self.aux_out_dtype))
        cute.copy(
            tiled_copy_dy_r2s,
            copy_utils.contiguous(tiled_copy_dy_r2s.retile(tRS_rDy)),
            tRS_sDy[None, None, None, epi_buffer],
        )
        cute.arch.fence_view_async_shared()
        epilogue_barrier.arrive_and_wait()
        cute.copy(tiled_copy_dy_s2r, tSR_sDy[None, None, None, epi_buffer], tSR_rDy)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        tRS_rD.store(tRS_rDy.load().to(self.acc_dtype))
        if const_expr(has_epi_load):
            if const_expr(use_tma_c):
                epi_pipeline.consumer_wait(epi_read_state)
                if const_expr(has_C):
                    cute.copy(
                        tiled_copy_s2r, tSR_sC[None, None, None, epi_read_state.index], tSR_rC
                    )
                self.epi_tile_load_s2r(params, epi_tensors, epi_read_state.index)
                cute.arch.fence_view_async_shared()
                epi_pipeline.consumer_release(epi_read_state)
                epi_read_state.advance()
            else:
                c_buffer = epi_idx % self.epi_c_stage
                cute.copy(tiled_copy_s2r, tSR_sC[None, None, None, c_buffer], tSR_rC)
                epilogue_barrier.arrive_and_wait()
        epi_loop_tensors = self.epi_begin_loop(params, epi_tensors, epi_coord)
        if const_expr(inline_epi_load and epi_idx + self.epi_c_stage < epi_tile_num):
            epi_coord_C = epi_tile_layout.get_hier_coord(epi_idx + self.epi_c_stage)
            if const_expr(use_tma_c):
                if is_tma_warp:
                    epi_pipeline.producer_acquire(epi_producer_state)
                    copy_C(src_idx=epi_coord_C, producer_state=epi_producer_state)
                    epi_pipeline.producer_commit(epi_producer_state)
                epi_producer_state.advance()
            else:
                epilogue_barrier.arrive_and_wait()
                copy_C(src_idx=epi_coord_C, dst_idx=(epi_idx + self.epi_c_stage) % self.epi_c_stage)
        tRS_rAuxOuts = self.epi_visit_subtile(params, epi_loop_tensors, tRS_rD, tRS_rC)
        self.epi_end_loop(
            params,
            epi_tensors,
            epi_coord,
            epi_tile,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tile_coord_mnkl,
            varlen_manager,
            tidx,
        )
        tRS_rAuxOuts_out = tuple(
            (
                self.epi_convert_aux_out(
                    i,
                    tRS_rAuxOuts[i],
                    epi_loop_tensors.get("sr_seed"),
                    tidx,
                    tile_coord_mnkl,
                    num_prev_subtiles,
                    epi_idx,
                )
                for i in range(len(aux_out_ctxs))
            )
        )
        if const_expr(has_D):
            tRS_sD_cur = tRS_sD[None, None, None, epi_buffer]
            if const_expr(use_stochastic_rounding):
                seed = epilogue_sr_seed(
                    epi_loop_tensors.get("sr_seed"), tile_coord_mnkl, num_prev_subtiles + epi_idx
                )
                copy_utils.sr_cvt_copy(tiled_copy_r2s, tRS_rD, tRS_sD_cur, seed, tidx)
            else:
                copy_utils.cvt_copy(tiled_copy_r2s, tRS_rD, tRS_sD_cur)
        for i in cutlass.range_constexpr(len(aux_out_ctxs)):
            tiled_copy_aux_out_r2s, tRS_sAuxOut, _ = aux_out_ctxs[i]
            cute.copy(
                tiled_copy_aux_out_r2s,
                tiled_copy_aux_out_r2s.retile(tRS_rAuxOuts_out[i]).contiguous(),
                tRS_sAuxOut[None, None, None, epi_buffer],
            )
        if const_expr(use_tma_epi):
            cute.arch.fence_view_async_shared()
            epilogue_barrier.arrive_and_wait()
            if is_tma_warp:
                if const_expr(has_D):
                    copy_D(src_idx=epi_buffer, dst_idx=epi_coord)
                for i in cutlass.range_constexpr(len(aux_out_ctxs)):
                    _, _, copy_aux_out = aux_out_ctxs[i]
                    copy_aux_out(src_idx=epi_buffer, dst_idx=epi_coord)
                epi_store_pipeline.producer_commit()
        else:
            epilogue_barrier.arrive_and_wait()
            if const_expr(has_D):
                copy_D(src_idx=epi_buffer, dst_idx=epi_coord)
            for i in cutlass.range_constexpr(len(aux_out_ctxs)):
                _, _, copy_aux_out = aux_out_ctxs[i]
                copy_aux_out(src_idx=epi_buffer, dst_idx=epi_coord)
            epilogue_barrier.arrive_and_wait()
    self.epi_end(
        params,
        epi_tensors,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    )
    return (epi_read_state, epi_producer_state)
