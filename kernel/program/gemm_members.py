"""The program's Quack SM90 GEMM members and the shared-memory page they are sized for.

A GEMM member is Quack's persistent GEMM run as a device function of the program:
`quack_gemm_bodies` holds the kernel body, and a member class computes what Quack's host code
would compute before a launch.  `ProgramPageGemmMixin` sizes the pipeline stages from the
program's page (through Quack's own `_compute_stages`) and builds the pipelines on mbarriers in
the member's shared storage.  `DefaultGemmMember` (Quack's default epilogue),
`ResidualGemmMember` (the default epilogue with a C operand) and `GatedGemmMember` (Quack's
gated epilogue) are the bases of every GEMM member; their `prepare_*` methods return the body's
17 arguments instead of launching a kernel.

Adapted from `GemmSm90.__call__` and `GemmSm90.make_sched_pipeline` in quack/gemm_sm90.py
(Copyright (c) 2025-2026, QuACK team) and `GemmTmaBase.make_ab_pipeline` and
`GemmTmaBase.make_epi_pipeline` in quack/gemm_base.py (Copyright (c) 2026, Tri Dao), from
quack-kernels 0.6.0.  Quack is distributed under the Apache License 2.0 (see
THIRD_PARTY_NOTICES.md).

- `make_ab_pipeline`, `make_epi_pipeline` and `make_sched_pipeline` are Quack's builders with an
  explicit `barrier_storage` pointer.
- `prepare_kernel_arguments` and the `prepare_generation_kernel_arguments` methods are
  `__call__` up to the launch: they return the kernel's arguments, add the three pipelines'
  mbarrier arrays and the scheduler scratch to `SharedStorage` (`ProgramSmemAllocator` finds
  the scratch field, `sched_data`, by name), and leave out the dtype checks, concat layouts,
  varlen and gather_A.  The generation variants declare a one-batch problem whose batch index
  comes from a device table.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Int32, const_expr
from cutlass.utils import LayoutEnum
from quack.gemm_act import GemmGatedSm90
from quack.gemm_default_epi import GemmDefaultSm90
from quack.pipeline import PipelineTmaAsync, PipelineTmaCpAsync
from quack.tile_scheduler import PersistenceMode, TileSchedulerArguments, TileSchedulerOptions
from quack.varlen_utils import VarlenArguments, VarlenManager


# The launch requests the page plus SMEM_OUTSIDE_PAGE_BYTES of dynamic shared memory
# (kernel/build.py writes the sum into the launch description).  The page is sized so that
# request stays SMEM_PAGE_MARGIN_BYTES under H100's opt-in limit per block; a launch above
# the limit is rejected.  Quack sizes each member's A/B pipeline from this page, not from the
# whole SM.
H100_OPTIN_SMEM_BYTES = 232448
SMEM_OUTSIDE_PAGE_BYTES = 8192
SMEM_PAGE_MARGIN_BYTES = 1024
PROGRAM_SMEM_PAGE_BYTES = (
    H100_OPTIN_SMEM_BYTES - SMEM_OUTSIDE_PAGE_BYTES - SMEM_PAGE_MARGIN_BYTES
)


class ProgramPageGemmMixin:
    """Quack's stage sizing and pipeline construction for a member on the program's page.

    The three pipeline builders take their mbarrier arrays from the member's `SharedStorage`.
    Without `barrier_storage`, CUTLASS allocates a new reserved mbarrier array at every
    pipeline construction; the body is traced once per role, so the roles would not share
    their barriers.
    """

    @classmethod
    def _compute_stages(
        cls,
        cta_tile_shape_mnk,
        epi_tile,
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        epilogue_args,
        smem_capacity,
        occupancy,
        warp_shape_mnk=None,
    ):
        """Quack's stage counts for `PROGRAM_SMEM_PAGE_BYTES`; at least two A/B stages."""

        # Quack passes the SM's opt-in capacity; the page replaces it.
        if smem_capacity != H100_OPTIN_SMEM_BYTES:
            raise AssertionError(f"unexpected SM90 shared-memory capacity {smem_capacity}")
        stages = super()._compute_stages(
            cta_tile_shape_mnk,
            epi_tile,
            a_dtype,
            b_dtype,
            d_dtype,
            c_dtype,
            epilogue_args,
            PROGRAM_SMEM_PAGE_BYTES,
            occupancy,
            warp_shape_mnk,
        )
        if stages[0] < 2:
            raise AssertionError(f"unsafe A/B pipeline stage count: {stages}")
        return stages

    def make_ab_pipeline(
        self,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        ab_pipeline_mbar_ptr=None,
    ):
        producer_count = (
            1 if const_expr(not self.gather_A) else 1 + self.num_ab_load_warps * 32
        )
        producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, producer_count
        )
        multicast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_count = multicast_size * tiled_mma.size // cute.arch.WARP_SIZE
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_count
        )
        pipeline_class = (
            pipeline.PipelineTmaAsync
            if not self.gather_A
            else PipelineTmaCpAsync
        )
        return pipeline_class.create(
            num_stages=self.ab_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.num_tma_load_bytes,
            barrier_storage=ab_pipeline_mbar_ptr,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    def make_epi_pipeline(self, tx_count: int, epi_pipeline_mbar_ptr=None):
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_epi_warps
        )
        return PipelineTmaAsync.create(
            num_stages=self.epi_c_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tx_count,
            barrier_storage=epi_pipeline_mbar_ptr,
            defer_sync=True,
            elect_one_release=True,
            syncwarp_before_release=True,
        )

    def make_sched_pipeline(
        self,
        cluster_layout_mnk: cute.Layout,
        varlen_k: bool,
        sched_pipeline_mbar_ptr=None,
    ):
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cluster_size = cute.size(cluster_layout_mnk)
        consumer_count = (
            (
                self.mma_warp_groups
                if not (self.pingpong and not varlen_k)
                else 1
            )
            * 4
            + self.num_ab_load_warps
        ) * cluster_size
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_count
        )
        return pipeline.PipelineAsync.create(
            num_stages=self.sched_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            barrier_storage=sched_pipeline_mbar_ptr,
            consumer_mask=None if const_expr(cluster_size == 1) else 0,
            defer_sync=True,
        )


class DefaultGemmMember(ProgramPageGemmMixin, GemmDefaultSm90):
    """Quack's default-epilogue GEMM as a program member.

    `prepare_kernel_arguments` covers the whole (M, N, L) problem and
    `prepare_generation_kernel_arguments` one batch slice, picked at run time by word 0 of a
    device table (the layer slot, for the decoder projections).  Both set
    `self.shared_storage`.  The scheduler parameters they return use static persistence and no
    semaphore: every caller of the body rebinds them to a dynamic queue
    (`tile_schedulers.dynamic_scheduler_params`) and passes its own scheduler class.  Quack uses
    `max_active_clusters` only to size its launch grid, so nothing here reads it.
    """

    @cute.jit
    def prepare_kernel_arguments(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        max_active_clusters: Int32,
    ):
        """The body's arguments for the whole problem, as `GemmSm90.__call__` computes them."""

        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.d_dtype = mD.element_type
        self.c_dtype = None
        self.a_layout = LayoutEnum.from_tensor(mA)
        self.b_layout = LayoutEnum.from_tensor(mB)
        self.d_layout = LayoutEnum.from_tensor(mD)
        self.c_layout = None

        epilogue_args = self.EpilogueArguments()
        varlen_args = VarlenArguments()
        self._setup_attributes(epilogue_args)

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b = self.make_tma_load_atoms_and_tensors(
            mA, mB, a_smem_layout, b_smem_layout, False
        )
        self.num_tma_load_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        self.num_tma_load_bytes += cute.size_in_bytes(self.b_dtype, b_smem_layout)

        tma_atom_d, tma_tensor_d, tma_atom_c, tma_tensor_c = self.make_tma_epilogue_atoms_and_tensors(
            mD, None, epilogue_args, False
        )
        epilogue_params = self.epi_to_underlying_arguments(epilogue_args)
        varlen_params = VarlenManager.to_underlying_arguments(varlen_args)
        self.epi_load_bytes_per_stage = self.epi_smem_bytes(
            epilogue_args,
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.epi_smem_warp_shape_mnk(),
        ).c_stage

        scheduler_args = TileSchedulerOptions(
            max_active_clusters=max_active_clusters,
            raster_order=None,
            max_swizzle_size=Int32(8),
            tile_count_semaphore=None,
            batch_idx_permute=None,
        )
        TileSchedulerCls = self.get_scheduler_class(varlen_m=False)
        tile_sched_args = self.get_scheduler_arguments(mA, mB, mD, scheduler_args, varlen_args, epilogue_args)
        tile_sched_params = TileSchedulerCls.to_underlying_arguments(tile_sched_args)

        epi_smem_size = cute.cosize(self.epi_smem_layout_staged)

        @cute.struct
        class SharedStorage:
            ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            epi_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.epi_c_stage * 2]
            sched_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.sched_stage * 2]
            sched_data: cute.struct.MemRange[Int32, self.sched_stage * 4]
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype, epi_smem_size],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[cute.struct.MemRange[Int32, 0], self.buffer_align_bytes]
            epi: self.epi_get_smem_struct(epilogue_params)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        return (
            self.tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_d,
            tma_tensor_d,
            tma_atom_c,
            tma_tensor_c,
            epilogue_params,
            varlen_params,
            self.cluster_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_c_smem_layout_staged,
            tile_sched_params,
        )

    @cute.jit
    def prepare_generation_kernel_arguments(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        mGenerationTable: cute.Tensor,
        max_active_clusters: Int32,
    ):
        """`prepare_kernel_arguments` for one batch slice of the operands.

        Only the scheduler arguments differ: the problem is declared with L = 1, and Quack's
        `batch_idx_permute` reads the batch index from `mGenerationTable[0]`.
        """

        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.d_dtype = mD.element_type
        self.c_dtype = None
        self.a_layout = LayoutEnum.from_tensor(mA)
        self.b_layout = LayoutEnum.from_tensor(mB)
        self.d_layout = LayoutEnum.from_tensor(mD)
        self.c_layout = None

        epilogue_args = self.EpilogueArguments()
        varlen_args = VarlenArguments()
        self._setup_attributes(epilogue_args)

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b = self.make_tma_load_atoms_and_tensors(
            mA, mB, a_smem_layout, b_smem_layout, False
        )
        self.num_tma_load_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        self.num_tma_load_bytes += cute.size_in_bytes(self.b_dtype, b_smem_layout)

        tma_atom_d, tma_tensor_d, tma_atom_c, tma_tensor_c = self.make_tma_epilogue_atoms_and_tensors(
            mD, None, epilogue_args, False
        )
        epilogue_params = self.epi_to_underlying_arguments(epilogue_args)
        varlen_params = VarlenManager.to_underlying_arguments(varlen_args)
        self.epi_load_bytes_per_stage = self.epi_smem_bytes(
            epilogue_args,
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.epi_smem_warp_shape_mnk(),
        ).c_stage

        # L and the group size are literals, so with static operand extents the scheduler
        # parameters are compile-time constants and the batch index is read from the table.
        TileSchedulerCls = self.get_scheduler_class(varlen_m=False)
        tile_sched_args = TileSchedulerArguments(
            problem_shape_ntile_mnl=(
                cute.ceil_div(cute.size(mA, mode=[0]), self.cta_tile_shape_mnk[0]),
                cute.ceil_div(cute.size(mB, mode=[0]), self.cta_tile_shape_mnk[1]),
                1,
            ),
            raster_order=None,
            group_size=8,
            cluster_shape_mnk=self.cluster_shape_mnk,
            tile_count_semaphore=None,
            batch_idx_permute=mGenerationTable,
            persistence_mode=PersistenceMode.STATIC,
        )
        tile_sched_params = TileSchedulerCls.to_underlying_arguments(tile_sched_args)

        epi_smem_size = cute.cosize(self.epi_smem_layout_staged)

        @cute.struct
        class SharedStorage:
            ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            epi_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.epi_c_stage * 2]
            sched_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.sched_stage * 2]
            sched_data: cute.struct.MemRange[Int32, self.sched_stage * 4]
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype, epi_smem_size],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[cute.struct.MemRange[Int32, 0], self.buffer_align_bytes]
            epi: self.epi_get_smem_struct(epilogue_params)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        return (
            self.tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_d,
            tma_tensor_d,
            tma_atom_c,
            tma_tensor_c,
            epilogue_params,
            varlen_params,
            self.cluster_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_c_smem_layout_staged,
            tile_sched_params,
        )


class ResidualGemmMember(ProgramPageGemmMixin, GemmDefaultSm90):
    """Quack's default-epilogue GEMM with a C operand, over one batch slice.

    Its `prepare_generation_kernel_arguments` is `DefaultGemmMember`'s with C added: the
    epilogue loads C through its TMA load pipeline into the `sC` stages.
    """

    @cute.jit
    def prepare_generation_kernel_arguments(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        mC: cute.Tensor,
        mGenerationTable: cute.Tensor,
        max_active_clusters: Int32,
    ):
        del max_active_clusters
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.d_dtype = mD.element_type
        self.c_dtype = mC.element_type
        self.a_layout = LayoutEnum.from_tensor(mA)
        self.b_layout = LayoutEnum.from_tensor(mB)
        self.d_layout = LayoutEnum.from_tensor(mD)
        self.c_layout = LayoutEnum.from_tensor(mC)

        epilogue_args = self.EpilogueArguments()
        varlen_args = VarlenArguments()
        self._setup_attributes(epilogue_args)

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b = self.make_tma_load_atoms_and_tensors(
            mA, mB, a_smem_layout, b_smem_layout, False
        )
        self.num_tma_load_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        self.num_tma_load_bytes += cute.size_in_bytes(self.b_dtype, b_smem_layout)

        tma_atom_d, tma_tensor_d, tma_atom_c, tma_tensor_c = self.make_tma_epilogue_atoms_and_tensors(
            mD,
            mC,
            epilogue_args,
            False,
        )
        epilogue_params = self.epi_to_underlying_arguments(epilogue_args)
        varlen_params = VarlenManager.to_underlying_arguments(varlen_args)
        self.epi_load_bytes_per_stage = self.epi_smem_bytes(
            epilogue_args,
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.epi_smem_warp_shape_mnk(),
        ).c_stage
        c_smem_layout = cute.slice_(self.epi_c_smem_layout_staged, (None, None, 0))
        self.epi_load_bytes_per_stage += cute.size_in_bytes(self.c_dtype, c_smem_layout)

        tile_sched_args = TileSchedulerArguments(
            problem_shape_ntile_mnl=(
                cute.ceil_div(cute.size(mA, mode=[0]), self.cta_tile_shape_mnk[0]),
                cute.ceil_div(cute.size(mB, mode=[0]), self.cta_tile_shape_mnk[1]),
                1,
            ),
            raster_order=None,
            group_size=8,
            cluster_shape_mnk=self.cluster_shape_mnk,
            tile_count_semaphore=None,
            batch_idx_permute=mGenerationTable,
            persistence_mode=PersistenceMode.STATIC,
        )
        TileSchedulerCls = self.get_scheduler_class(varlen_m=False)
        tile_sched_params = TileSchedulerCls.to_underlying_arguments(tile_sched_args)

        epi_smem_size = cute.cosize(self.epi_smem_layout_staged)
        epi_c_smem_size = cute.cosize(self.epi_c_smem_layout_staged)

        @cute.struct
        class SharedStorage:
            ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            epi_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.epi_c_stage * 2]
            sched_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.sched_stage * 2]
            sched_data: cute.struct.MemRange[Int32, self.sched_stage * 4]
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype, epi_smem_size],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, epi_c_smem_size],
                self.buffer_align_bytes,
            ]
            epi: self.epi_get_smem_struct(epilogue_params)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        return (
            self.tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_d,
            tma_tensor_d,
            tma_atom_c,
            tma_tensor_c,
            epilogue_params,
            varlen_params,
            self.cluster_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_c_smem_layout_staged,
            tile_sched_params,
        )


class GatedGemmMember(ProgramPageGemmMixin, GemmGatedSm90):
    """Quack's gated GEMM as a program member; `mlp_projections` adds its preparers."""
