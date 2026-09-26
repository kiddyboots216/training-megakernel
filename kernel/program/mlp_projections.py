"""The MLP block's projection members: gate/up and down forward, and down dX.

- `GateUpForwardMember` is Quack's gated GEMM with a SwiGLU epilogue.  It runs as two
  instances: one stores only the SwiGLU output (`prepare_activation_only_arguments`), the other
  also stores the gate/up preactivation (`prepare_dual_output_arguments`).
- `DownForwardMember` adds residual_mid to the BF16-rounded projection and stores the sum to
  the layer slot and to the activation chain, as the next layer's input.
- `DownDxSwiGLUBackwardMember` computes the down projection's dX and, from it and the saved
  preactivation, the gate and up gradients, which it stores instead of dX.

The gate/up weight holds all gate rows, then all up rows.  The gate/up preparers view it
interleaved, so adjacent accumulator columns are a (gate, up) pair and the preactivation D
stores is [g0, u0, g1, u1, ...]; the down dX member reads it back in that order.

Adapted from `GemmSm90.__call__` in quack/gemm_sm90.py (Copyright (c) 2025-2026, QuACK team)
and `GemmGatedMixin.epi_visit_subtile`, `GemmActMixin.epi_to_underlying_arguments`,
`GemmActMixin.epi_make_aux_out_copy_atom_r2s`, `GemmActMixin.epi_make_aux_out_tiled_copy_r2s`,
`GemmActMixin.epi_setup_aux_out` and `GemmActMixin.epi_convert_aux_out` in quack/gemm_act.py
(Copyright (c) 2025, Wentao Guo, Tri Dao), from quack-kernels 0.6.0.  Quack is distributed
under the Apache License 2.0 (see THIRD_PARTY_NOTICES.md).

- `GateUpForwardMember.prepare_activation_only_arguments` and `prepare_dual_output_arguments`
  are `__call__` up to the launch (see `gemm_members`), with B viewed interleaved and a
  one-batch problem whose batch index comes from a device table; the activation-only form has
  no D.
- `GateUpForwardMember.epi_visit_subtile` is `GemmGatedMixin.epi_visit_subtile` with gate and
  up rounded to BF16 first and the SwiGLU computed in place of `act_fn`.
- `DownForwardMember.epi_to_underlying_arguments` is `GemmActMixin`'s with the layer-index
  table in place of `act_fn` and without the concat-layout handling;
  `DownDxSwiGLUBackwardMember`'s sets the same auxiliary-output attributes from `mDGateHalf`
  and then runs the default conversion.
- `DownDxSwiGLUBackwardMember._aux_out_tiled_copy_r2s` is `GemmActMixin`'s SM90 copy atom and
  tiled copy for an output given by name; its `epi_setup_aux_out` is `GemmActMixin`'s for two
  outputs, and its `epi_convert_aux_out` keeps only the round-to-nearest conversion.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
import quack.copy_utils as copy_utils
from cutlass import BFloat16, Float32, Int32
from cutlass.utils import LayoutEnum
from quack import layout_utils
from quack.activation import gate_fn_map
from quack.cute_dsl_utils import mlir_namedtuple
from quack.epi_ops import EpiOp, TileStore
from quack.gemm_act import GemmActMixin
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.rounding import RoundingMode
from quack.tile_scheduler import PersistenceMode, TileSchedulerArguments
from quack.varlen_utils import VarlenArguments, VarlenManager

import model
from gemm_members import DefaultGemmMember
from gemm_members import GatedGemmMember
from projection_members import (
    PROJECTION_TILE_M,
    PROJECTION_TILE_N,
    add_residual_after_bf16_rounding,
    build_projection_member,
    setup_table_routed_aux_store,
)
from quack_gemm_bodies import down_dx_bf16_epilogue
from gemm_members import ResidualGemmMember


class GateUpForwardMember(GatedGemmMember):
    """Gate/up forward: Quack's gated GEMM with a SwiGLU epilogue on BF16-rounded gate and up.

    One instance stores only the SwiGLU output (`prepare_activation_only_arguments`); the
    other also stores the interleaved gate/up preactivation (`prepare_dual_output_arguments`).
    """

    @cute.jit
    def epi_visit_subtile(
        self,
        params,
        epi_loop_tensors,
        tRS_rD: cute.Tensor,
        tRS_rC=None,
    ):
        # Quack applies the activation to the FP32 accumulator.  Here gate and up are rounded
        # to BF16 first, as if the projection were stored in BF16 and the SwiGLU run on the
        # stored values; the down dX member reads that stored preactivation back.  The SwiGLU
        # is gate * sigmoid(gate) * up with sigmoid = rcp_approx(1 + exp(-gate)); `act_fn` is
        # not called.
        GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
        post_layout = cute.recast_layout(2, 1, tRS_rD.layout)
        post = cute.make_rmem_tensor(post_layout.shape, Float32)
        pair = cute.flat_divide(tRS_rD, cute.make_layout(2))
        gate_values = pair[0, ...]
        up_values = pair[1, ...]
        for index in cutlass.range(cute.size(post), unroll_full=True):
            gate = gate_values[index].to(BFloat16).to(Float32)
            up = up_values[index].to(BFloat16).to(Float32)
            sigmoid = cute.arch.rcp_approx(Float32(1.0) + cute.math.exp(Float32(0.0) - gate, fastmath=True))
            post[index] = gate * sigmoid * up
        return (post,)

    @cute.jit
    def prepare_activation_only_arguments(
        self,
        mA: cute.Tensor,
        mBPhysical: cute.Tensor,
        mAuxOut: cute.Tensor,
        mGenerationTable: cute.Tensor,
        max_active_clusters: Int32,
    ):
        """The gated arguments without D: only the SwiGLU output `mAuxOut` is stored.

        `mBPhysical` is the gate/up weight as stored; the batch index comes from
        `mGenerationTable[0]`.
        """

        # View the [all gate | all up] weight rows interleaved, so adjacent accumulator
        # columns are a (gate, up) pair for the gated epilogue.
        mB = layout_utils.concat_to_interleave(
            mBPhysical,
            1 - mBPhysical.leading_dim,
        )

        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.d_dtype = None
        self.c_dtype = None
        self.a_layout = LayoutEnum.from_tensor(mA)
        self.b_layout = LayoutEnum.from_tensor(mB)
        self.d_layout = None
        self.c_layout = None

        # Quack's gated arguments name an activation; epi_visit_subtile computes the SwiGLU
        # itself and does not call it.
        epilogue_args = self.EpilogueArguments(
            mAuxOut,
            gate_fn_map["swiglu"],
        )
        varlen_args = VarlenArguments()
        self._setup_attributes(epilogue_args)

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b = self.make_tma_load_atoms_and_tensors(
            mA,
            mB,
            a_smem_layout,
            b_smem_layout,
            False,
        )
        self.num_tma_load_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout) + cute.size_in_bytes(
            self.b_dtype, b_smem_layout
        )

        # No D: the body's `has_D` is false, so the epilogue skips the D store and keeps only
        # the auxiliary output.
        tma_atom_d, tma_tensor_d, tma_atom_c, tma_tensor_c = self.make_tma_epilogue_atoms_and_tensors(
            None,
            None,
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

        scheduler_cls = self.get_scheduler_class(varlen_m=False)
        scheduler_args = TileSchedulerArguments(
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
        tile_sched_params = scheduler_cls.to_underlying_arguments(scheduler_args)

        # sD is empty because there is no D; the auxiliary output's shared memory comes from
        # `epi_get_smem_struct`.
        @cute.struct
        class SharedStorage:
            ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            epi_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.epi_c_stage * 2]
            sched_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.sched_stage * 2]
            sched_data: cute.struct.MemRange[Int32, self.sched_stage * 4]
            sD: cute.struct.Align[cute.struct.MemRange[Int32, 0], self.buffer_align_bytes]
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
    def prepare_dual_output_arguments(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        mAuxOut: cute.Tensor,
        mGenerationTable: cute.Tensor,
        max_active_clusters: Int32,
    ):
        """The gated arguments with both outputs: the preactivation D and the SwiGLU `mAuxOut`.

        D is stored interleaved.  `mB` is the gate/up weight as stored, viewed interleaved
        here; the batch index comes from `mGenerationTable[0]`.
        """
        mB = layout_utils.concat_to_interleave(mB, 1 - mB.leading_dim)
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.d_dtype = mD.element_type
        self.c_dtype = None
        self.a_layout = LayoutEnum.from_tensor(mA)
        self.b_layout = LayoutEnum.from_tensor(mB)
        self.d_layout = LayoutEnum.from_tensor(mD)
        self.c_layout = None
        epilogue_args = self.EpilogueArguments(mAuxOut, gate_fn_map["swiglu"])
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
            epilogue_args, self.cta_tile_shape_mnk, self.epi_tile, self.epi_smem_warp_shape_mnk()
        ).c_stage
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
                cute.struct.MemRange[self.d_dtype, epi_smem_size], self.buffer_align_bytes
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



class DownForwardMember(ResidualGemmMember):
    """Down forward: the BF16-rounded projection plus residual_mid, stored twice.

    residual_mid is the C operand, read from the layer slot.  D stores the sum into the layer
    slot, and an auxiliary TileStore stores it into the activation chain at the layer index,
    where it is the next layer's input.
    """

    _epi_ops = (
        *ResidualGemmMember._epi_ops,
        TileStore("mAuxOut"),
    )
    _extra_param_fields = (("mLogicalGenerationTable", object, None),)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mAuxOut: Optional[cute.Tensor] = None
        mLogicalGenerationTable: Optional[cute.Tensor] = None
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        del loc, ip
        self.rounding_mode = args.rounding_mode
        self.aux_out_dtype = args.mAuxOut.element_type
        self.aux_out_layout = cutlass.utils.LayoutEnum.from_tensor(
            args.mAuxOut
        )
        self.cta_tile_shape_aux_out_mn = self.cta_tile_shape_mnk[:2]
        values = self._epi_ops_to_params_dict(args)
        values["mLogicalGenerationTable"] = args.mLogicalGenerationTable
        return self.EpilogueParams(**values)

    # Quack's auxiliary-output methods, taken from GemmActMixin: the member derives from the
    # default-epilogue GEMM, which has no auxiliary output.
    epi_make_aux_out_copy_atom_r2s = GemmActMixin.epi_make_aux_out_copy_atom_r2s
    epi_make_aux_out_tiled_copy_r2s = GemmActMixin.epi_make_aux_out_tiled_copy_r2s
    epi_convert_aux_out = GemmActMixin.epi_convert_aux_out

    def epi_setup_aux_out(
        self,
        params,
        epi_smem_tensors,
        tiled_copy_r2s,
        tiled_copy_t2r,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        return setup_table_routed_aux_store(
            self,
            params,
            epi_smem_tensors,
            tiled_copy_r2s,
            tiled_copy_t2r,
            tile_coord_mnkl,
            varlen_manager,
            tidx,
            params.mLogicalGenerationTable,
        )

    @cute.jit
    def prepare_generation_kernel_arguments(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mPhysicalD: cute.Tensor,
        mPhysicalResidual: cute.Tensor,
        mLogicalOutput: cute.Tensor,
        mPhysicalGenerationTable: cute.Tensor,
        max_active_clusters: Int32,
        mLogicalGenerationTable: cute.Tensor,
    ):
        """`ResidualGemmMember.prepare_generation_kernel_arguments` plus the auxiliary store.

        A, B, C and D follow the layer slot (`mPhysicalGenerationTable`); the activation-chain
        output `mLogicalOutput` follows the layer index (`mLogicalGenerationTable`).
        """

        # The base preparer builds its epilogue arguments with `self.EpilogueArguments()`;
        # an instance attribute supplies this member's for the duration of the call.
        arguments_cls = type(self).EpilogueArguments
        self.EpilogueArguments = lambda: arguments_cls(
            mAuxOut=mLogicalOutput,
            mLogicalGenerationTable=mLogicalGenerationTable,
        )
        try:
            return ResidualGemmMember.prepare_generation_kernel_arguments(
                self,
                mA,
                mB,
                mPhysicalD,
                mPhysicalResidual,
                mPhysicalGenerationTable,
                max_active_clusters,
            )
        finally:
            del self.EpilogueArguments

    @cute.jit
    def epi_visit_subtile(
        self,
        params,
        epi_loop_tensors,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ):
        del params, epi_loop_tensors
        assert tRS_rC is not None
        add_residual_after_bf16_rounding(tRS_rD, tRS_rC)
        # D and the auxiliary output both round this sum to BF16 to nearest, so they store the
        # same values.
        return (tRS_rD,)


class SavedPreactivationOp(EpiOp):
    """An epilogue op that gives the visit what it needs to read the saved preactivation.

    Per subtile: each fragment's coordinates in the tile, the tile's M and N indices, its batch
    index (the layer slot) and the flat preactivation tensor.
    """

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        return {self.name: getattr(args, self.name)}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        mcD = cute.make_identity_tensor((ctx.tile_M, ctx.tile_N))
        tRS_cD = ctx.partition_for_epilogue_fn(mcD)
        return (
            tRS_cD,
            ctx.tile_coord_mnkl[0],
            ctx.tile_coord_mnkl[1],
            ctx.tile_coord_mnkl[3],
            param,
        )

    def begin_loop(self, gemm, state, epi_coord):
        tRS_cD, tile_m, tile_n, generation, gate_up = state
        return (
            tRS_cD[(None, None, None, *epi_coord)],
            tile_m,
            tile_n,
            generation,
            gate_up,
        )


class DownDxSwiGLUBackwardMember(DefaultGemmMember):
    """Down dX with the SwiGLU backward in the epilogue; it stores the gate and up gradients.

    The epilogue (`quack_gemm_bodies.down_dx_bf16_epilogue`) rounds each dX subtile to BF16.
    The visit reads gate and up from the saved preactivation in global memory and returns the
    two gradients, which two TileStores write into the gate and up halves of the gate/up
    gradient.  dX itself is not stored.
    """

    _epi_ops = (
        *DefaultGemmMember._epi_ops,
        SavedPreactivationOp("mGateUpFlat"),
        TileStore("mDGateHalf"),
        TileStore("mDUpHalf"),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mGateUpFlat: Optional[cute.Tensor] = None
        mDGateHalf: Optional[cute.Tensor] = None
        mDUpHalf: Optional[cute.Tensor] = None
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    # Quack's epilogue, with each dX subtile rounded to BF16 through shared memory first.
    epilogue = down_dx_bf16_epilogue

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        # TileStore.smem_struct_field and the auxiliary register-to-shared copy read these
        # three attributes, which GemmActMixin sets from mAuxOut; both outputs share their
        # dtype and layout.
        self.aux_out_dtype = args.mDGateHalf.element_type
        self.aux_out_layout = cutlass.utils.LayoutEnum.from_tensor(args.mDGateHalf)
        self.cta_tile_shape_aux_out_mn = self.cta_tile_shape_mnk[:2]
        return DefaultGemmMember.epi_to_underlying_arguments(
            self, args, loc=loc, ip=ip
        )

    # GemmActMixin's auxiliary-output methods, for the two outputs mDGateHalf and mDUpHalf.

    def _aux_out_tiled_copy_r2s(self, params, tiled_copy_r2s, name):
        """The register-to-shared tiled copy of the auxiliary output `name`."""

        copy_atom = copy_utils.get_smem_store_atom(
            self.aux_out_dtype,
            transpose=self.aux_out_layout != cutlass.utils.LayoutEnum.ROW_MAJOR,
            major_mode_size=cute.size(getattr(params, f"epi_tile_{name}"), mode=[1])
            // self.atom_layout_mnk[1],
        )
        return cute.make_tiled_copy_S(copy_atom, tiled_copy_r2s)

    def epi_setup_aux_out(
        self,
        params,
        epi_smem_tensors,
        tiled_copy_r2s,
        tiled_copy_t2r,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        ctxs = []
        batch_idx = tile_coord_mnkl[3]
        for name in ("mDGateHalf", "mDUpHalf"):
            sAuxOut = epi_smem_tensors[name]
            tiled_copy_aux = self._aux_out_tiled_copy_r2s(params, tiled_copy_r2s, name)
            tRS_sAuxOut = tiled_copy_aux.get_slice(tidx).partition_D(sAuxOut)
            copy_aux_out, _, _ = self.epilog_gmem_copy_and_partition(
                getattr(params, f"tma_atom_{name}"),
                varlen_manager.offset_batch_epi(getattr(params, name), batch_idx),
                self.cta_tile_shape_aux_out_mn,
                getattr(params, f"epi_tile_{name}"),
                sAuxOut,
                tile_coord_mnkl,
            )
            ctxs.append((tiled_copy_aux, tRS_sAuxOut, copy_aux_out))
        return tuple(ctxs)

    @cute.jit
    def epi_convert_aux_out(
        self,
        output_idx: cutlass.Constexpr[int],
        tRS_rAuxOut,
        sr_seed,
        tidx,
        tile_coord_mnkl,
        num_prev_subtiles,
        epi_idx,
    ):
        """Round the FP32 gradient fragment to BF16 to nearest before the shared-memory copy."""

        tRS_rAuxOut_out = cute.make_rmem_tensor_like(tRS_rAuxOut, self.aux_out_dtype)
        tRS_rAuxOut_out.store(tRS_rAuxOut.load().to(self.aux_out_dtype))
        return tRS_rAuxOut_out

    @cute.jit
    def prepare_generation_kernel_arguments(
        self,
        mA,
        mB,
        mD,
        mGenerationTable,
        max_active_clusters,
        mGateUpFlat=None,
        mDGateHalf=None,
        mDUpHalf=None,
    ):
        """`DefaultGemmMember.prepare_generation_kernel_arguments` without the D store.

        The epilogue arguments are the saved preactivation `mGateUpFlat` and the two gradient
        outputs, `mDGateHalf` and `mDUpHalf`.
        """

        # The base preparer builds its epilogue arguments with `self.EpilogueArguments()`;
        # an instance attribute supplies this member's for the duration of the call.
        arguments_cls = type(self).EpilogueArguments
        self.EpilogueArguments = lambda: arguments_cls(
            mGateUpFlat=mGateUpFlat,
            mDGateHalf=mDGateHalf,
            mDUpHalf=mDUpHalf,
        )
        try:
            prepared = DefaultGemmMember.prepare_generation_kernel_arguments(
                self, mA, mB, mD, mGenerationTable, max_active_clusters
            )
        finally:
            del self.EpilogueArguments
        # Elements 5 and 6 are D's TMA atom and tensor; without them the body skips the D
        # store.
        return (*prepared[:5], None, None, *prepared[7:])

    @cute.jit
    def epi_visit_subtile(
        self,
        params,
        epi_loop_tensors,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ):
        # Quack's default visit returns no auxiliary outputs, so the result holds just the two
        # gradients.
        outs = DefaultGemmMember.epi_visit_subtile(
            self, params, epi_loop_tensors, tRS_rD, tRS_rC
        )
        carry_state = epi_loop_tensors.get("mGateUpFlat")
        tRS_rDGate = cute.make_rmem_tensor(tRS_rD.layout.shape, Float32)
        tRS_rDUp = cute.make_rmem_tensor(tRS_rD.layout.shape, Float32)
        tRS_cD_sub, tile_m, tile_n, generation, gate_up_flat = carry_state
        # `generation` is the tile's batch index: the layer slot of the saved preactivation.
        generation_base = (
            generation * Int32(model.SEQUENCE * model.GATE_UP)
        )
        for index in cutlass.range(cute.size(tRS_rD), unroll_full=True):
            coord = tRS_cD_sub[index]
            row = tile_m * Int32(PROJECTION_TILE_M) + Int32(coord[0])
            column = tile_n * Int32(PROJECTION_TILE_N) + Int32(coord[1])
            wide_base = generation_base + row * Int32(model.GATE_UP)

            # The SwiGLU backward takes dX as BF16, as if dX were stored; after the epilogue's
            # round trip tRS_rD already holds BF16 values.
            dy = tRS_rD[index].to(BFloat16).to(Float32)
            # The gate/up forward stores the preactivation interleaved, [g0, u0, g1, u1, ...];
            # the gradients go out in the concat layout [dgate | dup].
            pair_column = Int32(2) * column
            gate = gate_up_flat[wide_base + pair_column].to(Float32)
            up = gate_up_flat[wide_base + pair_column + Int32(1)].to(
                Float32
            )
            sigmoid = cute.arch.rcp_approx(
                Float32(1.0)
                + cute.math.exp(Float32(0.0) - gate, fastmath=True)
            )
            silu = gate * sigmoid
            silu_derivative = sigmoid * (
                Float32(1.0)
                + gate * (Float32(1.0) - sigmoid)
            )
            tRS_rDGate[index] = dy * up * silu_derivative
            tRS_rDUp[index] = dy * silu

        # epi_convert_aux_out rounds each gradient to BF16, and the epilogue stores it through
        # shared memory and TMA into its half of the gate/up gradient.
        return (*outs, tRS_rDGate, tRS_rDUp)


def make_gate_up_forward_member() -> GateUpForwardMember:
    return build_projection_member(GateUpForwardMember)


def make_down_forward_member() -> DownForwardMember:
    return build_projection_member(DownForwardMember)


def make_down_dx_member() -> DownDxSwiGLUBackwardMember:
    return build_projection_member(DownDxSwiGLUBackwardMember)
