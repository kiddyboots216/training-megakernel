"""The attention block's forward projection members.

- `QkvForwardMember` stores the packed QKV projection and also writes the V heads straight into
  the attention's V tensor.
- `AttentionOutputForwardMember` computes residual_mid: the BF16-rounded attention-output
  projection plus the layer input, which it loads from the activation chain.
- `AttentionOutputCheckpointMember` computes the same and also stores it into the residual_mid
  checkpoint; `run_attention_output_checkpoint_gemm` runs it in the top layers of the initial
  forward.

The members read the generation table the step writes for each layer visit: word 0 is the
layer slot, word 1 the layer index and word 2 the residual_mid checkpoint index.  A, B and D
follow the layer slot, through the tile scheduler's batch index.  The activation chain has one
entry per layer and the checkpoint one per checkpointed layer, so the epilogue operands stored
there are routed by words 1 and 2.

Adapted from `GemmSm90.__call__` in quack/gemm_sm90.py (Copyright (c) 2025-2026, QuACK team),
`TileLoad.load_g2s_copy_fn` in quack/epi_ops.py (Copyright (c) 2025, Tri Dao),
`GemmDefaultEpiMixin.epi_to_underlying_arguments` in quack/gemm_default_epi.py (Copyright (c)
2025, Wentao Guo, Tri Dao) and `GemmActMixin.epi_to_underlying_arguments` in quack/gemm_act.py
(Copyright (c) 2025, Wentao Guo, Tri Dao), from quack-kernels 0.6.0.  Quack is distributed
under the Apache License 2.0 (see THIRD_PARTY_NOTICES.md).

- `QkvForwardMember.prepare_value_publish_arguments` is `__call__` up to the launch, as in
  `gemm_members.DefaultGemmMember.prepare_generation_kernel_arguments`, with the V tensor as an
  epilogue argument.
- `LogicalLayerTileLoad.load_g2s_copy_fn` takes the batch index from the layer-index table
  instead of the tile coordinate.
- `AttentionOutputForwardMember.epi_to_underlying_arguments` (from `GemmDefaultEpiMixin`) and
  `AttentionOutputCheckpointMember.epi_to_underlying_arguments` (from `GemmActMixin`, with the
  table fields in place of `act_fn`) add the generation-table fields to the epilogue
  parameters and leave out the concat-layout handling.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32
from cutlass.utils import LayoutEnum
from quack import copy_utils
from quack.cute_dsl_utils import mlir_namedtuple
from quack.epi_ops import EpiOp, TileLoad, TileStore
from quack.gemm_act import GemmActMixin
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.rounding import RoundingMode
from quack.tile_scheduler import PersistenceMode, TileSchedulerArguments
from quack.varlen_utils import VarlenArguments, VarlenManager

from anchors import anchor_scheduler_params
from tile_schedulers import EpochDynamicTileScheduler
from decoder_gemm_runners import finish_decoder_gemm
from gemm_role_rotation import pop_role_rotation, push_role_rotation
from tile_schedulers import dynamic_scheduler_params
from gemm_members import DefaultGemmMember
from model import HEAD_DIM, KV_HEADS, QUERY_HEADS
from projection_members import (
    PROJECTION_TILE_M,
    PROJECTION_TILE_N,
    add_residual_after_bf16_rounding,
    build_projection_member,
    setup_table_routed_aux_store,
)
from quack_gemm_bodies import gemm_body

# The QKV member's N tiles are whole heads, in the packed order [q | k | v].
FIRST_VALUE_TILE_N = QUERY_HEADS + KV_HEADS

# The direct V stores address whole heads: one head per N tile, with Qwen3-8B's head counts.
if (
    HEAD_DIM != 128
    or PROJECTION_TILE_N != HEAD_DIM
    or QUERY_HEADS != 32
    or KV_HEADS != 8
):
    raise AssertionError("the QKV member stores V from whole-head N tiles")


class ValueOutputOp(EpiOp):
    """An epilogue op that gives the visit what the direct V stores need; it stores nothing.

    Per subtile: each D fragment's coordinates in the tile, the V tensor's slice at the tile's
    batch index (the layer slot) and the tile coordinate.
    """

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        return {self.name: getattr(args, self.name)}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        coords = ctx.partition_for_epilogue_fn(
            cute.make_identity_tensor((ctx.tile_M, ctx.tile_N))
        )
        return (coords, param[None, None, ctx.batch_idx], ctx.tile_coord_mnkl)

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        coords, output, tile_coord = state
        return (
            coords[None, None, None, epi_coord[0], epi_coord[1]],
            output,
            tile_coord,
        )


class QkvForwardMember(DefaultGemmMember):
    """QKV forward: D is the packed QKV projection.

    The visit of each V-head tile also stores its values, rounded to BF16, from registers
    straight into the attention's V tensor, without shared memory or TMA.
    """

    _epi_ops = GemmDefaultEpiMixin._epi_ops + (ValueOutputOp("mVOut"),)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mVOut: Optional[cute.Tensor] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        # Quack's default visit is not applied: the member takes no alpha, beta, C or
        # broadcast vector.
        state = epi_loop_tensors.get("mVOut")
        coords, output, tile_coord = state
        n_tile = tile_coord[1]
        if n_tile >= FIRST_VALUE_TILE_N:
            for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
                coord = coords[i]
                row = tile_coord[0] * PROJECTION_TILE_M + coord[0]
                column = (n_tile - FIRST_VALUE_TILE_N) * PROJECTION_TILE_N + coord[1]
                output[row, column] = tRS_rD[i].to(BFloat16)
        return ()

    @cute.jit
    def prepare_value_publish_arguments(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        mVOut: cute.Tensor,
        mGenerationTable: cute.Tensor,
        max_active_clusters: Int32,
    ):
        """`DefaultGemmMember.prepare_generation_kernel_arguments` with `mVOut` as well.

        `mVOut` is the attention's V tensor, one slice per layer slot.
        """

        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.d_dtype = mD.element_type
        self.c_dtype = None
        self.a_layout = LayoutEnum.from_tensor(mA)
        self.b_layout = LayoutEnum.from_tensor(mB)
        self.d_layout = LayoutEnum.from_tensor(mD)
        self.c_layout = None

        epilogue_args = self.EpilogueArguments(mVOut=mVOut)
        varlen_args = VarlenArguments()
        self._setup_attributes(epilogue_args)

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b = (
            self.make_tma_load_atoms_and_tensors(
                mA, mB, a_smem_layout, b_smem_layout, False
            )
        )
        self.num_tma_load_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        tma_atom_d, tma_tensor_d, tma_atom_c, tma_tensor_c = (
            self.make_tma_epilogue_atoms_and_tensors(
                mD, None, epilogue_args, False
            )
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
        epi_smem_size = cute.cosize(self.epi_smem_layout_staged)

        @cute.struct
        class SharedStorage:
            ab_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            epi_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.epi_c_stage * 2
            ]
            sched_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.sched_stage * 2
            ]
            sched_data: cute.struct.MemRange[Int32, self.sched_stage * 4]
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype, epi_smem_size],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[Int32, 0], self.buffer_align_bytes
            ]
            epi: self.epi_get_smem_struct(epilogue_params)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
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



class LogicalLayerTileLoad(TileLoad):
    """A TileLoad at the layer index, `params.mLogicalGenerationTable[0]`.

    A, B and D live in layer slots, so the tile scheduler's batch index is the slot.  The
    activation chain has one entry per layer, so an operand loaded from it needs the layer
    index instead.  Quack's C operand always uses the scheduler's batch index, which is why
    the layer input is a TileLoad rather than C.
    """

    def load_g2s_copy_fn(
        self,
        gemm,
        params,
        smem_tensor,
        tile_coord_mnkl,
        varlen_manager,
        epi_pipeline,
    ):
        tensor = getattr(params, self.name, None)
        logical_generation = params.mLogicalGenerationTable[0]
        copy_tile_fn, _, _ = gemm.epilog_gmem_copy_and_partition(
            getattr(params, self._tma_atom_key()),
            varlen_manager.offset_batch_epi(tensor, logical_generation),
            gemm.cta_tile_shape_mnk[:2],
            getattr(params, self._epi_tile_key()),
            smem_tensor,
            tile_coord_mnkl,
        )
        copy_tile = copy_utils.tma_producer_copy_fn(
            copy_tile_fn, epi_pipeline
        )
        return copy_tile


class AttentionOutputForwardMember(DefaultGemmMember):
    """Attention-output forward: the BF16-rounded projection plus the layer input.

    D is residual_mid in the layer slot; the layer input is loaded from the activation chain at
    the layer index.
    """

    _epi_ops = (
        *DefaultGemmMember._epi_ops,
        LogicalLayerTileLoad("mLogicalResidual"),
    )
    _extra_param_fields = (("mLogicalGenerationTable", object, None),)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mLogicalResidual: Optional[cute.Tensor] = None
        mLogicalGenerationTable: Optional[cute.Tensor] = None
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        values = self._epi_ops_to_params_dict(args)
        values["mLogicalGenerationTable"] = args.mLogicalGenerationTable
        return self.EpilogueParams(**values)

    @cute.jit
    def prepare_generation_kernel_arguments(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        mLogicalResidual: cute.Tensor,
        mPhysicalGenerationTable: cute.Tensor,
        max_active_clusters: Int32,
        mLogicalGenerationTable: cute.Tensor,
    ):
        """`DefaultGemmMember.prepare_generation_kernel_arguments` plus the layer input.

        A, B and D follow the layer slot (`mPhysicalGenerationTable`); the layer input
        `mLogicalResidual` follows the layer index (`mLogicalGenerationTable`).
        """

        # The base preparer builds its epilogue arguments with `self.EpilogueArguments()`;
        # an instance attribute supplies this member's for the duration of the call.
        arguments_cls = type(self).EpilogueArguments
        self.EpilogueArguments = lambda: arguments_cls(
            mLogicalResidual=mLogicalResidual,
            mLogicalGenerationTable=mLogicalGenerationTable,
        )
        try:
            return DefaultGemmMember.prepare_generation_kernel_arguments(
                self,
                mA,
                mB,
                mD,
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
        del params, tRS_rC
        activation_chain = epi_loop_tensors["mLogicalResidual"]
        assert activation_chain is not None
        add_residual_after_bf16_rounding(tRS_rD, activation_chain)
        # No auxiliary outputs, so the epilogue never indexes the result.
        return None


class AttentionOutputCheckpointMember(DefaultGemmMember):
    """Attention-output forward that also stores its result into the residual_mid checkpoint.

    The checkpoint store is an auxiliary TileStore at the checkpoint index.
    """

    _epi_ops = (
        *DefaultGemmMember._epi_ops,
        LogicalLayerTileLoad("mLogicalResidual"),
        TileStore("mAuxOut"),
    )
    _extra_param_fields = (
        ("mLogicalGenerationTable", object, None),
        ("mCheckpointGenerationTable", object, None),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mLogicalResidual: Optional[cute.Tensor] = None
        mLogicalGenerationTable: Optional[cute.Tensor] = None
        mAuxOut: Optional[cute.Tensor] = None
        mCheckpointGenerationTable: Optional[cute.Tensor] = None
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
        self.aux_out_layout = cutlass.utils.LayoutEnum.from_tensor(args.mAuxOut)
        self.cta_tile_shape_aux_out_mn = self.cta_tile_shape_mnk[:2]
        values = self._epi_ops_to_params_dict(args)
        values["mLogicalGenerationTable"] = args.mLogicalGenerationTable
        values["mCheckpointGenerationTable"] = args.mCheckpointGenerationTable
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
            params.mCheckpointGenerationTable,
        )

    @cute.jit
    def prepare_checkpoint_arguments(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mPhysicalD: cute.Tensor,
        mLogicalResidual: cute.Tensor,
        mCheckpointOut: cute.Tensor,
        mPhysicalGenerationTable: cute.Tensor,
        max_active_clusters: Int32,
        mLogicalGenerationTable: cute.Tensor,
        mCheckpointGenerationTable: cute.Tensor,
    ):
        """`AttentionOutputForwardMember.prepare_generation_kernel_arguments` plus the checkpoint.

        `mCheckpointOut` is stored at the checkpoint index (`mCheckpointGenerationTable`).
        """

        # The epilogue arguments reach the base preparer as in AttentionOutputForwardMember.
        arguments_cls = type(self).EpilogueArguments
        self.EpilogueArguments = lambda: arguments_cls(
            mLogicalResidual=mLogicalResidual,
            mLogicalGenerationTable=mLogicalGenerationTable,
            mAuxOut=mCheckpointOut,
            mCheckpointGenerationTable=mCheckpointGenerationTable,
        )
        try:
            return DefaultGemmMember.prepare_generation_kernel_arguments(
                self,
                mA,
                mB,
                mPhysicalD,
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
        del params, tRS_rC
        residual = epi_loop_tensors["mLogicalResidual"]
        assert residual is not None
        add_residual_after_bf16_rounding(tRS_rD, residual)
        # The checkpoint receives the same sum as D.
        return (tRS_rD,)


def make_qkv_forward_member() -> QkvForwardMember:
    return build_projection_member(QkvForwardMember)


def make_attention_output_forward_member() -> AttentionOutputForwardMember:
    return build_projection_member(AttentionOutputForwardMember)


def make_attention_output_checkpoint_member() -> AttentionOutputCheckpointMember:
    return build_projection_member(AttentionOutputCheckpointMember)


@cute.jit
def run_attention_output_checkpoint_gemm(
    self,
    role: cutlass.Constexpr[int],
    phase_counter: cute.Tensor,
    scheduler_state: cute.Tensor,
    *gemm_args,
):
    """Run the checkpoint member as `decoder_gemm_runners.run_decoder_gemm` runs a family's.

    The member is not one of the family members.  It takes the attention-output forward
    family's queue (`scheduler_state`), whose tile count it shares.
    """

    self.reanchor_smem_page()
    member = self.o_fwd_checkpoint_member
    sched = anchor_scheduler_params(gemm_args[16])
    sched = dynamic_scheduler_params(sched, scheduler_state)
    push_role_rotation(role)
    gemm_body(
        member,
        *gemm_args[:16],
        sched,
        EpochDynamicTileScheduler,
    )
    pop_role_rotation()
    finish_decoder_gemm(self, phase_counter)
