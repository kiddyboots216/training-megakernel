"""The decoder projection members' shared tile settings, plain member and epilogue helpers.

`build_projection_member` builds a member class with the projection tile settings, and
`make_projection_member` builds the plain member (Quack's default epilogue) that every dW
projection and every dX projection but down's runs.  `add_residual_after_bf16_rounding` and
`setup_table_routed_aux_store` are epilogue steps of the attention-output and down members in
`attention_projections` and `mlp_projections`.

`setup_table_routed_aux_store` is adapted from `GemmActMixin.epi_setup_aux_out` in
quack/gemm_act.py (Copyright (c) 2025, Wentao Guo, Tri Dao), from quack-kernels 0.6.0.  Quack
is distributed under the Apache License 2.0 (see THIRD_PARTY_NOTICES.md).  It takes the member
as an argument, reads the store's batch index from a device table instead of the tile
coordinate, and has no case without `mAuxOut`.
"""

from __future__ import annotations

from cutlass import BFloat16, Float32
from gemm_members import DefaultGemmMember

# Every decoder projection member runs 256x128 output tiles, without ping-pong, on a
# 1x1x1 cluster.
PROJECTION_TILE_M = 256
PROJECTION_TILE_N = 128
PROJECTION_CLUSTER_SHAPE_MNK = (1, 1, 1)


def build_projection_member(member_class):
    """An instance of `member_class` with the projection members' tile settings."""

    return member_class(
        Float32,
        BFloat16,
        (PROJECTION_TILE_M, PROJECTION_TILE_N),
        PROJECTION_CLUSTER_SHAPE_MNK,
        pingpong=False,
        is_persistent=True,
        gather_A=False,
        concat_layout=None,
    )


def make_projection_member() -> DefaultGemmMember:
    """The plain member: Quack's default epilogue with the projection tile settings.

    It runs every dW projection and the qkv, attention-output and gate/up dX projections.
    """

    return build_projection_member(DefaultGemmMember)


def add_residual_after_bf16_rounding(tRS_rD, residual) -> None:
    """Round the accumulator subtile to BF16, add the BF16 residual in FP32, store the sum."""

    rounded = tRS_rD.load().to(BFloat16).to(Float32)
    rounded += residual.load().to(Float32)
    tRS_rD.store(rounded)


def setup_table_routed_aux_store(
    gemm,
    params,
    epi_smem_tensors,
    tiled_copy_r2s,
    tiled_copy_t2r,
    tile_coord_mnkl,
    varlen_manager,
    tidx,
    batch_table,
):
    """Quack's `mAuxOut` store setup for `gemm`, with the output batch read from `batch_table[0]`.

    Returns the one auxiliary-output context the epilogue expects: the register-to-shared copy,
    its shared-memory partition and the TMA store into `mAuxOut`.
    """

    s_output = epi_smem_tensors["mAuxOut"]
    tiled_copy_output_r2s = gemm.epi_make_aux_out_tiled_copy_r2s(
        params, tiled_copy_r2s, tiled_copy_t2r
    )
    tRS_s_output = tiled_copy_output_r2s.get_slice(tidx).partition_D(s_output)
    batch = batch_table[0]
    copy_output, _, _ = gemm.epilog_gmem_copy_and_partition(
        params.tma_atom_mAuxOut,
        varlen_manager.offset_batch_epi(params.mAuxOut, batch),
        gemm.cta_tile_shape_aux_out_mn,
        params.epi_tile_mAuxOut,
        s_output,
        tile_coord_mnkl,
    )
    return ((tiled_copy_output_r2s, tRS_s_output, copy_output),)
