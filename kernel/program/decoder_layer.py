"""The decoder layer's tables and its row phases.

The tables describe one layer: its twelve projection GEMMs (`GEMM_FAMILY_BY_NAME`, of
`GemmFamily`), its per-layer tensors (`SLAB_SPECS` and `TRANSPOSED_WEIGHT_VIEWS`), the columns
of the row table (`ROW_TABLE_FIELDS`), and its phases in order (`LAYER_PHASES`), from which
training_program generates the step.  Import-time checks tie each GEMM's shape to the tensors
it names.

`run_row_phase` runs one row phase (an RMSNorm or its backward, a q/k norm-RoPE body, or an
RMSNorm weight-gradient reduction) with operands read through the row table; the program class
binds it as a method.  `LayerTensors` allocates one copy of the per-layer tensors per layer
slot, and `stacked_gemm_operand` turns them into kernel arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, const_expr
from cutlass.cute.runtime import from_dlpack

from anchors import row_table_view
from model import (
    GATE_UP,
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    KV_HEADS,
    KV_HIDDEN,
    NORM_BLOCKS,
    PROGRAM_CTAS,
    PROGRAM_WARPS,
    QKV_HIDDEN,
    QUERY_HEADS,
    QUERY_HIDDEN,
    RMS_EPSILON,
    SEQUENCE,
)
from attention import AttentionPlan, AttentionTensors
from program_smem import ProgramSmemAllocator
from qk_norm_rope import qk_norm_rope_backward, qk_norm_rope_forward, sum_partial_rows
from quack_rmsnorm_bodies import (
    RMS_FORWARD_WAVES,
    RMS_PARTIAL_ROWS,
    restore_full_page,
    rmsnorm_backward_body,
    rmsnorm_forward_body,
    use_rms_shard_page,
)


# The layer's twelve projection GEMMs.  A family fixes the GEMM's shape and names its operands
# by slab or transposed weight view; for each layer slot G:
#
#   A is [G, M, K], or [G, K, M] when `a_t` (the dW families read dY transposed)
#   B is [G, K, N]
#   D is [G, M, N], and D = (A.mT if a_t else A) @ B
#
# B is the stored weight for dX (N-major), the forward activation for dW (N-major) and a
# `*_weight_t` view of the weight for the forward projections (K-major).


@dataclass(frozen=True)
class GemmFamily:
    """One projection GEMM of the layer: its (M, N, K), its operands and D's dtype."""

    name: str
    m: int
    n: int
    k: int
    a: str          # A slab name
    a_t: bool
    b: str          # B slab or transposed weight view, [G, K, N]
    d: str          # D slab name
    d_dtype: torch.dtype


# Grid-stride waves of the q and k norm-RoPE bodies, one task per (head, 128-row block).
Q_NORM_ROPE_WAVES = -(-(QUERY_HEADS * NORM_BLOCKS) // PROGRAM_CTAS)
K_NORM_ROPE_WAVES = -(-(KV_HEADS * NORM_BLOCKS) // PROGRAM_CTAS)


# Each operand is the tensor the neighboring phases produce or consume: for example `o_fwd`
# writes `residual_mid`, `down_fwd` writes `layer_output`, and `o_dx` writes `attention_dout`,
# which is FA4's dO buffer.  A family's member can store more or other outputs from its
# epilogue (the residual add, SwiGLU and its backward, the V heads); see
# attention_projections.py and mlp_projections.py.
GEMM_FAMILY_BY_NAME: dict[str, GemmFamily] = {
    f.name: f
    for f in (
        # Forward projections.
        GemmFamily("qkv_fwd", SEQUENCE, QKV_HIDDEN, HIDDEN,
               "norm1", False, "qkv_weight_t", "qkv_raw", torch.bfloat16),
        GemmFamily("o_fwd", SEQUENCE, HIDDEN, HIDDEN,
               "attention_out", False, "o_weight_t", "residual_mid", torch.bfloat16),
        GemmFamily("gate_up_fwd", SEQUENCE, GATE_UP, HIDDEN,
               "norm2", False, "gate_up_weight_t", "gate_up", torch.bfloat16),
        GemmFamily("down_fwd", SEQUENCE, HIDDEN, INTERMEDIATE,
               "mlp_intermediate", False, "down_weight_t", "layer_output", torch.bfloat16),
        # Backward dX.
        GemmFamily("down_dx", SEQUENCE, INTERMEDIATE, HIDDEN,
               "layer_output_dy", False, "down_weight", "mlp_intermediate_dy", torch.bfloat16),
        GemmFamily("gate_up_dx", SEQUENCE, HIDDEN, GATE_UP,
               "gate_up_dy", False, "gate_up_weight", "norm2_dy", torch.bfloat16),
        GemmFamily("o_dx", SEQUENCE, HIDDEN, HIDDEN,
               "residual_mid_dy", False, "o_weight", "attention_dout", torch.bfloat16),
        GemmFamily("qkv_dx", SEQUENCE, HIDDEN, QKV_HIDDEN,
               "dqkv_raw", False, "qkv_weight", "norm1_dy", torch.bfloat16),
        # Backward dW, with FP32 outputs.
        GemmFamily("down_dw", HIDDEN, INTERMEDIATE, SEQUENCE,
               "layer_output_dy", True, "mlp_intermediate", "down_grad", torch.float32),
        GemmFamily("gate_up_dw", GATE_UP, HIDDEN, SEQUENCE,
               "gate_up_dy", True, "norm2", "gate_up_grad", torch.float32),
        GemmFamily("o_dw", HIDDEN, HIDDEN, SEQUENCE,
               "residual_mid_dy", True, "attention_out", "o_grad", torch.float32),
        GemmFamily("qkv_dw", QKV_HIDDEN, HIDDEN, SEQUENCE,
               "dqkv_raw", True, "norm1", "qkv_grad", torch.float32),
    )
}


# The layer's tensors ("slabs").  Each is stored once per layer slot, as [G, *shape], unless it
# is `shared`.

@dataclass(frozen=True)
class SlabSpec:
    """One per-layer tensor: its shape within a layer slot, dtype, alias and sharing."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    alias: str | None = None        # AttentionTensors field this slab aliases (view, not copy)
    shared: bool = False            # one tensor for all layer slots (the RoPE tables)


SLAB_SPECS: dict[str, SlabSpec] = {
    # Inputs: the layer input, the incoming gradient, the weights, the norm weights and the
    # RoPE tables.
    "residual_in":         SlabSpec((SEQUENCE, HIDDEN), torch.bfloat16),
    "staging_dy":          SlabSpec((SEQUENCE, HIDDEN), torch.bfloat16),
    "input_norm":          SlabSpec((HIDDEN,), torch.bfloat16),
    "qkv_weight":          SlabSpec((QKV_HIDDEN, HIDDEN), torch.bfloat16),
    "q_norm":              SlabSpec((HEAD_DIM,), torch.bfloat16),
    "k_norm":              SlabSpec((HEAD_DIM,), torch.bfloat16),
    "o_weight":            SlabSpec((HIDDEN, HIDDEN), torch.bfloat16),
    "post_attention_norm": SlabSpec((HIDDEN,), torch.bfloat16),
    "gate_up_weight":      SlabSpec((GATE_UP, HIDDEN), torch.bfloat16),
    "down_weight":         SlabSpec((HIDDEN, INTERMEDIATE), torch.bfloat16),
    "rotary_cos":          SlabSpec((SEQUENCE, HEAD_DIM), torch.bfloat16, shared=True),
    "rotary_sin":          SlabSpec((SEQUENCE, HEAD_DIM), torch.bfloat16, shared=True),
    # Forward activations.
    "norm1":               SlabSpec((SEQUENCE, HIDDEN), torch.bfloat16),
    "rms1_rstd":           SlabSpec((SEQUENCE,), torch.float32),
    "qkv_raw":             SlabSpec((SEQUENCE, QKV_HIDDEN), torch.bfloat16),
    "v":                   SlabSpec((SEQUENCE, KV_HIDDEN), torch.bfloat16, alias="v"),
    "q_rotary":            SlabSpec((SEQUENCE, QUERY_HIDDEN), torch.bfloat16, alias="q"),
    "k_rotary":            SlabSpec((SEQUENCE, KV_HIDDEN), torch.bfloat16, alias="k"),
    "attention_out":       SlabSpec((SEQUENCE, HIDDEN), torch.bfloat16, alias="out"),
    "residual_mid":        SlabSpec((SEQUENCE, HIDDEN), torch.bfloat16),
    "norm2":               SlabSpec((SEQUENCE, HIDDEN), torch.bfloat16),
    "rms2_rstd":           SlabSpec((SEQUENCE,), torch.float32),
    "gate_up":             SlabSpec((SEQUENCE, GATE_UP), torch.bfloat16),
    "mlp_intermediate":    SlabSpec((SEQUENCE, INTERMEDIATE), torch.bfloat16),
    "layer_output":        SlabSpec((SEQUENCE, HIDDEN), torch.bfloat16),
    # Backward gradients.
    "layer_output_dy":     SlabSpec((SEQUENCE, HIDDEN), torch.bfloat16),
    "mlp_intermediate_dy": SlabSpec((SEQUENCE, INTERMEDIATE), torch.bfloat16),
    "gate_up_dy":          SlabSpec((SEQUENCE, GATE_UP), torch.bfloat16),
    "norm2_dy":            SlabSpec((SEQUENCE, HIDDEN), torch.bfloat16),
    "residual_mid_dy":     SlabSpec((SEQUENCE, HIDDEN), torch.bfloat16),
    "attention_dout":      SlabSpec((SEQUENCE, HIDDEN), torch.bfloat16, alias="dout"),
    "dq_rotary":           SlabSpec((SEQUENCE, QUERY_HIDDEN), torch.bfloat16, alias="dq"),
    "dk_rotary":           SlabSpec((SEQUENCE, KV_HIDDEN), torch.bfloat16, alias="dk"),
    "dv":                  SlabSpec((SEQUENCE, KV_HIDDEN), torch.bfloat16, alias="dv"),
    "dqkv_raw":            SlabSpec((SEQUENCE, QKV_HIDDEN), torch.bfloat16),
    "norm1_dy":            SlabSpec((SEQUENCE, HIDDEN), torch.bfloat16),
    "layer_input_dx":      SlabSpec((SEQUENCE, HIDDEN), torch.bfloat16),
    # Norm-weight gradient partials and the weight gradients, in FP32.
    "input_norm_partial":     SlabSpec((RMS_PARTIAL_ROWS, HIDDEN), torch.float32),
    "post_attn_norm_partial": SlabSpec((RMS_PARTIAL_ROWS, HIDDEN), torch.float32),
    "q_norm_partial":         SlabSpec((QUERY_HEADS * NORM_BLOCKS, HEAD_DIM), torch.float32),
    "k_norm_partial":         SlabSpec((KV_HEADS * NORM_BLOCKS, HEAD_DIM), torch.float32),
    "input_norm_grad":        SlabSpec((HIDDEN,), torch.float32),
    "post_attn_norm_grad":    SlabSpec((HIDDEN,), torch.float32),
    "q_norm_grad":            SlabSpec((HEAD_DIM,), torch.float32),
    "k_norm_grad":            SlabSpec((HEAD_DIM,), torch.float32),
    "qkv_grad":               SlabSpec((QKV_HIDDEN, HIDDEN), torch.float32),
    "o_grad":                 SlabSpec((HIDDEN, HIDDEN), torch.float32),
    "gate_up_grad":           SlabSpec((GATE_UP, HIDDEN), torch.float32),
    "down_grad":              SlabSpec((HIDDEN, INTERMEDIATE), torch.float32),
}


# The forward projections take their weight as B in [K, N] form.  These names are `.mT` views of
# the stored [N, K] weight slabs, not slabs, so `LayerTensors.allocate` allocates nothing for
# them.  Quack takes B's major mode from its strides (a view is a K-major B).  TMA needs every
# non-unit stride to be a multiple of 16 bytes; every weight dimension (4096, 6144, 12288,
# 24576) is a multiple of 8 BF16 elements.
TRANSPOSED_WEIGHT_VIEWS: dict[str, str] = {
    "qkv_weight_t":     "qkv_weight",
    "o_weight_t":       "o_weight",
    "gate_up_weight_t": "gate_up_weight",
    "down_weight_t":    "down_weight",
}
assert not (set(TRANSPOSED_WEIGHT_VIEWS) & set(SLAB_SPECS)), "a transpose view must not also be a slab"
assert all(identity in SLAB_SPECS for identity in TRANSPOSED_WEIGHT_VIEWS.values())


def operand_shape(name: str) -> tuple[int, ...]:
    """The shape within one layer slot of a slab or a transposed weight view."""

    if name in SLAB_SPECS:
        return SLAB_SPECS[name].shape
    rows, columns = SLAB_SPECS[TRANSPOSED_WEIGHT_VIEWS[name]].shape
    return (columns, rows)


# Import-time checks: each family's (M, N, K) and D dtype agree with the operands it names.
for _f in GEMM_FAMILY_BY_NAME.values():
    _a, _b, _d = operand_shape(_f.a), operand_shape(_f.b), operand_shape(_f.d)
    assert _a == ((_f.k, _f.m) if _f.a_t else (_f.m, _f.k)), (_f.name, "A", _a)
    assert _b == (_f.k, _f.n), (_f.name, "B", _b)
    assert _d == (_f.m, _f.n), (_f.name, "D", _d)
    assert SLAB_SPECS[_f.d].dtype is _f.d_dtype, (_f.name, "D dtype")


# The row table's columns.  The row table holds one row of Int64 addresses per layer, in this
# order (training_tensors builds it); the row phases read their operands through the row of the
# layer they run for (`anchors.row_table_view`).  The GEMMs instead read the slot-stacked slabs
# by TMA and take the layer slot from the generation slot word (Quack's batch_idx_permute).
ROW_TABLE_FIELDS = (
    "residual_in", "input_norm", "norm1", "rms1_rstd",
    "qkv_raw", "v", "attention_out", "q_norm", "k_norm", "rotary_cos", "rotary_sin",
    "q_rotary", "k_rotary",
    "residual_mid", "post_attention_norm", "norm2", "rms2_rstd",
    "gate_up", "mlp_intermediate", "layer_output",
    "staging_dy", "layer_output_dy", "mlp_intermediate_dy", "gate_up_dy",
    "norm2_dy", "residual_mid_dy",
    "post_attn_norm_partial", "post_attn_norm_grad",
    "dq_rotary", "dk_rotary", "dv", "dqkv_raw",
    "q_norm_partial", "k_norm_partial", "q_norm_grad", "k_norm_grad",
    "norm1_dy", "layer_input_dx", "input_norm_partial", "input_norm_grad",
)
ROW_TABLE_INDEX = {name: index for index, name in enumerate(ROW_TABLE_FIELDS)}
ROW_TABLE_WIDTH = len(ROW_TABLE_FIELDS)
assert all(name in SLAB_SPECS for name in ROW_TABLE_FIELDS)


# The phases of one layer visit in order, as (name, kind).  kind is "gemm" (the family's Quack
# member), "fa4f" or "fa4b" (FA4's forward or backward) or "row" (`run_row_phase`).
# training_program generates the step from this list.  It leaves out `dy_ready` and the row
# phases whose work a GEMM epilogue does (the residual adds and SwiGLU), and it keeps
# `v_extract`, `dv_insert` and the q/k norm reductions only as their grid barrier (CTA 0 runs
# the q/k norm reductions during `qkv_dx`).

LAYER_PHASES: tuple[tuple[str, str], ...] = (
    ("rms1_fwd",              "row"),
    ("qkv_fwd",               "gemm"),
    ("v_extract",             "row"),
    ("q_norm_rope_fwd",       "row"),
    ("k_norm_rope_fwd",       "row"),
    ("fa4_fwd_main",          "fa4f"),
    ("o_fwd",                 "gemm"),
    ("o_fwd_residual",        "row"),
    ("rms2_fwd",              "row"),
    ("gate_up_fwd",           "gemm"),
    ("swiglu_fwd",            "row"),
    ("down_fwd",              "gemm"),
    ("down_fwd_residual",     "row"),
    ("dy_ready",              "row"),
    ("down_dx",               "gemm"),
    ("down_dw",               "gemm"),
    ("swiglu_bwd",            "row"),
    ("gate_up_dx",            "gemm"),
    ("gate_up_dw",            "gemm"),
    ("rms2_bwd",              "row"),
    ("post_attn_norm_reduce", "row"),
    ("o_dx",                  "gemm"),
    ("o_dw",                  "gemm"),
    ("fa4_bwd_main",          "fa4b"),
    ("dv_insert",             "row"),
    ("q_norm_rope_bwd",       "row"),
    ("k_norm_rope_bwd",       "row"),
    ("q_norm_reduce",         "row"),
    ("k_norm_reduce",         "row"),
    ("qkv_dx",                "gemm"),
    ("qkv_dw",                "gemm"),
    ("rms1_bwd",              "row"),
    ("input_norm_reduce",     "row"),
)

# The forward phases: those through `down_fwd_residual`, which produces the layer's output.
FORWARD_LAYER_PHASES = LAYER_PHASES[: [n for n, _ in LAYER_PHASES].index("dy_ready")]


@cute.jit
def rms_backward_from_row_table(
    self,
    which: cutlass.Constexpr[str],
    generation: Int32,
    row_table: cute.Tensor,
    rms_bwd_tiler_mn,
    rms_bwd_tiled_copy,
    rms_bwd_threads_per_row: cutlass.Constexpr[int],
    shard: cutlass.Constexpr[int],
):
    """RMS1 or RMS2 backward (`which`) for layer `generation` on shard `shard`.

    The operands come from the layer's row of `row_table`.  dX includes the residual gradient;
    dW goes to the shard's row of the norm's partial, which the matching `*_norm_reduce` phase
    sums.
    """

    memspace = row_table.iterator.memspace
    base = generation * Int32(ROW_TABLE_WIDTH)
    matrix_h = cute.make_layout((SEQUENCE, HIDDEN), stride=(HIDDEN, 1))
    weight_h = cute.make_layout((1, HIDDEN), stride=(0, 1))
    bwd_rstd_h = cute.make_layout(SEQUENCE)
    partial_h = cute.make_layout(
        (RMS_PARTIAL_ROWS, HIDDEN), stride=(HIDDEN, 1)
    )

    def view(name: cutlass.Constexpr[str], dtype, layout):
        return row_table_view(
            row_table,
            memspace,
            base + Int32(ROW_TABLE_INDEX[name]),
            dtype,
            layout,
        )

    if const_expr(which == "rms2_bwd"):
        member = self.rms2_bwd
        x = view("residual_mid", BFloat16, matrix_h)
        weight = view("post_attention_norm", BFloat16, weight_h)
        dout = view("norm2_dy", BFloat16, matrix_h)
        # RMS2's residual gradient is the gradient arriving from the layer above, which the
        # row's `staging_dy` entry points at.  The row's `layer_output_dy` entry holds the down
        # projection's forward output instead.
        dres_o = view("staging_dy", BFloat16, matrix_h)
        rstd = view("rms2_rstd", Float32, bwd_rstd_h)
        dx = view("residual_mid_dy", BFloat16, matrix_h)
        dweight = view("post_attn_norm_partial", Float32, partial_h)
    else:
        member = self.rms1_bwd
        x = view("residual_in", BFloat16, matrix_h)
        weight = view("input_norm", BFloat16, weight_h)
        dout = view("norm1_dy", BFloat16, matrix_h)
        dres_o = view("residual_mid_dy", BFloat16, matrix_h)
        rstd = view("rms1_rstd", Float32, bwd_rstd_h)
        dx = view("layer_input_dx", BFloat16, matrix_h)
        dweight = view("input_norm_partial", Float32, partial_h)

    rmsnorm_backward_body(
        member,
        x,
        weight,
        dout,
        dres_o,
        rstd,
        dx,
        dweight,
        None,
        None,
        None,
        None,
        None,
        None,
        rms_bwd_tiler_mn,
        rms_bwd_tiled_copy,
        rms_bwd_threads_per_row,
        shard=shard,
    )


@cute.jit
def run_row_phase(
    self,
    which: cutlass.Constexpr[str],
    role: cutlass.Constexpr[int],
    generation: Int32,
    phase_counter: cute.Tensor,
    row_table: cute.Tensor,
    rms_fwd_tiler_mn,
    rms_fwd_tiled_copy,
    rms_fwd_threads_per_row: cutlass.Constexpr[int],
    rms_bwd_tiler_mn,
    rms_bwd_tiled_copy,
    rms_bwd_threads_per_row: cutlass.Constexpr[int],
):
    """Run row phase `which` for layer `generation` in role `role`, then the grid barrier.

    The operands come from the layer's row of `row_table`.  The RMSNorm phases run on roles 1
    and 2 as two 128-thread shards, and role 0 only joins the barrier; the q/k norm-RoPE phases
    and the RMSNorm weight-gradient reductions run on all three roles.
    """
    if const_expr(
        which == "post_attn_norm_reduce" or which == "input_norm_reduce"
    ):
        # Each RMS shard of the grid wrote one partial row; add them in row order.
        self.reanchor_smem_page()
        memspace = row_table.iterator.memspace
        base = generation * Int32(ROW_TABLE_WIDTH)

        def reduce_view(name: cutlass.Constexpr[str], dtype, layout):
            return row_table_view(
                row_table,
                memspace,
                base + Int32(ROW_TABLE_INDEX[name]),
                dtype,
                layout,
            )

        flat_2ctas_h = cute.make_layout(RMS_PARTIAL_ROWS * HIDDEN)
        if const_expr(which == "post_attn_norm_reduce"):
            partial_name = "post_attn_norm_partial"
            output_name = "post_attn_norm_grad"
        else:
            partial_name = "input_norm_partial"
            output_name = "input_norm_grad"
        sum_partial_rows(
            reduce_view(partial_name, Float32, flat_2ctas_h),
            reduce_view(output_name, Float32, cute.make_layout(HIDDEN)),
            RMS_PARTIAL_ROWS,
            HIDDEN,
        )
        self.grid_phase_barrier(phase_counter)
    elif const_expr(which == "rms1_bwd" or which == "rms2_bwd"):
        # Roles 1 and 2 run shards 0 and 1, each on its half of the page; role 0 runs nothing.
        self.reanchor_smem_page()
        full_page = ProgramSmemAllocator.page
        if const_expr(role == 1):
            use_rms_shard_page(full_page, 0)
            rms_backward_from_row_table(
                self,
                which,
                generation,
                row_table,
                rms_bwd_tiler_mn,
                rms_bwd_tiled_copy,
                rms_bwd_threads_per_row,
                shard=0,
            )
            restore_full_page(full_page)
        elif const_expr(role == 2):
            use_rms_shard_page(full_page, 1)
            rms_backward_from_row_table(
                self,
                which,
                generation,
                row_table,
                rms_bwd_tiler_mn,
                rms_bwd_tiled_copy,
                rms_bwd_threads_per_row,
                shard=1,
            )
            restore_full_page(full_page)
        self.grid_phase_barrier(phase_counter)
    elif const_expr(
        which == "rms1_fwd"
        or which == "rms2_fwd"
    ):
        # Operand views with static extents.  rstd is a [SEQUENCE] vector broadcast over the
        # columns (stride 0), as Quack's RMSNorm.__call__ expands it for the kernel.
        self.reanchor_smem_page()
        memspace = row_table.iterator.memspace
        base = generation * Int32(ROW_TABLE_WIDTH)
        matrix_h = cute.make_layout((SEQUENCE, HIDDEN), stride=(HIDDEN, 1))
        weight_h = cute.make_layout((1, HIDDEN), stride=(0, 1))
        fwd_rstd_h = cute.make_layout((SEQUENCE, HIDDEN), stride=(1, 0))

        def view(name: cutlass.Constexpr[str], dtype, layout):
            return row_table_view(
                row_table,
                memspace,
                base + Int32(ROW_TABLE_INDEX[name]),
                dtype,
                layout,
            )

        if const_expr(which == "rms1_fwd"):
            member = self.rms1_fwd
            x = view("residual_in", BFloat16, matrix_h)
            weight = view("input_norm", BFloat16, weight_h)
            out = view("norm1", BFloat16, matrix_h)
            rstd = view("rms1_rstd", Float32, fwd_rstd_h)
        elif const_expr(which == "rms2_fwd"):
            member = self.rms2_fwd
            x = view("residual_mid", BFloat16, matrix_h)
            weight = view("post_attention_norm", BFloat16, weight_h)
            out = view("norm2", BFloat16, matrix_h)
            rstd = view("rms2_rstd", Float32, fwd_rstd_h)

        full_page = ProgramSmemAllocator.page
        if const_expr(role == 1):
            use_rms_shard_page(full_page, 0)
            rmsnorm_forward_body(
                member, x, weight, None, None, out, None, rstd, None,
                Float32(RMS_EPSILON), rms_fwd_tiler_mn,
                rms_fwd_tiled_copy, rms_fwd_threads_per_row, RMS_FORWARD_WAVES,
                shard=0,
            )
            restore_full_page(full_page)
        elif const_expr(role == 2):
            use_rms_shard_page(full_page, 1)
            rmsnorm_forward_body(
                member, x, weight, None, None, out, None, rstd, None,
                Float32(RMS_EPSILON), rms_fwd_tiler_mn,
                rms_fwd_tiled_copy, rms_fwd_threads_per_row, RMS_FORWARD_WAVES,
                shard=1,
            )
            restore_full_page(full_page)
        # Role 0 runs no RMS work; all three roles meet at the grid barrier.
        self.grid_phase_barrier(phase_counter)
    else:
        self.reanchor_smem_page()
        memspace = row_table.iterator.memspace
        base = generation * Int32(ROW_TABLE_WIDTH)

        # Flat layouts with static extents; only the addresses from the row table are dynamic.
        flat_qkv = cute.make_layout(SEQUENCE * QKV_HIDDEN)
        flat_qh = cute.make_layout(SEQUENCE * QUERY_HIDDEN)
        flat_kvh = cute.make_layout(SEQUENCE * KV_HIDDEN)
        flat_d = cute.make_layout(HEAD_DIM)
        flat_rope = cute.make_layout(SEQUENCE * HEAD_DIM)
        flat_qpartial = cute.make_layout(QUERY_HEADS * NORM_BLOCKS * HEAD_DIM)
        flat_kpartial = cute.make_layout(KV_HEADS * NORM_BLOCKS * HEAD_DIM)

        def view(name: cutlass.Constexpr[str], dtype, layout):
            return row_table_view(row_table, memspace, base + Int32(ROW_TABLE_INDEX[name]), dtype, layout)

        if const_expr(which == "q_norm_rope_fwd"):
            qk_norm_rope_forward(
                view("qkv_raw", BFloat16, flat_qkv),
                view("q_norm", BFloat16, flat_d),
                view("rotary_cos", BFloat16, flat_rope),
                view("rotary_sin", BFloat16, flat_rope),
                view("q_rotary", BFloat16, flat_qh),
                QUERY_HEADS, 0, Q_NORM_ROPE_WAVES,
            )
        elif const_expr(which == "k_norm_rope_fwd"):
            qk_norm_rope_forward(
                view("qkv_raw", BFloat16, flat_qkv),
                view("k_norm", BFloat16, flat_d),
                view("rotary_cos", BFloat16, flat_rope),
                view("rotary_sin", BFloat16, flat_rope),
                view("k_rotary", BFloat16, flat_kvh),
                KV_HEADS, QUERY_HEADS, K_NORM_ROPE_WAVES,
            )
        elif const_expr(which == "q_norm_rope_bwd"):
            q_smem = ProgramSmemAllocator()
            qk_norm_rope_backward(
                view("dq_rotary", BFloat16, flat_qh),
                view("qkv_raw", BFloat16, flat_qkv),
                view("q_norm", BFloat16, flat_d),
                view("rotary_cos", BFloat16, flat_rope),
                view("rotary_sin", BFloat16, flat_rope),
                view("dqkv_raw", BFloat16, flat_qkv),
                view("q_norm_partial", Float32, flat_qpartial),
                q_smem.allocate_tensor(
                    Float32, cute.make_layout(PROGRAM_WARPS * HEAD_DIM), byte_alignment=16
                ),
                QUERY_HEADS, 0, 0, Q_NORM_ROPE_WAVES,
            )
        elif const_expr(which == "k_norm_rope_bwd"):
            k_smem = ProgramSmemAllocator()
            qk_norm_rope_backward(
                view("dk_rotary", BFloat16, flat_kvh),
                view("qkv_raw", BFloat16, flat_qkv),
                view("k_norm", BFloat16, flat_d),
                view("rotary_cos", BFloat16, flat_rope),
                view("rotary_sin", BFloat16, flat_rope),
                view("dqkv_raw", BFloat16, flat_qkv),
                view("k_norm_partial", Float32, flat_kpartial),
                k_smem.allocate_tensor(
                    Float32, cute.make_layout(PROGRAM_WARPS * HEAD_DIM), byte_alignment=16
                ),
                KV_HEADS, QUERY_HEADS, QUERY_HIDDEN, K_NORM_ROPE_WAVES,
            )
        else:
            raise ValueError(f"unknown row phase: {which}")

        self.grid_phase_barrier(phase_counter)


def gemm_families() -> tuple[GemmFamily, ...]:
    """The GEMM families in the order LAYER_PHASES runs them."""
    return tuple(GEMM_FAMILY_BY_NAME[name] for name, kind in LAYER_PHASES if kind == "gemm")


@dataclass
class LayerTensors:
    """The per-layer tensors of each layer slot: FA4's tensors (`base`) and the slot-stacked
    slabs with the transposed weight views.

    training_tensors.TrainingTensors starts from these; it adds the row table (one row per
    layer) and rebinds the layer-boundary slabs to the activation chain and the gradient ring.
    """

    base: AttentionTensors
    slabs: dict[str, torch.Tensor]
    families: tuple[GemmFamily, ...]
    capacity: int

    @classmethod
    def allocate(cls, plan_: AttentionPlan, families: tuple[GemmFamily, ...],
                 device: torch.device) -> "LayerTensors":
        base = AttentionTensors.allocate(plan_, device)
        capacity = plan_.generations
        slabs: dict[str, torch.Tensor] = {}

        for name, slab in SLAB_SPECS.items():
            if slab.alias is not None:
                # FA4's contiguous [G * S, heads, dim] tensor, viewed as [G, S, heads * dim].
                slabs[name] = getattr(base, slab.alias).view(capacity, SEQUENCE, -1)
            elif slab.shared:
                slabs[name] = torch.empty(slab.shape, dtype=slab.dtype, device=device)
            else:
                slabs[name] = torch.empty(
                    (capacity, *slab.shape), dtype=slab.dtype, device=device
                )

        # The transposed weight views: `.mT` of the [G, N, K] weight is the [G, K, N] B operand
        # over the same storage.  Check each slot's 16-byte alignment here, where the name is
        # known; stacked_gemm_operand binds the GEMM operands with assumed_align=16.
        for view_name, identity_name in TRANSPOSED_WEIGHT_VIEWS.items():
            view = slabs[identity_name].mT
            for generation in range(capacity):
                if view[generation].data_ptr() % 16:
                    raise AssertionError(f"{view_name}[{generation}] is not 16-B aligned")
            slabs[view_name] = view

        return cls(base=base, slabs=slabs, families=tuple(families), capacity=capacity)


def stacked_gemm_operand(slab: torch.Tensor, transposed: bool):
    """A [G, R, C] slab as a CuTe tensor in (R, C, G) order, or (C, R, G) when `transposed`,
    with static extents and 16-byte assumed alignment."""

    view = slab.permute(2, 1, 0) if transposed else slab.permute(1, 2, 0)
    return from_dlpack(view, assumed_align=16, enable_tvm_ffi=True)
