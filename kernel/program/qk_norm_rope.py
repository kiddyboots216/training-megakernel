"""Qwen3's per-head q/k RMSNorm and rotary embedding, and the ordered norm-gradient reducers.

`qk_norm_rope_forward` normalizes each query or key head of the packed QKV rows with RMSNorm
and the learned HEAD_DIM weight, applies rotate-half RoPE, and writes FA4's q or k tensor.
`qk_norm_rope_backward` recomputes rstd from the raw values, writes the gradient of the raw
columns into the packed dQKV rows, and writes one FP32 weight-gradient partial per task.  A
task is one (head, 128-row block).  Both bodies run on all 384 threads of every CTA, called
from decoder_layer.run_row_phase.

`sum_q_norm_partials` and `sum_k_norm_partials` sum those partials into the q and k norm-weight
gradients, and `sum_partial_rows` sums the RMSNorm weight-gradient partials.  All three add the
rows in row order, so the gradients are deterministic.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, const_expr
from cutlass._mlir.dialects import math as mlir_math
from cutlass.cutlass_dsl import dsl_user_op

from model import (
    HEAD_DIM,
    KV_HEADS,
    NORM_BLOCK_ROWS,
    NORM_BLOCKS,
    PROGRAM_THREADS,
    PROGRAM_WARPS,
    QKV_HIDDEN,
    QUERY_HEADS,
    RMS_EPSILON,
    SEQUENCE,
)
from program_smem import ProgramSmemAllocator


@dsl_user_op
def fma_f32(a, b, c, *, loc=None, ip=None) -> Float32:
    """`a * b + c` with one rounding, as an explicit MLIR `math.fma` (one FFMA).

    The backward writes its multiply-adds with this, so which product is fused, and so the
    rounding, is fixed by the source rather than left to the compiler's contraction choices.
    """

    return Float32(
        mlir_math.fma(
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            Float32(c).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )



@cute.jit
def qk_norm_rope_forward(
    qkv_flat: cute.Tensor,
    norm_weight: cute.Tensor,
    cos_flat: cute.Tensor,
    sin_flat: cute.Tensor,
    out_flat: cute.Tensor,
    head_count: cutlass.Constexpr[int],
    qkv_head_offset: cutlass.Constexpr[int],
    task_waves: cutlass.Constexpr[int],
):
    """RMSNorm with the learned weight, then rotate-half RoPE, for `head_count` q or k heads.

    Reads head `qkv_head_offset + h` of each packed row of `qkv_flat` [SEQUENCE, QKV_HIDDEN]
    and writes head `h` of `out_flat` [SEQUENCE, head_count * HEAD_DIM].  CTAs take the tasks,
    one per (head, 128-row block), in `task_waves` grid-stride waves.  Each quarter-warp (8
    lanes) owns one row at a time, so a CTA works on 48 rows at once; each lane handles 16 of
    the row's 128 columns.
    """

    bidx, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    quarter = lane // 8
    quarter_lane = lane - quarter * 8
    row_worker = warp * 4 + quarter
    row_workers = PROGRAM_WARPS * 4
    # Each row owner caches its raw BF16 values in a private shared-memory row rather than
    # keeping sixteen FP32 values per lane live across the reduction.  The lane stride of 17
    # halfwords spreads the lanes' same-instruction accesses over the banks.
    row_cache_stride = 8 * 17
    row_smem = ProgramSmemAllocator()
    row_cache = row_smem.allocate_tensor(
        BFloat16,
        cute.make_layout(PROGRAM_WARPS * 4 * row_cache_stride),
        byte_alignment=1024,
    )
    lane_cache_base = row_worker * row_cache_stride + quarter_lane * 17
    # The tasks cover all SEQUENCE rows.  The rows can hold several packed documents, and
    # `cos_flat` and `sin_flat` already give each row its position within its document, so the
    # body needs no document boundaries.
    segment_begin = Int32(0)
    segment_end = Int32(SEQUENCE)
    for wave in cutlass.range_constexpr(task_waves):
        task = bidx + wave * grid_x
        if task < head_count * NORM_BLOCKS:
            head = task // NORM_BLOCKS
            block = task - head * NORM_BLOCKS
            row_begin = segment_begin + block * NORM_BLOCK_ROWS
            row_end = cutlass.min(row_begin + NORM_BLOCK_ROWS, segment_end)
            for row in cutlass.range(
                row_begin + row_worker, row_end, row_workers, unroll=1
            ):
                in_base = row * QKV_HIDDEN + (qkv_head_offset + head) * HEAD_DIM
                out_base = row * (head_count * HEAD_DIM) + head * HEAD_DIM
                rope_base = row * HEAD_DIM

                # With one warp per row, lane c would own columns c + 32 * i (as in the
                # backward).  This lane keeps the partial sums of squares of lanes q, q + 8,
                # q + 16 and q + 24 (q = quarter_lane) and combines them as the XOR 16 and
                # XOR 8 butterfly steps would.  With the XOR 4/2/1 shuffles below, the
                # additions happen in the order qk_norm_rope_backward uses to recompute rstd.
                partial0 = Float32(0.0)
                partial8 = Float32(0.0)
                partial16 = Float32(0.0)
                partial24 = Float32(0.0)
                for item in cutlass.range_constexpr(4):
                    base_index = quarter_lane + item * 32
                    raw0_bf16 = qkv_flat[in_base + base_index]
                    raw8_bf16 = qkv_flat[in_base + base_index + 8]
                    raw16_bf16 = qkv_flat[in_base + base_index + 16]
                    raw24_bf16 = qkv_flat[in_base + base_index + 24]
                    slot = item * 4
                    row_cache[lane_cache_base + slot] = raw0_bf16
                    row_cache[lane_cache_base + slot + 1] = raw8_bf16
                    row_cache[lane_cache_base + slot + 2] = raw16_bf16
                    row_cache[lane_cache_base + slot + 3] = raw24_bf16
                    raw0 = raw0_bf16.to(Float32)
                    raw8 = raw8_bf16.to(Float32)
                    raw16 = raw16_bf16.to(Float32)
                    raw24 = raw24_bf16.to(Float32)
                    partial0 += raw0 * raw0
                    partial8 += raw8 * raw8
                    partial16 += raw16 * raw16
                    partial24 += raw24 * raw24
                total = (partial0 + partial16) + (partial8 + partial24)
                for step in cutlass.range_constexpr(3):
                    total += cute.arch.shuffle_sync_bfly(total, 4 >> step)
                rstd = cute.math.rsqrt(
                    total / Float32(float(HEAD_DIM)) + Float32(RMS_EPSILON),
                    fastmath=True,
                )

                for item in cutlass.range_constexpr(16):
                    index = quarter_lane + item * 8
                    if const_expr(item < 8):
                        partner_item = item + 8
                        partner = index + 64
                    else:
                        partner_item = item - 8
                        partner = index - 64
                    value = (
                        row_cache[lane_cache_base + item].to(Float32)
                        * rstd
                        * norm_weight[index].to(Float32)
                    )
                    rotated = (
                        row_cache[lane_cache_base + partner_item].to(Float32)
                        * rstd
                        * norm_weight[partner].to(Float32)
                    )
                    if const_expr(item < 8):
                        rotated = Float32(0.0) - rotated
                    cos_value = cos_flat[rope_base + index].to(Float32)
                    sin_value = sin_flat[rope_base + index].to(Float32)
                    out_flat[out_base + index] = (
                        value * cos_value + rotated * sin_value
                    ).to(BFloat16)


@cute.jit
def qk_norm_rope_backward(
    drotary_flat: cute.Tensor,
    qkv_flat: cute.Tensor,
    norm_weight: cute.Tensor,
    cos_flat: cute.Tensor,
    sin_flat: cute.Tensor,
    dqkv_flat: cute.Tensor,
    partial_flat: cute.Tensor,
    smem_partial: cute.Tensor,
    head_count: cutlass.Constexpr[int],
    qkv_head_offset: cutlass.Constexpr[int],
    out_offset_elements: cutlass.Constexpr[int],
    task_waves: cutlass.Constexpr[int],
):
    """Backward of `qk_norm_rope_forward` for `head_count` heads.

    From the gradient of the rotated output (`drotary_flat`), writes the gradient of the raw q
    or k columns into the packed dQKV rows of `dqkv_flat`, at column offset
    `out_offset_elements` (row stride QKV_HIDDEN), and the task's norm-weight gradient partial
    into row `task` of `partial_flat`.  One warp handles one row, each lane owning columns
    lane + 32 * i.  The partial is deterministic: each warp adds into its own row of
    `smem_partial` (PROGRAM_WARPS x HEAD_DIM), then threads 0-127 add the warp rows in order.
    """

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    # As in the forward, the tasks cover all SEQUENCE rows; the RoPE tables carry the
    # positions within each document.
    segment_begin = Int32(0)
    segment_end = Int32(SEQUENCE)
    for wave in cutlass.range_constexpr(task_waves):
        task = bidx + wave * grid_x
        if task < head_count * NORM_BLOCKS:
            head = task // NORM_BLOCKS
            block = task - head * NORM_BLOCKS
            row_begin = segment_begin + block * NORM_BLOCK_ROWS
            row_end = cutlass.min(row_begin + NORM_BLOCK_ROWS, segment_end)
            for item in cutlass.range_constexpr(4):
                smem_partial[warp * HEAD_DIM + lane + item * 32] = Float32(0.0)
            for row in cutlass.range(row_begin + warp, row_end, PROGRAM_WARPS, unroll=1):
                in_base = row * QKV_HIDDEN + (qkv_head_offset + head) * HEAD_DIM
                grad_base = row * (head_count * HEAD_DIM) + head * HEAD_DIM
                out_base = row * QKV_HIDDEN + out_offset_elements + head * HEAD_DIM
                rope_base = row * HEAD_DIM
                raw0 = qkv_flat[in_base + lane].to(Float32)
                raw1 = qkv_flat[in_base + lane + 32].to(Float32)
                raw2 = qkv_flat[in_base + lane + 64].to(Float32)
                raw3 = qkv_flat[in_base + lane + 96].to(Float32)
                square_sum = Float32(0.0)
                square_sum += raw0 * raw0
                square_sum += raw1 * raw1
                square_sum += raw2 * raw2
                square_sum += raw3 * raw3
                total = square_sum
                for step in cutlass.range_constexpr(5):
                    total += cute.arch.shuffle_sync_bfly(total, 16 >> step)
                rstd = cute.math.rsqrt(
                    total / Float32(float(HEAD_DIM)) + Float32(RMS_EPSILON),
                    fastmath=True,
                )
                # dz, the gradient before the rotation: dy[c] * cos[c], plus
                # dy[c + 64] * sin[c + 64] for c < 64, or minus dy[c - 64] * sin[c - 64]
                # for c >= 64.  x_hat, the weights and dz are used on both sides of the warp
                # reduction.
                xhat_values = (
                    raw0 * rstd,
                    raw1 * rstd,
                    raw2 * rstd,
                    raw3 * rstd,
                )
                weight_values = (
                    norm_weight[lane].to(Float32),
                    norm_weight[lane + 32].to(Float32),
                    norm_weight[lane + 64].to(Float32),
                    norm_weight[lane + 96].to(Float32),
                )
                own0 = drotary_flat[grad_base + lane].to(Float32) * cos_flat[
                    rope_base + lane
                ].to(Float32)
                own1 = drotary_flat[grad_base + lane + 32].to(Float32) * cos_flat[
                    rope_base + lane + 32
                ].to(Float32)
                dz0 = fma_f32(
                    drotary_flat[grad_base + lane + 64].to(Float32),
                    sin_flat[rope_base + lane + 64].to(Float32),
                    own0,
                )
                dz1 = fma_f32(
                    drotary_flat[grad_base + lane + 96].to(Float32),
                    sin_flat[rope_base + lane + 96].to(Float32),
                    own1,
                )
                cross2 = drotary_flat[grad_base + lane].to(Float32) * sin_flat[
                    rope_base + lane
                ].to(Float32)
                cross3 = drotary_flat[grad_base + lane + 32].to(Float32) * sin_flat[
                    rope_base + lane + 32
                ].to(Float32)
                dz2 = fma_f32(
                    drotary_flat[grad_base + lane + 64].to(Float32),
                    cos_flat[rope_base + lane + 64].to(Float32),
                    Float32(0.0) - cross2,
                )
                dz3 = fma_f32(
                    drotary_flat[grad_base + lane + 96].to(Float32),
                    cos_flat[rope_base + lane + 96].to(Float32),
                    Float32(0.0) - cross3,
                )
                dz_values = (dz0, dz1, dz2, dz3)
                dot = Float32(0.0)
                for item in cutlass.range_constexpr(4):
                    dot = fma_f32(
                        xhat_values[item],
                        dz_values[item] * weight_values[item],
                        dot,
                    )
                dot_total = dot
                for step in cutlass.range_constexpr(5):
                    dot_total += cute.arch.shuffle_sync_bfly(dot_total, 16 >> step)
                dot_mean = dot_total / Float32(float(HEAD_DIM))
                for item in cutlass.range_constexpr(4):
                    index = lane + item * 32
                    dqkv_flat[out_base + index] = (
                        rstd
                        * fma_f32(
                            dz_values[item],
                            weight_values[item],
                            Float32(0.0) - xhat_values[item] * dot_mean,
                        )
                    ).to(BFloat16)
                    cell = warp * HEAD_DIM + index
                    smem_partial[cell] = (
                        smem_partial[cell] + dz_values[item] * xhat_values[item]
                    )
            cute.arch.sync_threads()
            if tidx < HEAD_DIM:
                column_total = Float32(0.0)
                for source_warp in cutlass.range(0, PROGRAM_WARPS, 1, unroll=1):
                    column_total += smem_partial[source_warp * HEAD_DIM + tidx]
                partial_flat[task * HEAD_DIM + tidx] = column_total
            cute.arch.sync_threads()


@cute.jit
def sum_partial_rows(
    partials_flat: cute.Tensor,
    output_flat: cute.Tensor,
    partial_rows: cutlass.Constexpr[int],
    output_elements: cutlass.Constexpr[int],
):
    """output[e] = the sum of partials[r, e] over r = 0 .. partial_rows - 1, in row order.

    The fixed order makes the FP32 sum deterministic.  Every thread of the grid strides over
    the `output_elements` columns.
    """

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    global_thread = bidx * PROGRAM_THREADS + tidx
    global_threads = grid_x * PROGRAM_THREADS
    for element in cutlass.range(global_thread, output_elements, global_threads, unroll=1):
        total = Float32(0.0)
        for row in cutlass.range(0, partial_rows, 1, unroll=1):
            total += partials_flat[row * output_elements + element]
        output_flat[element] = total


@cute.jit
def sum_q_norm_partials(
    partials_flat: cute.Tensor,
    output_flat: cute.Tensor,
):
    """The q-norm weight gradient: the QUERY_HEADS * NORM_BLOCKS partial rows, summed in order.

    Each output column is one thread's sum.  Eight rows are loaded before they are added, which
    keeps eight loads in flight without changing the order of the additions.
    """

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    global_thread = bidx * PROGRAM_THREADS + tidx
    global_threads = grid_x * PROGRAM_THREADS
    for element in cutlass.range(global_thread, HEAD_DIM, global_threads, unroll=1):
        total = Float32(0.0)
        for row in cutlass.range(0, QUERY_HEADS * NORM_BLOCKS, 8, unroll=1):
            value0 = partials_flat[row * HEAD_DIM + element]
            value1 = partials_flat[(row + Int32(1)) * HEAD_DIM + element]
            value2 = partials_flat[(row + Int32(2)) * HEAD_DIM + element]
            value3 = partials_flat[(row + Int32(3)) * HEAD_DIM + element]
            value4 = partials_flat[(row + Int32(4)) * HEAD_DIM + element]
            value5 = partials_flat[(row + Int32(5)) * HEAD_DIM + element]
            value6 = partials_flat[(row + Int32(6)) * HEAD_DIM + element]
            value7 = partials_flat[(row + Int32(7)) * HEAD_DIM + element]
            total += value0
            total += value1
            total += value2
            total += value3
            total += value4
            total += value5
            total += value6
            total += value7
        output_flat[element] = total


@cute.jit
def sum_k_norm_partials(
    partials_flat: cute.Tensor,
    output_flat: cute.Tensor,
):
    """The k-norm weight gradient: the KV_HEADS * NORM_BLOCKS partial rows, summed in order.

    As `sum_q_norm_partials`.  Both row counts (SEQUENCE / 4 and SEQUENCE / 16) are multiples
    of 8 at every supported sequence length, so the eight-row groups need no remainder loop.
    """

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    global_thread = bidx * PROGRAM_THREADS + tidx
    global_threads = grid_x * PROGRAM_THREADS
    for element in cutlass.range(global_thread, HEAD_DIM, global_threads, unroll=1):
        total = Float32(0.0)
        for row in cutlass.range(0, KV_HEADS * NORM_BLOCKS, 8, unroll=1):
            value0 = partials_flat[row * HEAD_DIM + element]
            value1 = partials_flat[(row + Int32(1)) * HEAD_DIM + element]
            value2 = partials_flat[(row + Int32(2)) * HEAD_DIM + element]
            value3 = partials_flat[(row + Int32(3)) * HEAD_DIM + element]
            value4 = partials_flat[(row + Int32(4)) * HEAD_DIM + element]
            value5 = partials_flat[(row + Int32(5)) * HEAD_DIM + element]
            value6 = partials_flat[(row + Int32(6)) * HEAD_DIM + element]
            value7 = partials_flat[(row + Int32(7)) * HEAD_DIM + element]
            total += value0
            total += value1
            total += value2
            total += value3
            total += value4
            total += value5
            total += value6
            total += value7
        output_flat[element] = total
