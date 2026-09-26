"""The LM head and loss: `HeadPhase`, its three GEMM members and the device code that runs them.

`HeadPhase.run` is the head's part of the shell in each step.  It moves the program from the
decoder's register split (56/224/224 for roles 0/1/2) to the head's (232/232/40), computes the
logits chunk by chunk into the dlogits slab (HEAD_CHUNK_ROWS rows per chunk, the chunk index
published as the forward GEMM's batch coordinate), turns them in place into per-token losses
and dlogits (`cross_entropy_rows`), runs the dX and dW GEMMs, sums the loss and returns to the
decoder's split.  The GEMMs are Quack default-epilogue members run by
`quack_gemm_bodies.gemm_body_for_warpgroup` on dynamic tile schedulers, with roles 0 and 1 as
the MMA warpgroups and role 2 as the load warpgroup.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, const_expr

from gemm_members import DefaultGemmMember
from grid_barrier import grid_barrier
from model import SEQUENCE
import model
from cutlass import Int64
from tile_schedulers import DynamicTileScheduler
from tile_schedulers import SCHEDULER_STATE_WORDS
from tile_schedulers import dynamic_scheduler_params
from tile_schedulers import dynamic_scheduler_params_offset
from quack_gemm_bodies import gemm_body_for_warpgroup

# Named barriers (all 384 threads) of the register-split switches around the head: into the
# head's split, and back to the decoder's.
HEAD_REGISTER_SPLIT_BARRIER = 10
DECODER_REGISTER_SPLIT_BARRIER = 13


@cute.jit
def aligned_flat_view(tensor: cute.Tensor, elements: cutlass.Constexpr[int]):
    """`tensor` as a flat, static-layout view of `elements` elements whose pointer is assumed
    16-byte aligned (the shell passes its tensors with dynamic layouts)."""

    pointer = cute.make_ptr(
        dtype=tensor.element_type,
        value=tensor.iterator.toint(),
        mem_space=tensor.iterator.memspace,
        assumed_align=16,
    )
    return cute.make_tensor(pointer, cute.make_layout(elements))


def make_head_forward_member() -> DefaultGemmMember:
    """The logits GEMM member: 320 x 128 tiles, not ping-pong, persistent; Quack sizes its
    pipeline stages from the program page."""

    member = DefaultGemmMember(
        Float32,
        BFloat16,
        (320, 128),
        (1, 1, 1),
        pingpong=False,
        is_persistent=True,
        gather_A=False,
        concat_layout=None,
    )
    # The head's register split, which HeadPhase.enter_head_register_split sets.  Quack's own
    # kernel requests these counts; gemm_body_for_warpgroup makes no such request and does not
    # read them.
    member.num_regs_load = 40
    member.num_regs_mma = 232
    return member


def make_head_backward_member() -> DefaultGemmMember:
    """The dX and dW GEMM member: 256 x 128 tiles, not ping-pong, persistent; Quack sizes its
    pipeline stages from the program page."""

    member = DefaultGemmMember(
        Float32,
        BFloat16,
        (256, 128),
        (1, 1, 1),
        pingpong=False,
        is_persistent=True,
        gather_A=False,
    )
    # The head's register split, as in make_head_forward_member.
    member.num_regs_load = 40
    member.num_regs_mma = 232
    return member


class HeadPhase:
    """The head phase: the forward, dX and dW GEMM members and the device code of `run`."""

    @cute.jit
    def publish_chunk_route(
        self,
        task_records: cute.Tensor,
        generation_slot: cute.Tensor,
        epoch: Int32,
        phase_counter: cute.Tensor,
    ):
        """CTA 0 writes chunk `epoch`'s index (`task_records[epoch]`) to `generation_slot[0]`,
        where the forward GEMM reads its batch coordinate; the grid barrier then makes the index
        visible to every CTA."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if bidx == 0:
            if tidx == 0:
                route = task_records[epoch]
                generation_slot[0] = route
        grid_barrier(phase_counter)

    @cute.jit
    def sum_token_losses(
        self,
        per_token_loss: cute.Tensor,
        loss: cute.Tensor,
        scratch: cute.Tensor,
        physical_warpgroup: cutlass.Constexpr[int],
    ):
        """`loss[0]` = the sum of `per_token_loss`, on CTA 0 in a fixed order: threads 0-255
        (roles 0 and 1) sum strided rows, each warp reduces by butterfly, and thread 0 adds the
        eight warp sums in order."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        total = Float32(0.0)
        if const_expr(physical_warpgroup < 2):
            if bidx == 0:
                for row in cutlass.range(tidx, SEQUENCE, 256, unroll=1):
                    total += per_token_loss[row]
                for offset in cutlass.range_constexpr(5):
                    total += cute.arch.shuffle_sync_bfly(total, 16 >> offset)
                lane = tidx % 32
                warp = tidx // 32
                if lane == 0:
                    scratch[warp] = total
        cute.arch.sync_threads()
        if bidx == 0 and tidx == 0:
            total = Float32(0.0)
            for warp in cutlass.range_constexpr(8):
                total += scratch[warp]
            loss[0] = total
        cute.arch.sync_threads()

    @cute.jit
    def cross_entropy_rows(
        self,
        slab_flat: cute.Tensor,
        labels: cute.Tensor,
        global_valid_tokens: cute.Tensor,
        per_token_loss: cute.Tensor,
        scratch: cute.Tensor,
        row_begin: Int32,
        row_count: Int32,
        physical_warpgroup: cutlass.Constexpr[int],
    ):
        """Cross-entropy, in place, of logits rows row_begin .. row_begin + row_count - 1.

        For each row, the loss (the log-sum-exp minus the label's logit, times the approximate
        reciprocal of the global valid-token count; 0 when the label is negative) goes to
        `per_token_loss`, and the row's logits are overwritten with
        the BF16 gradient, (softmax minus one-hot) times that reciprocal.  CTAs take rows in
        grid stride; threads 0-255 (roles 0 and 1) do the arithmetic.
        """

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        for local_row in cutlass.range(
            bidx, row_count, model.PROGRAM_CTAS, unroll=1
        ):
            global_row = row_begin + local_row
            row_base = Int64(global_row) * Int64(model.VOCAB)
            # Each thread reads eight consecutive columns per visit into eight independent
            # online (max, sum of exp) accumulators, so the eight load streams stay
            # independent.  They merge in the fixed order 0 to 7 before the warp and CTA
            # reductions.
            mx0 = Float32(-3.402823466e38)
            mx1 = Float32(-3.402823466e38)
            mx2 = Float32(-3.402823466e38)
            mx3 = Float32(-3.402823466e38)
            mx4 = Float32(-3.402823466e38)
            mx5 = Float32(-3.402823466e38)
            mx6 = Float32(-3.402823466e38)
            mx7 = Float32(-3.402823466e38)
            se0 = Float32(0.0)
            se1 = Float32(0.0)
            se2 = Float32(0.0)
            se3 = Float32(0.0)
            se4 = Float32(0.0)
            se5 = Float32(0.0)
            se6 = Float32(0.0)
            se7 = Float32(0.0)
            # Role 2 skips the accumulation below, but the dynamic `if tidx == 0` merge assigns
            # mx and se in every role's trace, and the DSL needs them bound before that branch.
            mx = mx0
            se = se0
            if const_expr(physical_warpgroup < 2):
                for logit_base in cutlass.range(
                    tidx * Int32(8), model.VOCAB, 2048, unroll=1
                ):
                    pack_base = row_base + Int64(logit_base)
                    value0 = slab_flat[pack_base + Int64(0)].to(Float32)
                    value1 = slab_flat[pack_base + Int64(1)].to(Float32)
                    value2 = slab_flat[pack_base + Int64(2)].to(Float32)
                    value3 = slab_flat[pack_base + Int64(3)].to(Float32)
                    value4 = slab_flat[pack_base + Int64(4)].to(Float32)
                    value5 = slab_flat[pack_base + Int64(5)].to(Float32)
                    value6 = slab_flat[pack_base + Int64(6)].to(Float32)
                    value7 = slab_flat[pack_base + Int64(7)].to(Float32)
                    if value0 > mx0:
                        se0 = se0 * cute.math.exp(
                            mx0 - value0, fastmath=True
                        ) + Float32(1.0)
                        mx0 = value0
                    else:
                        se0 += cute.math.exp(value0 - mx0, fastmath=True)
                    if value1 > mx1:
                        se1 = se1 * cute.math.exp(
                            mx1 - value1, fastmath=True
                        ) + Float32(1.0)
                        mx1 = value1
                    else:
                        se1 += cute.math.exp(value1 - mx1, fastmath=True)
                    if value2 > mx2:
                        se2 = se2 * cute.math.exp(
                            mx2 - value2, fastmath=True
                        ) + Float32(1.0)
                        mx2 = value2
                    else:
                        se2 += cute.math.exp(value2 - mx2, fastmath=True)
                    if value3 > mx3:
                        se3 = se3 * cute.math.exp(
                            mx3 - value3, fastmath=True
                        ) + Float32(1.0)
                        mx3 = value3
                    else:
                        se3 += cute.math.exp(value3 - mx3, fastmath=True)
                    if value4 > mx4:
                        se4 = se4 * cute.math.exp(
                            mx4 - value4, fastmath=True
                        ) + Float32(1.0)
                        mx4 = value4
                    else:
                        se4 += cute.math.exp(value4 - mx4, fastmath=True)
                    if value5 > mx5:
                        se5 = se5 * cute.math.exp(
                            mx5 - value5, fastmath=True
                        ) + Float32(1.0)
                        mx5 = value5
                    else:
                        se5 += cute.math.exp(value5 - mx5, fastmath=True)
                    if value6 > mx6:
                        se6 = se6 * cute.math.exp(
                            mx6 - value6, fastmath=True
                        ) + Float32(1.0)
                        mx6 = value6
                    else:
                        se6 += cute.math.exp(value6 - mx6, fastmath=True)
                    if value7 > mx7:
                        se7 = se7 * cute.math.exp(
                            mx7 - value7, fastmath=True
                        ) + Float32(1.0)
                        mx7 = value7
                    else:
                        se7 += cute.math.exp(value7 - mx7, fastmath=True)

                merge_mx = cutlass.max(mx0, mx1)
                se0 = se0 * cute.math.exp(mx0 - merge_mx, fastmath=True)
                se0 += se1 * cute.math.exp(mx1 - merge_mx, fastmath=True)
                mx0 = merge_mx
                merge_mx = cutlass.max(mx0, mx2)
                se0 = se0 * cute.math.exp(mx0 - merge_mx, fastmath=True)
                se0 += se2 * cute.math.exp(mx2 - merge_mx, fastmath=True)
                mx0 = merge_mx
                merge_mx = cutlass.max(mx0, mx3)
                se0 = se0 * cute.math.exp(mx0 - merge_mx, fastmath=True)
                se0 += se3 * cute.math.exp(mx3 - merge_mx, fastmath=True)
                mx0 = merge_mx
                merge_mx = cutlass.max(mx0, mx4)
                se0 = se0 * cute.math.exp(mx0 - merge_mx, fastmath=True)
                se0 += se4 * cute.math.exp(mx4 - merge_mx, fastmath=True)
                mx0 = merge_mx
                merge_mx = cutlass.max(mx0, mx5)
                se0 = se0 * cute.math.exp(mx0 - merge_mx, fastmath=True)
                se0 += se5 * cute.math.exp(mx5 - merge_mx, fastmath=True)
                mx0 = merge_mx
                merge_mx = cutlass.max(mx0, mx6)
                se0 = se0 * cute.math.exp(mx0 - merge_mx, fastmath=True)
                se0 += se6 * cute.math.exp(mx6 - merge_mx, fastmath=True)
                mx0 = merge_mx
                merge_mx = cutlass.max(mx0, mx7)
                se0 = se0 * cute.math.exp(mx0 - merge_mx, fastmath=True)
                se0 += se7 * cute.math.exp(mx7 - merge_mx, fastmath=True)
                mx = merge_mx
                se = se0
                for offset in cutlass.range_constexpr(5):
                    warp_other_mx = cute.arch.shuffle_sync_bfly(
                        mx, 16 >> offset
                    )
                    warp_other_se = cute.arch.shuffle_sync_bfly(
                        se, 16 >> offset
                    )
                    warp_merged_mx = cutlass.max(mx, warp_other_mx)
                    se = se * cute.math.exp(
                        mx - warp_merged_mx, fastmath=True
                    )
                    se += warp_other_se * cute.math.exp(
                        warp_other_mx - warp_merged_mx, fastmath=True
                    )
                    mx = warp_merged_mx
                lane = tidx % 32
                warp_index = tidx // 32
                if lane == 0:
                    scratch[warp_index * 2] = mx
                    scratch[warp_index * 2 + 1] = se
            cute.arch.sync_threads()
            if tidx == 0:
                mx = scratch[0]
                se = scratch[1]
                for consumer_warp in cutlass.range_constexpr(1, 8):
                    cta_other_mx = scratch[consumer_warp * 2]
                    cta_other_se = scratch[consumer_warp * 2 + 1]
                    cta_merged_mx = cutlass.max(mx, cta_other_mx)
                    se = se * cute.math.exp(
                        mx - cta_merged_mx, fastmath=True
                    )
                    se += cta_other_se * cute.math.exp(
                        cta_other_mx - cta_merged_mx, fastmath=True
                    )
                    mx = cta_merged_mx
                scratch[0] = mx
                scratch[1] = se
            cute.arch.sync_threads()
            row_lse = scratch[0] + cute.math.log(scratch[1], fastmath=True)
            label = labels[global_row]
            inv_valid = cute.arch.rcp_approx(
                global_valid_tokens[0].to(Float32)
            )
            if tidx == 0:
                if label >= 0:
                    target = slab_flat[
                        row_base + Int64(label)
                    ].to(Float32)
                    per_token_loss[global_row] = (
                        row_lse - target
                    ) * inv_valid
                else:
                    per_token_loss[global_row] = Float32(0.0)
            cute.arch.sync_threads()
            if const_expr(physical_warpgroup < 2):
                scale = Float32(0.0)
                if label >= 0:
                    scale = inv_valid
                for gradient_base in cutlass.range(
                    tidx * Int32(8), model.VOCAB, 2048, unroll=1
                ):
                    for vector_lane in cutlass.range_constexpr(8):
                        gradient_column = gradient_base + Int32(vector_lane)
                        probability = cute.math.exp(
                            slab_flat[
                                row_base + Int64(gradient_column)
                            ].to(Float32)
                            - row_lse,
                            fastmath=True,
                        )
                        gradient = scale * probability
                        if gradient_column == label:
                            gradient -= scale
                        slab_flat[
                            row_base + Int64(gradient_column)
                        ] = gradient.to(BFloat16)
            cute.arch.sync_threads()

    def __init__(self):
        """Build the forward member, and the dX and dW members as two separate instances."""
        self.head_forward = make_head_forward_member()
        self.head_dx = make_head_backward_member()
        self.head_dw = make_head_backward_member()

    @cute.jit
    def enter_head_register_split(
        self,
        physical_warpgroup: cutlass.Constexpr[int],
    ):
        """Switch from the decoder's register split (56/224/224) to the head's (232/232/40).

        Role 2 releases down to 40 registers before roles 0 and 1 raise theirs to 232: the
        first named barrier orders the release before the increases, and the second holds every
        role until both are done.
        """

        if const_expr(physical_warpgroup == 2):
            cute.arch.setmaxregister_decrease(40)
            cute.arch.barrier(
                barrier_id=HEAD_REGISTER_SPLIT_BARRIER,
                number_of_threads=model.PROGRAM_THREADS,
            )
            cute.arch.barrier(
                barrier_id=HEAD_REGISTER_SPLIT_BARRIER,
                number_of_threads=model.PROGRAM_THREADS,
            )
        else:
            cute.arch.barrier(
                barrier_id=HEAD_REGISTER_SPLIT_BARRIER,
                number_of_threads=model.PROGRAM_THREADS,
            )
            cute.arch.setmaxregister_increase(232)
            cute.arch.barrier(
                barrier_id=HEAD_REGISTER_SPLIT_BARRIER,
                number_of_threads=model.PROGRAM_THREADS,
            )

    @cute.jit
    def run_head_dx(
        self,
        dx_args,
        scheduler_state: cute.Tensor,
        physical_warpgroup: cutlass.Constexpr[int],
    ):
        """The dX GEMM (dhidden = dlogits @ head weight, FP32) on its dynamic tile scheduler."""
        dx_params = dynamic_scheduler_params(dx_args[16], scheduler_state)
        gemm_body_for_warpgroup(
            self.head_dx,
            *dx_args[:16],
            dx_params,
            DynamicTileScheduler,
            physical_warpgroup,
        )

    @cute.jit
    def restore_decoder_register_split(
        self,
        physical_warpgroup: cutlass.Constexpr[int],
    ):
        """Return from the head's register split (232/232/40) to the decoder's (56/224/224).

        Roles 0 and 1 release 176 and 8 registers per thread before role 2 raises its count from
        40 to 224, with the same two-barrier order as `enter_head_register_split`.
        """

        if const_expr(physical_warpgroup == 0):
            cute.arch.setmaxregister_decrease(56)
            cute.arch.barrier(
                barrier_id=DECODER_REGISTER_SPLIT_BARRIER,
                number_of_threads=model.PROGRAM_THREADS,
            )
            cute.arch.barrier(
                barrier_id=DECODER_REGISTER_SPLIT_BARRIER,
                number_of_threads=model.PROGRAM_THREADS,
            )
        elif const_expr(physical_warpgroup == 1):
            cute.arch.setmaxregister_decrease(224)
            cute.arch.barrier(
                barrier_id=DECODER_REGISTER_SPLIT_BARRIER,
                number_of_threads=model.PROGRAM_THREADS,
            )
            cute.arch.barrier(
                barrier_id=DECODER_REGISTER_SPLIT_BARRIER,
                number_of_threads=model.PROGRAM_THREADS,
            )
        else:
            cute.arch.barrier(
                barrier_id=DECODER_REGISTER_SPLIT_BARRIER,
                number_of_threads=model.PROGRAM_THREADS,
            )
            cute.arch.setmaxregister_increase(224)
            cute.arch.barrier(
                barrier_id=DECODER_REGISTER_SPLIT_BARRIER,
                number_of_threads=model.PROGRAM_THREADS,
            )

    @cute.jit
    def run(
        self,
        fwd_args,
        dx_args,
        dw_args,
        dlogits_flat: cute.Tensor,
        labels: cute.Tensor,
        global_valid_tokens: cute.Tensor,
        per_token_loss: cute.Tensor,
        loss: cute.Tensor,
        task_records: cute.Tensor,
        active_chunks: cute.Tensor,
        generation_slot: cute.Tensor,
        scratch: cute.Tensor,
        fwd_scheduler_state: cute.Tensor,
        dx_scheduler_state: cute.Tensor,
        dw_scheduler_state: cute.Tensor,
        phase_counter: cute.Tensor,
        physical_warpgroup: cutlass.Constexpr[int],
    ):
        """The head phase of one step; the generated step calls it in every role.

        Enters the head's register split; for each of `active_chunks[0]` chunks, publishes the
        chunk and runs the logits GEMM into the dlogits slab (`fwd_scheduler_state` holds one
        scheduler state per chunk); runs `cross_entropy_rows` over all SEQUENCE rows; runs the
        dX and dW GEMMs; sums the loss; and restores the decoder's split.  Grid barriers
        separate the stages.
        """
        self.enter_head_register_split(physical_warpgroup)
        count = active_chunks[0]
        for epoch in cutlass.range(0, count, 1, unroll=1):
            self.publish_chunk_route(
                task_records, generation_slot, epoch, phase_counter
            )
            word_offset = epoch * Int32(SCHEDULER_STATE_WORDS)
            fwd_params = dynamic_scheduler_params_offset(
                fwd_args[16], fwd_scheduler_state, word_offset
            )
            gemm_body_for_warpgroup(
                self.head_forward,
                *fwd_args[:16],
                fwd_params,
                DynamicTileScheduler,
                physical_warpgroup,
            )
            cute.arch.sync_threads()
            cute.arch.sync_threads()
            grid_barrier(phase_counter)

        self.cross_entropy_rows(
            dlogits_flat,
            labels,
            global_valid_tokens,
            per_token_loss,
            scratch,
            Int32(0),
            Int32(model.SEQUENCE),
            physical_warpgroup,
        )
        grid_barrier(phase_counter)

        self.run_head_dx(
            dx_args, dx_scheduler_state, physical_warpgroup
        )
        grid_barrier(phase_counter)

        dw_params = dynamic_scheduler_params(dw_args[16], dw_scheduler_state)
        gemm_body_for_warpgroup(
            self.head_dw,
            *dw_args[:16],
            dw_params,
            DynamicTileScheduler,
            physical_warpgroup,
        )
        cute.arch.sync_threads()
        cute.arch.sync_threads()
        grid_barrier(phase_counter)
        self.sum_token_losses(per_token_loss, loss, scratch, physical_warpgroup)
        self.restore_decoder_register_split(physical_warpgroup)
