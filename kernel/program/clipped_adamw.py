"""The optimizer step: all-reduce the final-norm gradient, clip by the global norm, apply AdamW.

``clipped_adamw_step`` runs on every thread of the program after the gradients are reduced, bound
as ``TrainingProgram.run_optimizer_step``. Each rank owns an eighth of every trained parameter
except the final RMSNorm weight, which every rank holds whole; that weight's gradient is
all-reduced here through the multicast address of the all-reduce arena
(``training_megakernel.arenas``). The global gradient norm is the square root of the all-reduced sum of
each rank's squared owner gradient plus an eighth of the squared final-norm gradient, and the
clip coefficient is ``min(max_norm / (norm + 1e-6), 1)``, as in
``torch.nn.utils.clip_grad_norm_``. AdamW then updates the owner slice and the final-norm weight
with decoupled weight decay and bias correction, as ``torch.optim.AdamW`` does, and refreshes
their BF16 copies.

The squared-norm reduction (``thread_squared_sum``, ``store_squared_sum_partials``,
``rank_squared_sum``) uses no atomics; its summation order depends only on the grid.
"""

from __future__ import annotations

import operator

import cutlass
import cutlass.cute as cute
import model
from model import SHAPE
from training_megakernel.arenas import (
    ALL_REDUCE_SCALAR_CONSUMED_OFFSET,
    ALL_REDUCE_SCALAR_DONE_OFFSET,
    ALL_REDUCE_SCALAR_LOCAL_OFFSET,
    ALL_REDUCE_SCALAR_READY_OFFSET,
    ALL_REDUCE_SCALAR_RESULT_OFFSET,
    ALL_REDUCE_SLOTS,
    ALL_REDUCE_VECTOR_CONSUMED_OFFSET,
    ALL_REDUCE_VECTOR_DONE_OFFSET,
    ALL_REDUCE_VECTOR_LOCAL_OFFSET,
    ALL_REDUCE_VECTOR_READY_OFFSET,
    ALL_REDUCE_VECTOR_RESULT_OFFSET,
)
from communication.memory_ops import (
    atomic_cas_gpu_u32,
    fence_sys,
    global_timer_ns,
    multicast_min_acquire_u32,
    multicast_sum_f32,
    store_f32,
    store_relaxed_gpu_u32,
    store_release_sys_u32,
)
from cutlass import BFloat16, Float32, Int32, Int64
from model import PROGRAM_CTAS, PROGRAM_THREADS, PROGRAM_WARPS, WORLD

# The rank's owner slice of every trained parameter except the final RMSNorm weight, as one flat
# vector.
OPTIMIZER_ELEMENTS = SHAPE.optimizer_elements

# The optimizer's integer control words: the all-reduce arena's multicast address, the epoch
# (the step being completed; the step loop sets it before the optimizer runs) and the wait
# timeout.
CONTROL_MULTICAST_BASE = 0
CONTROL_STEP = 1
CONTROL_TIMEOUT_NS = 2
CONTROL_WORDS = 3

HYPERPARAMETER_LEARNING_RATE = 0
HYPERPARAMETER_BETA1 = 1
HYPERPARAMETER_BETA2 = 2
HYPERPARAMETER_EPSILON = 3
HYPERPARAMETER_WEIGHT_DECAY = 4
HYPERPARAMETER_MAX_GRAD_NORM = 5
HYPERPARAMETER_WORDS = 6

# The optimizer status words. The first wait that times out records its kind in word 0
# (STATUS_OK while none has) and its CTA and expected and observed epochs in words 1-3. Word 2 is
# also set to 1 when the epoch does not follow the completed step count, and every step ends by
# writing its epoch to word 3.
STATUS_WORDS = 4
STATUS_OK = 0
STATUS_WAIT_KIND = 0
STATUS_WAIT_CTA = 1
STATUS_WAIT_EXPECTED = 2
STATUS_WAIT_OBSERVED = 3
STATUS_VECTOR_READY_TIMEOUT = 2
STATUS_SCALAR_READY_TIMEOUT = 4
STATUS_STEP_MISMATCH = 2
STATUS_COMPLETED_STEP = 3

# The squared-norm partials vector: one partial per thread, per warp and per CTA, then the
# warp partials of CTA 0's final reduction.
WARP_SIZE = 32
LOCAL_ACCUMULATORS = 4
THREAD_PARTIAL_BASE = 0
THREAD_PARTIAL_ELEMENTS = PROGRAM_CTAS * PROGRAM_THREADS
WARP_PARTIAL_BASE = THREAD_PARTIAL_BASE + THREAD_PARTIAL_ELEMENTS
WARP_PARTIAL_ELEMENTS = PROGRAM_CTAS * PROGRAM_WARPS
CTA_PARTIAL_BASE = WARP_PARTIAL_BASE + WARP_PARTIAL_ELEMENTS
CTA_PARTIAL_ELEMENTS = PROGRAM_CTAS
ROOT_PARTIAL_BASE = CTA_PARTIAL_BASE + CTA_PARTIAL_ELEMENTS
ROOT_PARTIAL_ELEMENTS = PROGRAM_WARPS
NORM_PARTIAL_ELEMENTS = ROOT_PARTIAL_BASE + ROOT_PARTIAL_ELEMENTS


@cute.jit
def thread_squared_sum(
    gradient: cute.Tensor,
    final_gradient: cute.Tensor,
    global_thread: Int32,
    global_threads: Int32,
    owner_elements: cutlass.Constexpr[int],
    final_elements: cutlass.Constexpr[int],
    world_size: cutlass.Constexpr[int],
) -> Float32:
    """This thread's sum of squared gradient elements, in FP32.

    A grid-stride pass over the ``owner_elements`` owner gradient, unrolled into four independent
    FMA chains. The first ``final_elements`` threads also add one final-norm gradient element's
    square divided by ``world_size``: every rank holds the whole all-reduced final-norm gradient,
    so the global sum counts it once.
    """

    square0 = Float32(0.0)
    square1 = Float32(0.0)
    square2 = Float32(0.0)
    square3 = Float32(0.0)
    stride = global_threads * Int32(LOCAL_ACCUMULATORS)
    for index in cutlass.range(
        global_thread,
        Int32(owner_elements),
        stride,
        unroll=1,
    ):
        value0 = gradient[index].to(Float32)
        square0 = cute.math.fma(value0, value0, square0)
        index1 = index + global_threads
        if index1 < Int32(owner_elements):
            value1 = gradient[index1].to(Float32)
            square1 = cute.math.fma(value1, value1, square1)
        index2 = index1 + global_threads
        if index2 < Int32(owner_elements):
            value2 = gradient[index2].to(Float32)
            square2 = cute.math.fma(value2, value2, square2)
        index3 = index2 + global_threads
        if index3 < Int32(owner_elements):
            value3 = gradient[index3].to(Float32)
            square3 = cute.math.fma(value3, value3, square3)
    local_square = (square0 + square1) + (square2 + square3)
    if global_thread < Int32(final_elements):
        value = final_gradient[global_thread].to(Float32)
        local_square = cute.math.fma(
            value,
            value / Float32(float(world_size)),
            local_square,
        )
    return local_square


@cute.jit
def store_squared_sum_partials(
    local_square: Float32,
    norm_partials: cute.Tensor,
    threads_per_cta: cutlass.Constexpr[int],
):
    """Store this thread's partial and its warp's sum, then sum the CTA's warps into its partial."""

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    global_thread = bidx * Int32(threads_per_cta) + tidx
    warp_count = threads_per_cta // WARP_SIZE

    norm_partials[THREAD_PARTIAL_BASE + global_thread] = local_square
    warp_square = cute.arch.warp_reduction(local_square, operator.add)
    if lane == Int32(0):
        scratch = WARP_PARTIAL_BASE + bidx * Int32(PROGRAM_WARPS) + warp
        norm_partials[scratch] = warp_square
    cute.arch.sync_threads()

    cta_input = Float32(0.0)
    if warp == Int32(0) and lane < Int32(warp_count):
        scratch = WARP_PARTIAL_BASE + bidx * Int32(PROGRAM_WARPS) + lane
        cta_input = norm_partials[scratch].to(Float32)
    cta_square = cute.arch.warp_reduction(cta_input, operator.add)
    if tidx == Int32(0):
        norm_partials[CTA_PARTIAL_BASE + bidx] = cta_square


@cute.jit
def rank_squared_sum(
    norm_partials: cute.Tensor,
    threads_per_cta: cutlass.Constexpr[int],
) -> Float32:
    """The rank's squared norm: CTA 0 sums the 132 CTA partials.

    The result is valid in warp 0 of CTA 0; every other thread gets 0. The CTA partials must be
    complete, so a grid barrier separates this from ``store_squared_sum_partials``.
    """

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    warp_count = threads_per_cta // WARP_SIZE

    cta_input = Float32(0.0)
    if bidx == Int32(0) and tidx < Int32(PROGRAM_CTAS):
        cta_input = norm_partials[CTA_PARTIAL_BASE + tidx].to(Float32)
    warp_square = cute.arch.warp_reduction(cta_input, operator.add)
    if bidx == Int32(0) and lane == Int32(0):
        norm_partials[ROOT_PARTIAL_BASE + warp] = warp_square
    cute.arch.sync_threads()

    root_input = Float32(0.0)
    if bidx == Int32(0) and warp == Int32(0) and lane < Int32(warp_count):
        root_input = norm_partials[ROOT_PARTIAL_BASE + lane].to(Float32)
    return cute.arch.warp_reduction(root_input, operator.add)


@cute.jit
def _record_wait_failure(
    status: cute.Tensor,
    kind: Int32,
    cta: Int32,
    expected: Int32,
    observed: Int32,
):
    """Record a timed-out wait: the first failure claims status word 0 and fills words 1-3."""

    won = atomic_cas_gpu_u32(
        status.iterator.toint() + Int64(STATUS_WAIT_KIND * 4),
        Int32(STATUS_OK),
        kind,
    )
    if won == Int32(STATUS_OK):
        _ = store_relaxed_gpu_u32(status.iterator.toint() + Int64(STATUS_WAIT_CTA * 4), cta)
        _ = store_relaxed_gpu_u32(status.iterator.toint() + Int64(STATUS_WAIT_EXPECTED * 4), expected)
        _ = store_relaxed_gpu_u32(status.iterator.toint() + Int64(STATUS_WAIT_OBSERVED * 4), observed)


@cute.jit
def _wait_for_all_ranks(
    address: Int64,
    expected: Int32,
    status: cute.Tensor,
    kind: Int32,
    cta: Int32,
    timeout_ns: Int64,
):
    """Spin until every rank's copy of ``address`` reaches ``expected``, or record a timeout.

    The minimum over ranks is read through the multicast address. After ``timeout_ns`` the
    failure goes to ``status`` and the wait returns anyway.
    """

    observed = Int32(0)
    deadline = global_timer_ns() + timeout_ns
    waiting = Int32(1)
    while waiting == Int32(1):
        observed = multicast_min_acquire_u32(address)
        if observed >= expected:
            waiting = Int32(0)
        elif global_timer_ns() >= deadline:
            _record_wait_failure(status, kind, cta, expected, observed)
            waiting = Int32(0)


@cute.jit
def clipped_adamw_step(
    self,
    all_reduce_arena: cute.Tensor,
    parameter: cute.Tensor,
    gradient: cute.Tensor,
    exp_avg: cute.Tensor,
    exp_avg_sq: cute.Tensor,
    bf16_parameter: cute.Tensor,
    final_parameter: cute.Tensor,
    final_gradient: cute.Tensor,
    final_exp_avg: cute.Tensor,
    final_exp_avg_sq: cute.Tensor,
    final_bf16_parameter: cute.Tensor,
    norm_partials: cute.Tensor,
    outputs: cute.Tensor,
    status: cute.Tensor,
    completed_steps: cute.Tensor,
    phase_counter: cute.Tensor,
    integer_control: cute.Tensor,
    hyperparameters: cute.Tensor,
    threads_per_block: cutlass.Constexpr[int],
):
    """One optimizer step, for the epoch in ``integer_control[CONTROL_STEP]``.

    Uses slot ``epoch % ALL_REDUCE_SLOTS`` of the all-reduce arena. Writes ``outputs`` (the
    rank's squared norm, the global squared norm, the global norm and the clip coefficient), then
    sets ``completed_steps[0]`` and status word 3 to the epoch. Grid barriers separate the phases;
    ``phase_counter`` is the program's grid barrier word. A timed-out wait is recorded in
    ``status`` and the step goes on.
    """

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    global_thread = bidx * Int32(threads_per_block) + tidx
    global_threads = Int32(PROGRAM_CTAS * threads_per_block)
    epoch = Int32(integer_control[CONTROL_STEP])
    slot = epoch % Int32(ALL_REDUCE_SLOTS)

    # Copy this rank's final-RMSNorm gradient into its slot of the all-reduce arena.
    for index in cutlass.range(global_thread, Int32(model.HIDDEN), global_threads, unroll=1):
        _ = store_f32(
            all_reduce_arena.iterator.toint()
            + Int64(ALL_REDUCE_VECTOR_LOCAL_OFFSET)
            + (Int64(slot) * Int64(model.HIDDEN) + Int64(index)) * Int64(4),
            final_gradient[index].to(Float32),
        )
    self.grid_phase_barrier(phase_counter)

    # Thread 0 of CTA 0 checks the epoch against the completed steps, publishes the copied
    # gradient and waits until every rank has published this epoch.
    if bidx == Int32(0) and tidx == Int32(0):
        if Int64(completed_steps[0]) + Int64(1) != Int64(epoch):
            status[STATUS_STEP_MISMATCH] = Int32(1)
        _ = fence_sys()
        _ = store_release_sys_u32(
            all_reduce_arena.iterator.toint() + Int64(ALL_REDUCE_VECTOR_READY_OFFSET) + Int64(slot) * Int64(4),
            epoch,
        )
        _wait_for_all_ranks(
            Int64(integer_control[CONTROL_MULTICAST_BASE]) + Int64(ALL_REDUCE_VECTOR_READY_OFFSET) + Int64(slot) * Int64(4),
            epoch,
            status,
            Int32(STATUS_VECTOR_READY_TIMEOUT),
            Int32(0),
            Int64(integer_control[CONTROL_TIMEOUT_NS]),
        )
    self.grid_phase_barrier(phase_counter)

    # Sum the ranks' copies through the multicast address: every rank ends with the same
    # all-reduced final-norm gradient, whose square thread_squared_sum counts once.
    for index in cutlass.range(global_thread, Int32(model.HIDDEN), global_threads, unroll=1):
        value = multicast_sum_f32(
            Int64(integer_control[CONTROL_MULTICAST_BASE])
            + Int64(ALL_REDUCE_VECTOR_LOCAL_OFFSET)
            + (Int64(slot) * Int64(model.HIDDEN) + Int64(index)) * Int64(4)
        )
        final_gradient[index] = value
        _ = store_f32(
            all_reduce_arena.iterator.toint() + Int64(ALL_REDUCE_VECTOR_RESULT_OFFSET) + Int64(index) * Int64(4),
            value,
        )
    self.grid_phase_barrier(phase_counter)
    if bidx == Int32(0) and tidx == Int32(0):
        _ = fence_sys()
        _ = store_release_sys_u32(
            all_reduce_arena.iterator.toint() + Int64(ALL_REDUCE_VECTOR_CONSUMED_OFFSET) + Int64(slot) * Int64(4),
            epoch,
        )
        _ = store_release_sys_u32(all_reduce_arena.iterator.toint() + Int64(ALL_REDUCE_VECTOR_DONE_OFFSET), epoch)

    # The rank's squared gradient norm: per-thread sums, then warp and CTA sums, then CTA 0's
    # sum of the CTA partials.
    local_square = thread_squared_sum(
        gradient,
        final_gradient,
        global_thread,
        global_threads,
        OPTIMIZER_ELEMENTS,
        model.HIDDEN,
        WORLD,
    )
    store_squared_sum_partials(
        local_square,
        norm_partials,
        threads_per_block,
    )
    self.grid_phase_barrier(phase_counter)

    rank_square = rank_squared_sum(
        norm_partials,
        threads_per_block,
    )

    # All-reduce the squared norm across ranks, then derive the norm and the clip coefficient.
    if bidx == Int32(0) and tidx == Int32(0):
        outputs[0] = rank_square
        _ = store_f32(
            all_reduce_arena.iterator.toint() + Int64(ALL_REDUCE_SCALAR_LOCAL_OFFSET) + Int64(slot) * Int64(4),
            rank_square,
        )
        _ = fence_sys()
        _ = store_release_sys_u32(
            all_reduce_arena.iterator.toint() + Int64(ALL_REDUCE_SCALAR_READY_OFFSET) + Int64(slot) * Int64(4),
            epoch,
        )
        _wait_for_all_ranks(
            Int64(integer_control[CONTROL_MULTICAST_BASE]) + Int64(ALL_REDUCE_SCALAR_READY_OFFSET) + Int64(slot) * Int64(4),
            epoch,
            status,
            Int32(STATUS_SCALAR_READY_TIMEOUT),
            Int32(0),
            Int64(integer_control[CONTROL_TIMEOUT_NS]),
        )
        global_square = multicast_sum_f32(
            Int64(integer_control[CONTROL_MULTICAST_BASE]) + Int64(ALL_REDUCE_SCALAR_LOCAL_OFFSET) + Int64(slot) * Int64(4)
        )
        global_norm = cute.math.sqrt(global_square)
        clip = cute.math.min(
            Float32(hyperparameters[HYPERPARAMETER_MAX_GRAD_NORM]) / (global_norm + Float32(1.0e-6)),
            Float32(1.0),
        )
        outputs[1] = global_square
        outputs[2] = global_norm
        outputs[3] = clip
        _ = store_f32(
            all_reduce_arena.iterator.toint() + Int64(ALL_REDUCE_SCALAR_RESULT_OFFSET),
            global_square,
        )
        _ = fence_sys()
        _ = store_release_sys_u32(
            all_reduce_arena.iterator.toint() + Int64(ALL_REDUCE_SCALAR_CONSUMED_OFFSET) + Int64(slot) * Int64(4),
            epoch,
        )
        _ = store_release_sys_u32(all_reduce_arena.iterator.toint() + Int64(ALL_REDUCE_SCALAR_DONE_OFFSET), epoch)
    self.grid_phase_barrier(phase_counter)

    learning_rate = Float32(hyperparameters[HYPERPARAMETER_LEARNING_RATE])
    beta1 = Float32(hyperparameters[HYPERPARAMETER_BETA1])
    beta2 = Float32(hyperparameters[HYPERPARAMETER_BETA2])
    epsilon = Float32(hyperparameters[HYPERPARAMETER_EPSILON])
    weight_decay = Float32(hyperparameters[HYPERPARAMETER_WEIGHT_DECAY])
    clip = Float32(outputs[3])
    step_count = Float32(epoch)
    bias_correction1 = Float32(1.0) - cute.math.pow(beta1, step_count)
    bias_correction2 = Float32(1.0) - cute.math.pow(beta2, step_count)
    bias_correction2_sqrt = cute.math.sqrt(bias_correction2)
    step_size = learning_rate / bias_correction1

    # AdamW on the owner slice, then on the final-norm weight, which every rank updates
    # identically from the same all-reduced gradient.
    for index in cutlass.range(global_thread, Int32(OPTIMIZER_ELEMENTS), global_threads, unroll=1):
        grad = gradient[index].to(Float32) * clip
        updated = parameter[index].to(Float32)
        if weight_decay != Float32(0.0):
            updated -= learning_rate * weight_decay * updated
        mean = cute.math.fma(
            beta1,
            exp_avg[index].to(Float32),
            cute.math.fma(-beta1, grad, grad),
        )
        grad_square = grad * grad
        variance = cute.math.fma(
            beta2,
            exp_avg_sq[index].to(Float32),
            cute.math.fma(-beta2, grad_square, grad_square),
        )
        denominator = cute.math.sqrt(variance) / bias_correction2_sqrt + epsilon
        updated -= step_size * mean / denominator
        parameter[index] = updated
        exp_avg[index] = mean
        exp_avg_sq[index] = variance
        bf16_parameter[index] = updated.to(BFloat16)

    for index in cutlass.range(global_thread, Int32(model.HIDDEN), global_threads, unroll=1):
        grad = final_gradient[index].to(Float32) * clip
        updated = final_parameter[index].to(Float32)
        if weight_decay != Float32(0.0):
            updated -= learning_rate * weight_decay * updated
        mean = cute.math.fma(
            beta1,
            final_exp_avg[index].to(Float32),
            cute.math.fma(-beta1, grad, grad),
        )
        grad_square = grad * grad
        variance = cute.math.fma(
            beta2,
            final_exp_avg_sq[index].to(Float32),
            cute.math.fma(-beta2, grad_square, grad_square),
        )
        denominator = cute.math.sqrt(variance) / bias_correction2_sqrt + epsilon
        updated -= step_size * mean / denominator
        final_parameter[index] = updated
        final_exp_avg[index] = mean
        final_exp_avg_sq[index] = variance
        final_bf16_parameter[index] = updated.to(BFloat16)
    self.grid_phase_barrier(phase_counter)

    if bidx == Int32(0) and tidx == Int32(0):
        completed_steps[0] = Int64(epoch)
        status[STATUS_COMPLETED_STEP] = epoch
