"""The step loop's device code, as generated-source lines around one training step.

``training_program._generate_training_kernel`` emits one training step as the body of
``training_step_kernel``; the functions here return the source lines around it that let one
launch run many steps. The returned strings are kernel code. They run on every thread unless a
condition such as ``BLOCK0_THREAD0`` (thread 0 of CTA 0) guards them.

- ``step_prologue`` opens the step: thread indices, the wait for this step's token slot, and
  the epoch the shell's collectives use.
- ``slot_view`` views one token-slot tensor at this step's slot.
- ``pre_optimizer`` sets the optimizer's epoch before ``clipped_adamw_step``.
- ``step_epilogue`` closes the step: the per-step record and telemetry slot, the token slot's
  release, the step counter and the checkpoint decision, the in-launch checkpoint, the reset of
  per-step state, and the exit.

The exit is a branch over the exit sentinel's store, taken when another step follows. ptxas
assembles it as a predicated EXIT just before that store, and after assembly the build rewrites
the EXIT into a branch to the kernel's first instruction (``kernel/tools/patch_step_backedge.py``).
That branch is the step loop: ptxas drops the register-split instructions when the step sits
inside a source-level loop, so the loop cannot be written here.

The step loop's two kernel operands, ``nstep_control`` (Int64 control words) and
``nstep_records`` (one float row per step), form the runtime ABI's ``nstep`` group, and the
generated locals share the prefix. Their word offsets, and every offset in the mapped mailbox and
token-refill page, come from ``training_megakernel.resident_protocol``, which the host runtime
reads too. The words that order host and device accesses use system-scope release stores and
acquire loads.
"""

from __future__ import annotations

from model import SHAPE
from training_megakernel import checkpoint_protocol
from training_megakernel import resident_protocol as P

# The condition for code that only thread 0 of CTA 0 runs.
BLOCK0_THREAD0 = "nstep_bidx == Int32(0) and nstep_tidx == Int32(0)"


def _control(word: int) -> str:
    return f"nstep_control[{word}]"


def _indent(lines: list[str], spaces: int) -> list[str]:
    pad = " " * spaces
    return [pad + line if line.strip() else line for line in lines]


def _lines(text: str) -> list[str]:
    return [line + "\n" for line in text.strip("\n").split("\n")]


def step_prologue() -> list[str]:
    """The step's first lines: thread indices, the token-slot wait, the shell epoch, a barrier.

    When the host maps a token-refill page, thread 0 of CTA 0 stores the refill generation it
    waits for (the launch's first generation plus the steps done) and spins until the host marks
    this step's slot ready with that generation. If the host's generation is ahead, the host
    aborts, or the wait times out, it writes the reason to the page's status word and parks. The
    shell's collectives read their epoch from ``optimizer_integer_control[CONTROL_STEP]``; the
    prologue sets it to this launch's step number, since the shell transport counts epochs from
    1 in each launch.
    """

    c = _control
    return _indent(
        _lines(
            f"""
nstep_bidx, _, _ = cute.arch.block_idx()
nstep_tidx, _, _ = cute.arch.thread_idx()
nstep_global_thread = nstep_bidx * Int32(model.PROGRAM_THREADS) + nstep_tidx
nstep_global_threads = Int32(model.PROGRAM_CTAS * model.PROGRAM_THREADS)
nstep_refill_generation = Int32({c(P.CONTROL_REFILL_GENERATION)}) + Int32({c(P.CONTROL_STEP)})
nstep_refill_slot = Int32({c(P.CONTROL_STEP)}) & Int32({c(P.CONTROL_REFILL_SLOT_MASK)})
if Int64({c(P.CONTROL_REFILL_PAGE)}) != Int64(0):
    if {BLOCK0_THREAD0}:
        nstep_refill_base = Int64({c(P.CONTROL_REFILL_PAGE)})
        _ = store_release_sys_u32(
            nstep_refill_base + Int64({P.REFILL_STARTED}), nstep_refill_generation
        )
        nstep_refill_ready_address = (
            nstep_refill_base
            + Int64({P.REFILL_READY_BASE})
            + Int64(nstep_refill_slot) * Int64({P.REFILL_SLOT_STRIDE})
        )
        nstep_refill_deadline = global_timer_ns() + Int64({c(P.CONTROL_REFILL_TIMEOUT_NS)})
        nstep_refill_waiting = Int32(1)
        nstep_refill_reported = Int32(0)
        while nstep_refill_waiting == Int32(1):
            if nstep_refill_reported == Int32(0):
                nstep_refill_observed = load_acquire_sys_u32(nstep_refill_ready_address)
                nstep_refill_abort = load_acquire_sys_u32(
                    nstep_refill_base + Int64({P.REFILL_ABORT})
                )
                if nstep_refill_observed == nstep_refill_generation:
                    nstep_refill_waiting = Int32(0)
                elif nstep_refill_observed > nstep_refill_generation:
                    nstep_refill_reported = Int32({P.REFILL_STATUS_AHEAD})
                elif nstep_refill_abort != Int32(0):
                    nstep_refill_reported = Int32({P.REFILL_STATUS_ABORT})
                elif global_timer_ns() >= nstep_refill_deadline:
                    nstep_refill_reported = Int32({P.REFILL_STATUS_TIMEOUT})
                if nstep_refill_reported != Int32(0):
                    _ = store_release_sys_u32(
                        nstep_refill_base + Int64({P.REFILL_STATUS}), nstep_refill_reported
                    )
            else:
                # Park: the host sees the status word and stops the launch.
                _ = load_acquire_sys_u32(nstep_refill_base + Int64({P.REFILL_ABORT}))
if {BLOCK0_THREAD0}:
    optimizer_integer_control[clipped_adamw.CONTROL_STEP] = Int64({c(P.CONTROL_STEP)}) + Int64(1)
self.grid_phase_barrier(phase_counter)
"""
        ),
        4,
    )


def slot_view(name: str, source: str, extent: str, indent: int) -> str:
    """A line binding ``name`` to ``source`` viewed at this step's token slot.

    Each slot holds ``extent`` elements; the line is indented by ``indent`` spaces.
    """

    return (
        " " * indent
        + f"{name} = cute.make_tensor({source}.iterator + Int32(nstep_refill_slot) * "
        + f"Int32({extent}), cute.make_layout({extent}))\n"
    )


def pre_optimizer() -> list[str]:
    """Lines that set the optimizer's epoch (completed steps plus one), then a grid barrier."""

    return _indent(
        _lines(
            f"""
if {BLOCK0_THREAD0}:
    optimizer_integer_control[clipped_adamw.CONTROL_STEP] = Int64(optimizer_completed_steps[0]) + Int64(1)
self.grid_phase_barrier(phase_counter)
"""
        ),
        4,
    )


def _record_and_decide() -> str:
    """Thread 0 of CTA 0 records the step and decides whether to write a checkpoint.

    It writes this step's row of ``nstep_records`` (loss, global norm, clip coefficient,
    completed steps) and, when the mailbox is live, its telemetry slot, storing the slot's step
    word and then the header's latest step last, after a fence, so the host never reads a
    partial slot. It releases the token slot when a refill page is mapped, counts the step, and
    sets the checkpoint decision: WRITE when the host's requested step is this one, PARK (with a
    status) when the request is for a step already completed, NONE otherwise.
    """

    c = _control
    return f"""
if {BLOCK0_THREAD0}:
    nstep_completed = Int32(optimizer_completed_steps[0])
    nstep_record_base = Int32({c(P.CONTROL_STEP)}) * Int32({c(P.CONTROL_RECORD_WIDTH)})
    nstep_records[nstep_record_base + Int32(0)] = shell_loss[0]
    nstep_records[nstep_record_base + Int32(1)] = optimizer_outputs[2]
    nstep_records[nstep_record_base + Int32(2)] = optimizer_outputs[3]
    nstep_records[nstep_record_base + Int32(3)] = nstep_completed.to(Float32)
    if Int64({c(P.CONTROL_MAILBOX_ENABLED)}) != Int64(0):
        nstep_mailbox = Int64({c(P.CONTROL_MAILBOX)})
        nstep_slot = nstep_completed & Int32({c(P.CONTROL_RING_MASK)})
        nstep_slot_address = (
            nstep_mailbox
            + Int64({P.MAILBOX_HEADER_BYTES})
            + Int64(nstep_slot) * Int64({P.MAILBOX_SLOT_BYTES})
        )
        _ = store_release_sys_u32(nstep_slot_address + Int64({P.SLOT_STEP}), nstep_completed)
        _ = store_f32(nstep_slot_address + Int64({P.SLOT_LOSS}), shell_loss[0])
        _ = store_f32(nstep_slot_address + Int64({P.SLOT_GLOBAL_NORM}), optimizer_outputs[2])
        _ = store_f32(nstep_slot_address + Int64({P.SLOT_CLIP}), optimizer_outputs[3])
        _ = store_release_sys_u32(
            nstep_slot_address + Int64({P.SLOT_STEP_END}), nstep_completed
        )
        _ = store_release_sys_u32(
            nstep_slot_address + Int64({P.SLOT_STATUS}), optimizer_status[3]
        )
        _ = fence_sys()
        _ = store_release_sys_u32(nstep_slot_address, nstep_completed)
        _ = store_release_sys_u32(nstep_mailbox, nstep_completed)
    if Int64({c(P.CONTROL_REFILL_PAGE)}) != Int64(0) and {BLOCK0_THREAD0}:
        nstep_refill_base = Int64({c(P.CONTROL_REFILL_PAGE)})
        nstep_refill_release_generation = (
            Int32({c(P.CONTROL_REFILL_GENERATION)}) + Int32({c(P.CONTROL_STEP)})
        )
        nstep_refill_release_slot = (
            Int32({c(P.CONTROL_STEP)}) & Int32({c(P.CONTROL_REFILL_SLOT_MASK)})
        )
        nstep_refill_free_address = (
            nstep_refill_base
            + Int64({P.REFILL_FREE_BASE})
            + Int64(nstep_refill_release_slot) * Int64({P.REFILL_SLOT_STRIDE})
        )
        _ = store_release_sys_u32(nstep_refill_free_address, nstep_refill_release_generation)
    {c(P.CONTROL_STEP)} = Int64({c(P.CONTROL_STEP)}) + Int64(1)
    nstep_ckpt_mailbox = Int64({c(P.CONTROL_MAILBOX)})
    if Int64({c(P.CONTROL_MAILBOX_ENABLED)}) != Int64(0) and nstep_ckpt_mailbox != Int64(0):
        nstep_ckpt_request = load_acquire_sys_u32(
            nstep_ckpt_mailbox + Int64({P.CHECKPOINT_HOST_REQUEST_STEP})
        )
        nstep_ckpt_completed = Int32(optimizer_completed_steps[0])
        if nstep_ckpt_request == Int32(0) or nstep_ckpt_request > nstep_ckpt_completed:
            {c(P.CONTROL_CHECKPOINT)} = Int64({P.CHECKPOINT_NONE})
        elif nstep_ckpt_request == nstep_ckpt_completed:
            {c(P.CONTROL_CHECKPOINT)} = Int64({P.CHECKPOINT_WRITE})
        else:
            _ = store_release_sys_u32(
                nstep_ckpt_mailbox + Int64({P.CHECKPOINT_DEVICE_STATUS}),
                Int32({P.CHECKPOINT_STATUS_REQUEST_BEHIND}),
            )
            {c(P.CONTROL_CHECKPOINT)} = Int64({P.CHECKPOINT_PARK})
    else:
        {c(P.CONTROL_CHECKPOINT)} = Int64({P.CHECKPOINT_NONE})
"""


def _park() -> str:
    """Every thread spins while the decision is PARK.

    Nothing clears the decision, so a parked launch stays parked until the host stops it.
    """

    c = _control
    return f"""
if Int32({c(P.CONTROL_CHECKPOINT)}) == Int32({P.CHECKPOINT_PARK}):
    nstep_ckpt_park_mailbox = Int64({c(P.CONTROL_MAILBOX)})
    while Int32({c(P.CONTROL_CHECKPOINT)}) == Int32({P.CHECKPOINT_PARK}):
        _ = load_acquire_sys_u32(nstep_ckpt_park_mailbox + Int64({P.CHECKPOINT_HOST_ABORT}))
"""


def _checkpoint_sources() -> list[tuple[str, int]]:
    """Kernel argument and byte size of each stored tensor, in checkpoint order.

    These are ``checkpoint_protocol.compact_stored_surfaces``: the FP32 parameters, moments and
    hyperparameters. The BF16 parameter copies are left out; they are the FP32 parameters rounded
    to BF16, so a restore rebuilds them.
    """

    return [
        (f"optimizer_{surface.name}", surface.nbytes)
        for surface in checkpoint_protocol.compact_stored_surfaces(SHAPE.optimizer_elements)
    ]


def _checkpoint() -> str:
    """The in-launch checkpoint, run by every thread when the decision is WRITE.

    The stored tensors go to the host in chunks of ``CHECKPOINT_CHUNK_BYTES`` through two mapped
    payload slots, chosen by chunk generation; generations continue from the host's last
    acknowledged one. For each chunk the grid copies it into its slot, then thread 0 of CTA 0
    publishes the chunk's description and generation and waits: it may start the next chunk
    once the host has claimed this one and acknowledged the one before, whose slot the next
    chunk reuses; after the last chunk it waits for the host's acknowledgement. A claim or an
    acknowledgement ahead of the device, or a host abort before the last chunk, makes it record
    a status and park.
    """

    c = _control
    sources = _checkpoint_sources()
    selector = []
    for index, (argument, nbytes) in enumerate(sources):
        keyword = "if" if index == 0 else "elif"
        selector.append(
            f"        {keyword} nstep_ckpt_surface == Int32({index}):\n"
            f"            nstep_ckpt_source_address = {argument}.iterator.toint()\n"
            f"            nstep_ckpt_surface_bytes = Int64({nbytes})"
        )
    selector_text = "\n".join(selector)
    last = len(sources) - 1
    chunk = P.CHECKPOINT_CHUNK_BYTES
    page = P.CHECKPOINT_PAGE_BYTES
    return f"""
if Int32({c(P.CONTROL_CHECKPOINT)}) == Int32({P.CHECKPOINT_WRITE}):
    nstep_ckpt_mailbox = Int64({c(P.CONTROL_MAILBOX)})
    nstep_ckpt_ring_slots = Int64({c(P.CONTROL_RING_MASK)}) + Int64(1)
    nstep_ckpt_claim_address = (
        nstep_ckpt_mailbox
        + Int64({P.MAILBOX_HEADER_BYTES})
        + nstep_ckpt_ring_slots * Int64({P.MAILBOX_SLOT_BYTES})
    )
    nstep_ckpt_payload_offset = (
        (
            Int64({P.MAILBOX_HEADER_BYTES})
            + nstep_ckpt_ring_slots * Int64({P.MAILBOX_SLOT_BYTES})
            + Int64({page - 1})
        )
        // Int64({page})
        * Int64({page})
    )
    nstep_ckpt_generation = load_acquire_sys_u32(
        nstep_ckpt_mailbox + Int64({P.CHECKPOINT_HOST_ACK_GENERATION})
    ) + Int32(1)
    nstep_ckpt_surface = Int32(0)
    while nstep_ckpt_surface < Int32({len(sources)}):
        nstep_ckpt_source_address = Int64(0)
        nstep_ckpt_surface_bytes = Int64(0)
{selector_text}
        nstep_ckpt_source_offset = Int64(0)
        nstep_ckpt_chunk_index = Int32(0)
        while nstep_ckpt_source_offset < nstep_ckpt_surface_bytes:
            nstep_ckpt_remaining = nstep_ckpt_surface_bytes - nstep_ckpt_source_offset
            nstep_ckpt_chunk_bytes = Int64({chunk})
            if nstep_ckpt_remaining < nstep_ckpt_chunk_bytes:
                nstep_ckpt_chunk_bytes = nstep_ckpt_remaining
            nstep_ckpt_payload_slot = (
                (nstep_ckpt_generation - Int32(1)) % Int32({P.CHECKPOINT_PAYLOAD_SLOTS})
            )
            nstep_ckpt_payload_address = (
                nstep_ckpt_mailbox
                + nstep_ckpt_payload_offset
                + Int64(nstep_ckpt_payload_slot) * Int64({chunk})
            )
            # 16-byte copies across the grid; stored tensors are whole 4-byte words,
            # so at most three scalar words remain for thread 0.
            nstep_ckpt_copy_offset = Int64(nstep_global_thread) * Int64(16)
            while nstep_ckpt_copy_offset + Int64(16) <= nstep_ckpt_chunk_bytes:
                _ = copy_b128(
                    nstep_ckpt_source_address + nstep_ckpt_source_offset + nstep_ckpt_copy_offset,
                    nstep_ckpt_payload_address + nstep_ckpt_copy_offset,
                )
                nstep_ckpt_copy_offset = (
                    nstep_ckpt_copy_offset + Int64(nstep_global_threads) * Int64(16)
                )
            nstep_ckpt_tail_offset = nstep_ckpt_chunk_bytes - nstep_ckpt_chunk_bytes % Int64(16)
            if nstep_global_thread == Int32(0):
                if nstep_ckpt_tail_offset < nstep_ckpt_chunk_bytes:
                    _ = store_f32(
                        nstep_ckpt_payload_address + nstep_ckpt_tail_offset,
                        load_f32(
                            nstep_ckpt_source_address
                            + nstep_ckpt_source_offset
                            + nstep_ckpt_tail_offset
                        ),
                    )
                if nstep_ckpt_tail_offset + Int64(4) < nstep_ckpt_chunk_bytes:
                    _ = store_f32(
                        nstep_ckpt_payload_address + nstep_ckpt_tail_offset + Int64(4),
                        load_f32(
                            nstep_ckpt_source_address
                            + nstep_ckpt_source_offset
                            + nstep_ckpt_tail_offset
                            + Int64(4)
                        ),
                    )
                if nstep_ckpt_tail_offset + Int64(8) < nstep_ckpt_chunk_bytes:
                    _ = store_f32(
                        nstep_ckpt_payload_address + nstep_ckpt_tail_offset + Int64(8),
                        load_f32(
                            nstep_ckpt_source_address
                            + nstep_ckpt_source_offset
                            + nstep_ckpt_tail_offset
                            + Int64(8)
                        ),
                    )
            self.grid_phase_barrier(phase_counter)
            if {BLOCK0_THREAD0}:
                _ = store_release_sys_u32(
                    nstep_ckpt_mailbox + Int64({P.CHECKPOINT_DEVICE_STEP}),
                    Int32(optimizer_completed_steps[0]),
                )
                _ = store_release_sys_u32(
                    nstep_ckpt_mailbox + Int64({P.CHECKPOINT_DEVICE_SURFACE}), nstep_ckpt_surface
                )
                _ = store_release_sys_u32(
                    nstep_ckpt_mailbox + Int64({P.CHECKPOINT_DEVICE_CHUNK}), nstep_ckpt_chunk_index
                )
                _ = store_release_sys_u32(
                    nstep_ckpt_mailbox + Int64({P.CHECKPOINT_DEVICE_CHUNK_BYTES}),
                    Int32(nstep_ckpt_chunk_bytes),
                )
                nstep_ckpt_final = Int32(
                    nstep_ckpt_surface == Int32({last})
                    and nstep_ckpt_source_offset + nstep_ckpt_chunk_bytes
                    == nstep_ckpt_surface_bytes
                )
                if nstep_ckpt_final != Int32(0):
                    _ = store_release_sys_u32(
                        nstep_ckpt_mailbox + Int64({P.CHECKPOINT_DEVICE_DONE_GENERATION}),
                        nstep_ckpt_generation,
                    )
                _ = fence_sys()
                _ = store_release_sys_u32(
                    nstep_ckpt_mailbox + Int64({P.CHECKPOINT_DEVICE_READY_GENERATION}),
                    nstep_ckpt_generation,
                )
                # Two payload slots: the next chunk may start once the host has claimed
                # this one; the last chunk waits for the host's acknowledgement.
                nstep_ckpt_double_waiting = Int32(1)
                while nstep_ckpt_double_waiting == Int32(1):
                    nstep_ckpt_observed_claim = load_acquire_sys_u32(nstep_ckpt_claim_address)
                    nstep_ckpt_observed_ack = load_acquire_sys_u32(
                        nstep_ckpt_mailbox + Int64({P.CHECKPOINT_HOST_ACK_GENERATION})
                    )
                    nstep_ckpt_abort = load_acquire_sys_u32(
                        nstep_ckpt_mailbox + Int64({P.CHECKPOINT_HOST_ABORT})
                    )
                    if nstep_ckpt_final != Int32(0):
                        if nstep_ckpt_observed_ack == nstep_ckpt_generation:
                            nstep_ckpt_double_waiting = Int32(0)
                            {c(P.CONTROL_CHECKPOINT)} = Int64({P.CHECKPOINT_NONE})
                        elif nstep_ckpt_observed_ack > nstep_ckpt_generation:
                            _ = store_release_sys_u32(
                                nstep_ckpt_mailbox + Int64({P.CHECKPOINT_DEVICE_STATUS}),
                                Int32({P.CHECKPOINT_STATUS_ACK_AHEAD}),
                            )
                            {c(P.CONTROL_CHECKPOINT)} = Int64({P.CHECKPOINT_PARK})
                            nstep_ckpt_double_waiting = Int32(0)
                    elif (
                        nstep_ckpt_observed_claim == nstep_ckpt_generation
                        and nstep_ckpt_observed_ack >= nstep_ckpt_generation - Int32(1)
                    ):
                        nstep_ckpt_double_waiting = Int32(0)
                    elif (
                        nstep_ckpt_observed_claim > nstep_ckpt_generation
                        or nstep_ckpt_observed_ack > nstep_ckpt_generation
                    ):
                        _ = store_release_sys_u32(
                            nstep_ckpt_mailbox + Int64({P.CHECKPOINT_DEVICE_STATUS}),
                            Int32({P.CHECKPOINT_STATUS_CLAIM_AHEAD}),
                        )
                        {c(P.CONTROL_CHECKPOINT)} = Int64({P.CHECKPOINT_PARK})
                        nstep_ckpt_double_waiting = Int32(0)
                    elif nstep_ckpt_abort != Int32(0):
                        _ = store_release_sys_u32(
                            nstep_ckpt_mailbox + Int64({P.CHECKPOINT_DEVICE_STATUS}),
                            Int32({P.CHECKPOINT_STATUS_HOST_ABORT}),
                        )
                        {c(P.CONTROL_CHECKPOINT)} = Int64({P.CHECKPOINT_PARK})
                        nstep_ckpt_double_waiting = Int32(0)
            self.grid_phase_barrier(phase_counter)
            if Int32({c(P.CONTROL_CHECKPOINT)}) == Int32({P.CHECKPOINT_PARK}):
                while Int32({c(P.CONTROL_CHECKPOINT)}) == Int32({P.CHECKPOINT_PARK}):
                    _ = load_acquire_sys_u32(nstep_ckpt_mailbox + Int64({P.CHECKPOINT_HOST_ABORT}))
            nstep_ckpt_source_offset = nstep_ckpt_source_offset + nstep_ckpt_chunk_bytes
            nstep_ckpt_chunk_index = nstep_ckpt_chunk_index + Int32(1)
            nstep_ckpt_generation = nstep_ckpt_generation + Int32(1)
        nstep_ckpt_surface = nstep_ckpt_surface + Int32(1)
"""


def _scheduler_reset(state: str) -> str:
    """The line that returns work-sharing scheduler state ``state``'s tile queue to 0."""

    return f"""
        {state}[0] = Int32(0)"""


def _next_step(scheduler_states: tuple[str, ...]) -> str:
    """The step's last lines: reset per-step state when another step follows, then the exit.

    Before another step, thread 0 of CTA 0 sets the optimizer's epoch for it and returns every
    work-sharing tile queue in ``scheduler_states``, and the head's per-chunk forward queues, to
    0; after a grid barrier, every warp returns to the 168 registers per thread it had at launch.
    Then ``step_exit_branch`` skips the exit sentinel's store when another step follows.
    """

    c = _control
    resets = "".join(_scheduler_reset(state) for state in scheduler_states)
    return f"""
nstep_more = Int32(Int32({c(P.CONTROL_STEP)}) < Int32({c(P.CONTROL_STEPS)}))
if nstep_more != Int32(0):
    if {BLOCK0_THREAD0}:
        optimizer_integer_control[clipped_adamw.CONTROL_STEP] = Int64(optimizer_completed_steps[0]) + Int64(1){resets}
        for nstep_reset_index in cutlass.range_constexpr(model.HEAD_CHUNKS):
            shell_head_fwd_scheduler_state[nstep_reset_index] = Int32(0)
    self.grid_phase_barrier(phase_counter)
    # Every warp returns to the launch's uniform 168 registers before re-entry.
    nstep_tail_warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if nstep_tail_warp_idx < 4:
        cute.arch.barrier(barrier_id=7, number_of_threads=model.PROGRAM_THREADS)
        cute.arch.setmaxregister_increase(168)
        cute.arch.barrier(barrier_id=7, number_of_threads=model.PROGRAM_THREADS)
    else:
        cute.arch.setmaxregister_decrease(168)
        cute.arch.barrier(barrier_id=7, number_of_threads=model.PROGRAM_THREADS)
        cute.arch.barrier(barrier_id=7, number_of_threads=model.PROGRAM_THREADS)
step_exit_branch(nstep_more)
{c(P.CONTROL_CHECKPOINT)} = Int64({P.EXIT_SENTINEL})
step_exit_label()
"""


def step_epilogue(scheduler_states: tuple[str, ...]) -> list[str]:
    """The lines after the optimizer step: record and decide, park, checkpoint, reset, exit."""

    text = (
        "\nself.grid_phase_barrier(phase_counter)"
        + _record_and_decide()
        + "self.grid_phase_barrier(phase_counter)"
        + _park()
        + _checkpoint()
        + _next_step(scheduler_states)
    )
    return _indent(_lines(text), 4)
