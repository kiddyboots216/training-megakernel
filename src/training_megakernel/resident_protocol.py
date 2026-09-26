"""Words shared by the host runtime and the resident step loop's device code.

The device side is emitted by ``kernel/program/resident_step.py``.  The
host side reads and writes the same words through ``nstep.py``,
``circular_refill_runtime.py``, ``logger.py`` and the DCLM example's checkpoint
publisher.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# nstep_control: one int64 vector per launch.
# ---------------------------------------------------------------------------
CONTROL_STEPS = 0  # steps to run in this launch (host)
CONTROL_MAILBOX = 1  # mapped mailbox address, or 0 (host)
CONTROL_RING_MASK = 2  # telemetry ring slots - 1 (host)
CONTROL_MAILBOX_ENABLED = 3  # nonzero when the mailbox is live (host)
CONTROL_RECORD_WIDTH = 4  # floats per nstep_records row (host)
CONTROL_STEP = 5  # steps completed in this launch (device)
CONTROL_CHECKPOINT = 6  # checkpoint decision, then EXIT_SENTINEL (device)
CONTROL_REFILL_PAGE = 7  # mapped token-refill page address, or 0 (host)
CONTROL_REFILL_SLOT_MASK = 8  # token slots - 1 (host)
CONTROL_REFILL_GENERATION = 9  # refill generation of the launch's first step (host)
CONTROL_REFILL_TIMEOUT_NS = 10  # device wait for a READY slot (host)
CONTROL_WORDS = 11

# Values of nstep_control[CONTROL_CHECKPOINT] during a step.
CHECKPOINT_NONE = 0
CHECKPOINT_WRITE = 1
CHECKPOINT_PARK = 2
# Written to nstep_control[CONTROL_CHECKPOINT] after the last step.
EXIT_SENTINEL = 0x76543210

# nstep_records: one row per step of loss, global norm, clip coefficient, completed step.
RECORD_WIDTH = 4

# ---------------------------------------------------------------------------
# Mapped mailbox: a 64-byte header, then a ring of 64-byte telemetry slots.
# ---------------------------------------------------------------------------
MAILBOX_HEADER_BYTES = 64
MAILBOX_SLOT_BYTES = 64
MAILBOX_LATEST_STEP = 0  # header: last completed step, stored last (device)
SLOT_PUBLISHED_STEP = 0  # slot: completed step, stored after the fields below
SLOT_STEP = 4
SLOT_LOSS = 8
SLOT_GLOBAL_NORM = 12
SLOT_CLIP = 16
SLOT_STEP_END = 20
SLOT_STATUS = 24

# Checkpoint words in the tail of the mailbox header.
CHECKPOINT_HOST_REQUEST_STEP = 24
CHECKPOINT_HOST_ACK_GENERATION = 28
CHECKPOINT_HOST_ABORT = 32
CHECKPOINT_DEVICE_READY_GENERATION = 36
CHECKPOINT_DEVICE_STEP = 40
CHECKPOINT_DEVICE_SURFACE = 44
CHECKPOINT_DEVICE_CHUNK = 48
CHECKPOINT_DEVICE_CHUNK_BYTES = 52
CHECKPOINT_DEVICE_STATUS = 56
CHECKPOINT_DEVICE_DONE_GENERATION = 60
# device_status codes.
CHECKPOINT_STATUS_HOST_ABORT = 1
CHECKPOINT_STATUS_ACK_AHEAD = 2
CHECKPOINT_STATUS_REQUEST_BEHIND = 3
CHECKPOINT_STATUS_CLAIM_AHEAD = 4
# The payload follows the ring, page aligned, as two chunk-sized slots.
CHECKPOINT_PAGE_BYTES = 4_096
CHECKPOINT_CHUNK_BYTES = 256 * 1_024 * 1_024
CHECKPOINT_PAYLOAD_SLOTS = 2

# ---------------------------------------------------------------------------
# Mapped token-refill page.
# ---------------------------------------------------------------------------
REFILL_ABORT = 0  # host: nonzero aborts the wait
REFILL_STATUS = 64  # device: why a wait ended early
REFILL_STARTED = 128  # device: generation whose slot the device is waiting for
REFILL_READY_BASE = 256  # host: per-slot generation that is ready
REFILL_FREE_BASE = 768  # device: per-slot generation that was consumed
REFILL_SLOT_STRIDE = 64
# REFILL_STATUS codes.
REFILL_STATUS_ABORT = 1
REFILL_STATUS_TIMEOUT = 2
REFILL_STATUS_AHEAD = 3


def checkpoint_claim_offset(ring_slots: int) -> int:
    """Mailbox offset of the host's claim word, just past the telemetry ring."""

    return MAILBOX_HEADER_BYTES + ring_slots * MAILBOX_SLOT_BYTES


def checkpoint_payload_offset(ring_slots: int) -> int:
    """Mailbox offset of the first payload slot, page aligned past the claim word."""

    end = checkpoint_claim_offset(ring_slots)
    return (end + CHECKPOINT_PAGE_BYTES - 1) // CHECKPOINT_PAGE_BYTES * CHECKPOINT_PAGE_BYTES
