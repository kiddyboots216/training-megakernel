"""Byte layouts of the symmetric-memory arenas the program's collectives run through.

Each arena is one byte buffer per rank: payload slots, then the u32 epoch words that pass the
slots between ranks, READY when a rank's data is in place and CONSUMED or DONE when a rank has
finished reading. The decoder weight ring and decoder gradient ring have one slot per layer
slot; the head weight ring and all-reduce arena have two slots used in turn. The embedding route
arena's layout is in ``route_layout``. The program and the host runtime both allocate and address
the arenas by these offsets.
"""

from __future__ import annotations

from .contract import PHYSICAL_SLOTS as LAYER_SLOTS
from .contract import WORLD_SIZE
from .layout import (
    BF16_BYTES,
    EMBEDDING_OWNER_ROWS,
    FP32_BYTES,
    HIDDEN,
    REDUCTION_SITES,
    VOCAB,
    WEIGHT_PANELS,
)

# Every region of an arena starts on a GUARD_BYTES boundary, at least GUARD_BYTES past the
# end of the region before it.
GUARD_BYTES = 4_096


def _align(offset: int) -> int:
    return (offset + GUARD_BYTES - 1) // GUARD_BYTES * GUARD_BYTES


def _after(offset: int, nbytes: int) -> int:
    """Where the region after the ``nbytes`` bytes at ``offset`` starts."""

    return _align(offset + nbytes + GUARD_BYTES)


def _slot_offsets(region_bytes) -> tuple[tuple[int, ...], int]:
    """Offsets of consecutive regions inside one slot, and the slot's stride."""

    offsets = []
    cursor = 0
    for nbytes in region_bytes:
        offsets.append(cursor)
        cursor = _after(cursor, nbytes)
    return tuple(offsets), cursor


# Each rank owns VOCAB_ROWS_PER_RANK = VOCAB / WORLD rows of the embedding and of the head. The
# head rings count epochs per panel of VOCAB_PANEL_ROWS rows; VOCAB_PANELS rounds up, so the last
# panel can be partial.
assert VOCAB % WORLD_SIZE == 0
VOCAB_ROWS_PER_RANK = EMBEDDING_OWNER_ROWS
VOCAB_PANEL_ROWS = 4_096
VOCAB_PANELS = (VOCAB + VOCAB_PANEL_ROWS - 1) // VOCAB_PANEL_ROWS

# Decoder weight ring: one slot per layer slot holding the five full BF16 weight panels (the
# decoder's weight slabs are views of them), then a READY epoch per slot and panel and a CONSUMED
# epoch per slot.
DECODER_WEIGHT_PANEL_OFFSETS, DECODER_WEIGHT_SLOT_STRIDE = _slot_offsets(
    panel.full_bytes for panel in WEIGHT_PANELS
)
DECODER_WEIGHT_PAYLOAD_OFFSET = GUARD_BYTES
DECODER_WEIGHT_READY_OFFSET = _after(
    DECODER_WEIGHT_PAYLOAD_OFFSET, LAYER_SLOTS * DECODER_WEIGHT_SLOT_STRIDE
)
DECODER_WEIGHT_CONSUMED_OFFSET = _after(
    DECODER_WEIGHT_READY_OFFSET, LAYER_SLOTS * len(WEIGHT_PANELS) * 4
)
DECODER_WEIGHT_ARENA_BYTES = _after(DECODER_WEIGHT_CONSUMED_OFFSET, LAYER_SLOTS * 4)

# Decoder gradient ring: one slot per layer slot holding the five full FP32 weight gradients (the
# decoder's gradient slabs are views of them), then a READY epoch per slot and site and a DONE
# epoch per slot.
DECODER_GRADIENT_SITE_OFFSETS, DECODER_GRADIENT_SLOT_STRIDE = _slot_offsets(
    site.full_surface_bytes for site in REDUCTION_SITES
)
DECODER_GRADIENT_PAYLOAD_OFFSET = GUARD_BYTES
DECODER_GRADIENT_READY_OFFSET = _after(
    DECODER_GRADIENT_PAYLOAD_OFFSET, LAYER_SLOTS * DECODER_GRADIENT_SLOT_STRIDE
)
DECODER_GRADIENT_DONE_OFFSET = _after(
    DECODER_GRADIENT_READY_OFFSET, LAYER_SLOTS * len(REDUCTION_SITES) * 4
)
DECODER_GRADIENT_ARENA_BYTES = _after(DECODER_GRADIENT_DONE_OFFSET, LAYER_SLOTS * 4)

# Head weight ring: two slots of one BF16 vocabulary panel, then a READY and a CONSUMED epoch
# per slot. Each step advances it by one epoch per panel of the embedding and of the head
# (shell_fabric.VOCAB_MATRICES); only the head's panels pass through the slots.
HEAD_WEIGHT_SLOTS = 2
HEAD_WEIGHT_PANEL_BYTES = VOCAB_PANEL_ROWS * HIDDEN * BF16_BYTES
HEAD_WEIGHT_SLOT_STRIDE = _after(0, HEAD_WEIGHT_PANEL_BYTES)
HEAD_WEIGHT_PAYLOAD_OFFSET = GUARD_BYTES
HEAD_WEIGHT_READY_OFFSET = _after(
    HEAD_WEIGHT_PAYLOAD_OFFSET, HEAD_WEIGHT_SLOTS * HEAD_WEIGHT_SLOT_STRIDE
)
HEAD_WEIGHT_CONSUMED_OFFSET = _after(HEAD_WEIGHT_READY_OFFSET, HEAD_WEIGHT_SLOTS * 4)
HEAD_WEIGHT_ARENA_BYTES = _after(HEAD_WEIGHT_CONSUMED_OFFSET, HEAD_WEIGHT_SLOTS * 4)

# Head gradient ring: a READY and a CONSUMED epoch per slot, with no payload. The epochs bracket
# the reduce-scatter of the head weight gradient, which reads that gradient through its own
# multicast address, and advance like the head weight ring's.
HEAD_GRADIENT_SLOTS = 2
HEAD_GRADIENT_READY_OFFSET = GUARD_BYTES
HEAD_GRADIENT_CONSUMED_OFFSET = _after(HEAD_GRADIENT_READY_OFFSET, HEAD_GRADIENT_SLOTS * 4)
HEAD_GRADIENT_ARENA_BYTES = _after(HEAD_GRADIENT_CONSUMED_OFFSET, HEAD_GRADIENT_SLOTS * 4)

# All-reduce arena, for a vector of HIDDEN floats and for one scalar: two slots of local
# values, READY and CONSUMED epochs per slot, the all-rank result and a DONE epoch. The
# optimizer's arena sums the final RMSNorm weight gradient (the vector) and the squared gradient
# norm; the shell's sums the valid-token count.
ALL_REDUCE_SLOTS = 2
ALL_REDUCE_VECTOR_LOCAL_OFFSET = GUARD_BYTES
ALL_REDUCE_VECTOR_READY_OFFSET = _after(
    ALL_REDUCE_VECTOR_LOCAL_OFFSET, ALL_REDUCE_SLOTS * HIDDEN * FP32_BYTES
)
ALL_REDUCE_VECTOR_CONSUMED_OFFSET = _after(ALL_REDUCE_VECTOR_READY_OFFSET, ALL_REDUCE_SLOTS * 4)
ALL_REDUCE_VECTOR_RESULT_OFFSET = _after(ALL_REDUCE_VECTOR_CONSUMED_OFFSET, ALL_REDUCE_SLOTS * 4)
ALL_REDUCE_VECTOR_DONE_OFFSET = _after(ALL_REDUCE_VECTOR_RESULT_OFFSET, HIDDEN * FP32_BYTES)
ALL_REDUCE_SCALAR_LOCAL_OFFSET = _after(ALL_REDUCE_VECTOR_DONE_OFFSET, 4)
ALL_REDUCE_SCALAR_READY_OFFSET = _after(
    ALL_REDUCE_SCALAR_LOCAL_OFFSET, ALL_REDUCE_SLOTS * FP32_BYTES
)
ALL_REDUCE_SCALAR_CONSUMED_OFFSET = _after(ALL_REDUCE_SCALAR_READY_OFFSET, ALL_REDUCE_SLOTS * 4)
ALL_REDUCE_SCALAR_RESULT_OFFSET = _after(ALL_REDUCE_SCALAR_CONSUMED_OFFSET, ALL_REDUCE_SLOTS * 4)
ALL_REDUCE_SCALAR_DONE_OFFSET = _after(ALL_REDUCE_SCALAR_RESULT_OFFSET, FP32_BYTES)
ALL_REDUCE_ARENA_BYTES = _after(ALL_REDUCE_SCALAR_DONE_OFFSET, 4)
