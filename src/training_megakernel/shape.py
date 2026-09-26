"""The shape of one megakernel build: depth and tokens per GPU, and what follows from them.

The model's width, heads and vocabulary are Qwen3-8B's, and the kernel runs on eight
H100s; a build chooses only the decoder depth D and the tokens per GPU S.  The
program reads its shape when it is imported (``kernel/build.py`` sets
``TMK_DEPTH`` and ``TMK_SEQUENCE`` first); the host runtime reads the same two
numbers from a bundle's ``launch_abi.json``.  Both derive everything else here.

This module is data only: it imports neither torch nor the CUDA DSL.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

HIDDEN = 4_096
MLP = 12_288
QUERY_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
VOCAB = 151_936
WORLD = 8
VOCAB_SLICE = VOCAB // WORLD

# Depth: two physical layer slots, and a flat optimizer descriptor that must stay
# within a signed 32-bit extent (O(83) exceeds it).
MIN_DEPTH = 2
MAX_DEPTH = 82
# Tokens per GPU, in multiples of 1,024 (the head splits S into eight chunks whose
# rows must be multiples of 128).
MIN_SEQUENCE = 1_024
MAX_SEQUENCE = 32_768
SEQUENCE_QUANTUM = 1_024

SIGNED_INT32_MAX = 2**31 - 1

# Every GPU's token window is four packed segments (documents plus padding); the
# count is compiled into argument shapes, the lengths are set when a run starts.
PACKED_SEGMENTS = 4
# The head processes each GPU's tokens in eight chunks of S/8 rows.
HEAD_CHUNKS = 8

# Optimizer elements each GPU owns: its eighth of every trained parameter.
LAYER_OWNER_ELEMENTS = (
    MLP * HIDDEN  # down
    + 2 * HIDDEN * MLP  # gate, up
    + HIDDEN * HIDDEN  # o
    + HIDDEN * (QUERY_HEADS + 2 * KV_HEADS) * HEAD_DIM  # q, k, v
    + 2 * HIDDEN
    + 2 * HEAD_DIM  # input and post-attention norms, q and k norms
) // WORLD
SHELL_OWNER_ELEMENTS = 2 * VOCAB_SLICE * HIDDEN  # embedding and head rows
OWNER_SEGMENTS_PER_LAYER = 5
# FP32 parameter, gradient and two moments, plus the BF16 parameter.
OPTIMIZER_BYTES_PER_ELEMENT = 18

# Attention checkpoint: output (BF16) plus log-sum-exp (FP32) per token and layer.
ATTENTION_RECORD_BYTES_PER_TOKEN = QUERY_HEADS * HEAD_DIM * 2 + QUERY_HEADS * 4
RESIDUAL_RECORD_BYTES_PER_TOKEN = HIDDEN * 2
# The top layers keep attention checkpoints: nine borrow the not-yet-written part
# of the optimizer-gradient buffer, the rest live in two explicit banks.
CHECKPOINT_WINDOW = 36
CHECKPOINT_PREFIX_LIMIT = 9
RESIDUAL_CHECKPOINT_LIMIT = 12
# Two int64 words after the residual bank hold the ordinary two-slot addresses.
RESIDUAL_ROUTE_TABLE_BYTES = 16
OWNER_GRADIENT_LAYER_BYTES = 4 * LAYER_OWNER_ELEMENTS

# Embedding-route records per (source GPU, owner GPU): below this many tokens per
# GPU the route can never overflow, above it data preparation rejects steps that do.
ROUTE_RECORD_CAPACITY = 6_144


@dataclass(frozen=True)
class CheckpointPlan:
    """Which layers keep attention checkpoints, and where.

    Layers at or above ``prefix_first`` borrow the optimizer-gradient buffer's
    not-yet-written low layers; ``split <= gen < prefix_first`` go in bank 0,
    ``extra_first <= gen < split`` in bank 1.  Layer 0 is never saved.  With no
    prefix, attention checkpoints are off and every boundary equals the depth.
    """

    depth: int
    sequence: int
    prefix: int
    extra: int
    bank0: int
    bank1: int
    prefix_first: int
    extra_first: int
    split: int
    first: int
    owner_layers: int
    residual: int

    @property
    def layer_bytes(self) -> int:
        return self.sequence * ATTENTION_RECORD_BYTES_PER_TOKEN

    @property
    def bank0_bytes(self) -> int:
        return self.bank0 * self.layer_bytes

    @property
    def bank1_bytes(self) -> int:
        return self.bank1 * self.layer_bytes

    @property
    def prefix_bytes(self) -> int:
        return self.prefix * self.layer_bytes

    @property
    def residual_bank_bytes(self) -> int:
        """The BF16 residual records; the route-table words start at this byte."""

        return self.residual * self.sequence * RESIDUAL_RECORD_BYTES_PER_TOKEN

    @property
    def residual_storage_bytes(self) -> int:
        # Two int64 route-table words follow the BF16 residual bank.
        return self.residual_bank_bytes + RESIDUAL_ROUTE_TABLE_BYTES

    # BF16 elements allocated for each store.  An empty attention bank keeps one
    # element, because a tensor view of it must have a nonzero extent.
    @property
    def bank0_elements(self) -> int:
        return max(1, self.bank0_bytes // 2)

    @property
    def bank1_elements(self) -> int:
        return max(1, self.bank1_bytes // 2)

    @property
    def residual_storage_elements(self) -> int:
        return self.residual_storage_bytes // 2


def plan_checkpoints(depth: int, sequence: int) -> CheckpointPlan:
    layer_bytes = sequence * ATTENTION_RECORD_BYTES_PER_TOKEN
    if depth >= CHECKPOINT_WINDOW:
        prefix, extra, bank0, bank1 = 9, 27, 13, 14
        prefix_first = depth - prefix
        extra_first = depth - CHECKPOINT_WINDOW
        owner_layers = 26
    else:
        prefix = extra = 0
        for candidate in range(min(CHECKPOINT_PREFIX_LIMIT, depth - 1), 0, -1):
            candidate_extra = depth - candidate - 1
            if (
                candidate_extra >= 1
                and candidate * layer_bytes < candidate_extra * OWNER_GRADIENT_LAYER_BYTES
            ):
                prefix, extra = candidate, candidate_extra
                break
        prefix_first = depth - prefix
        extra_first = prefix_first - extra
        owner_layers = extra if prefix else 0
        bank1 = (extra + 1) // 2
        bank0 = extra - bank1
    plan = CheckpointPlan(
        depth=depth,
        sequence=sequence,
        prefix=prefix,
        extra=extra,
        bank0=bank0,
        bank1=bank1,
        prefix_first=prefix_first,
        extra_first=extra_first,
        split=extra_first + bank1,
        first=max(1, extra_first),
        owner_layers=owner_layers,
        residual=min(RESIDUAL_CHECKPOINT_LIMIT, depth),
    )
    if plan.prefix and not plan.prefix_bytes < plan.owner_layers * OWNER_GRADIENT_LAYER_BYTES:
        raise ValueError("attention checkpoint prefix does not fit the borrowed gradient buffer")
    if max(plan.bank0_bytes // 2, plan.bank1_bytes // 2, plan.residual_storage_bytes // 2) > (
        SIGNED_INT32_MAX
    ):
        raise ValueError("a checkpoint bank exceeds a signed 32-bit descriptor extent")
    return plan


def optimizer_elements(depth: int) -> int:
    """Optimizer elements each GPU owns at ``depth`` decoder layers: O(D)."""

    return SHELL_OWNER_ELEMENTS + depth * LAYER_OWNER_ELEMENTS


@dataclass(frozen=True)
class Shape:
    depth: int
    sequence: int

    def __post_init__(self) -> None:
        if not MIN_DEPTH <= self.depth <= MAX_DEPTH:
            raise ValueError(f"depth must be in [{MIN_DEPTH}, {MAX_DEPTH}], got {self.depth}")
        if not MIN_SEQUENCE <= self.sequence <= MAX_SEQUENCE or self.sequence % SEQUENCE_QUANTUM:
            raise ValueError(
                f"tokens per GPU must be a multiple of {SEQUENCE_QUANTUM} in "
                f"[{MIN_SEQUENCE}, {MAX_SEQUENCE}], got {self.sequence}"
            )

    @classmethod
    def from_environment(cls) -> Shape:
        """The shape the build selected, defaulting to the released 36 layers and 32,768 tokens."""

        return cls(
            int(os.environ.get("TMK_DEPTH", "36")),
            int(os.environ.get("TMK_SEQUENCE", "32768")),
        )

    @property
    def optimizer_elements(self) -> int:
        return optimizer_elements(self.depth)

    @property
    def owner_segments(self) -> int:
        return OWNER_SEGMENTS_PER_LAYER * self.depth + 2

    @property
    def head_chunk_rows(self) -> int:
        return self.sequence // HEAD_CHUNKS

    @property
    def slot_rows(self) -> int:
        return self.sequence + 384

    @property
    def control_elements(self) -> int:
        """Int32 words of the attention control table (``layout.control_layout``): 79 words of
        per-entry and per-generation metadata for two generations, then the forward schedule of
        (S/32 + 3) * 8 tasks per generation."""

        return 127 + self.sequence // 2

    @property
    def route_records(self) -> int:
        """Embedding-route records per (source, owner) GPU pair."""

        return min(self.sequence, ROUTE_RECORD_CAPACITY)

    @property
    def checkpoint_plan(self) -> CheckpointPlan:
        return plan_checkpoints(self.depth, self.sequence)

    def memory_estimate_bytes(self) -> dict[str, int]:
        """Per-GPU device memory the training state allocates, before CUDA context.

        The terms are the unique storage bytes of the runtime's state, to within 4 bytes at
        every supported shape; the CUDA context, communication buffers and stack add about
        3.2 GiB on top.
        """

        s, d = self.sequence, self.depth
        plan = self.checkpoint_plan
        return {
            "optimizer": OPTIMIZER_BYTES_PER_ELEMENT * self.optimizer_elements,
            "activation_chain": (d + 1) * s * HIDDEN * 2,
            "head_dlogits": s * VOCAB * 2,
            "attention_checkpoints": plan.bank0_bytes + plan.bank1_bytes,
            "residual_checkpoints": plan.residual_storage_bytes,
            # Layer slabs, attention workspace, gradient ring, head per-token buffers, token
            # slots, the control table and the embedding route's records, plus the fixed
            # symmetric arenas and the head weight.
            "slabs_and_arenas": (
                639_774 * s + 131_136 * self.route_records + 400 * d + 6_464_645_036
            ),
        }


def reserved_context_bytes() -> int:
    """Device memory outside the training state: CUDA context, NCCL and stack."""

    return int(3.5 * 2**30)


__all__ = [
    "HEAD_CHUNKS",
    "HIDDEN",
    "LAYER_OWNER_ELEMENTS",
    "MAX_DEPTH",
    "MAX_SEQUENCE",
    "MIN_DEPTH",
    "MIN_SEQUENCE",
    "PACKED_SEGMENTS",
    "SHELL_OWNER_ELEMENTS",
    "VOCAB",
    "WORLD",
    "CheckpointPlan",
    "Shape",
    "optimizer_elements",
    "plan_checkpoints",
    "reserved_context_bytes",
]

