"""The model, workload and grid dimensions the program is built for.

Qwen3-8B's width, the eight-GPU data-parallel layout, the per-layer weight panels and gradient
reduction sites, and the program's grid are the host runtime's own definitions
(``training_megakernel.layout`` and ``training_megakernel.contract``); the depth and the tokens
per GPU come from the build: ``kernel/build.py`` sets ``TMK_DEPTH`` and ``TMK_SEQUENCE``
before it imports the program, and ``Shape.from_environment`` reads them (without them the shape
is the released 36 layers and 32,768 tokens). Every depth- or length-dependent constant in the
program follows from ``SHAPE`` through ``training_megakernel.shape``, which the host runtime uses
too. Program modules take all of these from here.
"""

from __future__ import annotations

from training_megakernel.contract import PROGRAM_CTAS, PROGRAM_THREADS  # noqa: F401 - re-exported
from training_megakernel.contract import WORLD_SIZE as WORLD  # noqa: F401 - re-exported
from training_megakernel.layout import (  # noqa: F401 - re-exported
    BF16_BYTES,
    FP32_BYTES,
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    KV_HEADS,
    KV_HIDDEN,
    NORM_ELEMENTS,
    QKV_HIDDEN,
    QUERY_HEADS,
    QUERY_HIDDEN,
    REDUCTION_SITES,
    VOCAB,
    WEIGHT_PANELS,
    ReductionSite,
    WeightPanel,
)
# The fused gate and up projections' width, 2 x INTERMEDIATE.
from training_megakernel.layout import TWO_I as GATE_UP  # noqa: F401 - re-exported
from training_megakernel.shape import Shape

SHAPE = Shape.from_environment()

DEPTH = SHAPE.depth
SEQUENCE = SHAPE.sequence

# The packed QKV row is [q | k | v]; V starts after the query and key columns.
V_OFFSET_ELEMENTS = QUERY_HIDDEN + KV_HIDDEN
RMS_EPSILON = 1e-6

# The LM head runs over the sequence in HEAD_CHUNKS chunks of HEAD_CHUNK_ROWS rows.
HEAD_CHUNK_ROWS = SHAPE.head_chunk_rows
HEAD_CHUNKS = SEQUENCE // HEAD_CHUNK_ROWS

PROGRAM_WARPS = PROGRAM_THREADS // 32

# The q/k norm-and-RoPE phases work on NORM_BLOCK_ROWS-token blocks; their norm-weight gradient
# partials keep one row per head and block.
NORM_BLOCK_ROWS = 128
NORM_BLOCKS = SEQUENCE // NORM_BLOCK_ROWS
