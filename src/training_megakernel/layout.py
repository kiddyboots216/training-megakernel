"""Data-only layout contract shared by the host runtime and the program.

This module deliberately contains no CuTe, FlashAttention, Quack, or device-program
imports.  The model width, heads and GEMM families are fixed; every size that
depends on a build's depth D or tokens per GPU S is a function of them, with the
formulas from ``shape.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import torch

from .contract import (
    PHYSICAL_SLOTS,
    PROGRAM_CTAS,
    PROGRAM_THREADS,  # noqa: F401 - public fixed-layout constant
    VOCAB,
    WORLD_SIZE,
)
from .shape import PACKED_SEGMENTS

QUERY_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = 4_096
INTERMEDIATE = 12_288
QUERY_HIDDEN = QUERY_HEADS * HEAD_DIM
KV_HIDDEN = KV_HEADS * HEAD_DIM
QKV_HIDDEN = QUERY_HIDDEN + 2 * KV_HIDDEN
TWO_I = 2 * INTERMEDIATE

MAX_DOCS = PACKED_SEGMENTS
CUM_SLOTS = MAX_DOCS + 1
PAD_TILE = 64
KV_TILE_ROWS = 128
PRE_TILE_ROWS = 64
FWD_TILE_ROWS = 32
BWD_TILE_ROWS = 64
# The q/k-norm gradient partials keep one row per 128-token block.
NORM_PARTIAL_BLOCK_ROWS = 128


@dataclass(frozen=True)
class ControlLayout:
    """Offsets of the packed-FA4 control table; they are part of the executable ABI.

    The table has one generation per layer slot and four entries (packed segments) per
    generation.  The names follow the program's ``CONTROL_*`` constants.
    """

    max_gens: int
    max_entries: int
    max_gen_fwd_tasks: int
    fwd_schedule_elems: int
    seqused: int
    route: int
    ebase: int
    ndocs: int
    kbase: int
    krows: int
    fwdcum: int
    precum: int
    kvcum: int
    padq: int
    padk: int
    dqsh: int
    kvsh: int
    fwdsched: int
    control_elems: int


@cache
def control_layout(depth: int, sequence: int) -> ControlLayout:
    max_gens = PHYSICAL_SLOTS
    max_entries = max_gens * MAX_DOCS
    max_gen_fwd_blocks = sequence // FWD_TILE_ROWS + MAX_DOCS - 1
    max_gen_fwd_tasks = max_gen_fwd_blocks * KV_HEADS
    route = max_entries
    ebase = route + 1
    ndocs = ebase + max_gens
    kbase = ndocs + max_gens
    krows = kbase + max_gens
    fwdcum = krows + max_gens
    precum = fwdcum + max_gens * CUM_SLOTS
    kvcum = precum + max_gens * CUM_SLOTS
    padq = kvcum + max_gens * CUM_SLOTS
    padk = padq + max_entries
    dqsh = padk + max_entries
    kvsh = dqsh + max_entries
    fwdsched = kvsh + max_entries
    control_elems = fwdsched + max_gens * max_gen_fwd_tasks
    return ControlLayout(
        max_gens=max_gens,
        max_entries=max_entries,
        max_gen_fwd_tasks=max_gen_fwd_tasks,
        fwd_schedule_elems=max_gens * max_gen_fwd_tasks,
        seqused=0,
        route=route,
        ebase=ebase,
        ndocs=ndocs,
        kbase=kbase,
        krows=krows,
        fwdcum=fwdcum,
        precum=precum,
        kvcum=kvcum,
        padq=padq,
        padk=padk,
        dqsh=dqsh,
        kvsh=kvsh,
        fwdsched=fwdsched,
        control_elems=control_elems,
    )


ROW_TABLE_FIELDS = (
    "residual_in",
    "input_norm",
    "norm1",
    "rms1_rstd",
    "qkv_raw",
    "v",
    "attention_out",
    "q_norm",
    "k_norm",
    "rotary_cos",
    "rotary_sin",
    "q_rotary",
    "k_rotary",
    "residual_mid",
    "post_attention_norm",
    "norm2",
    "rms2_rstd",
    "gate_up",
    "mlp_intermediate",
    "layer_output",
    "staging_dy",
    "layer_output_dy",
    "mlp_intermediate_dy",
    "gate_up_dy",
    "norm2_dy",
    "residual_mid_dy",
    "post_attn_norm_partial",
    "post_attn_norm_grad",
    "dq_rotary",
    "dk_rotary",
    "dv",
    "dqkv_raw",
    "q_norm_partial",
    "k_norm_partial",
    "q_norm_grad",
    "k_norm_grad",
    "norm1_dy",
    "layer_input_dx",
    "input_norm_partial",
    "input_norm_grad",
)
ROW_SLOT = {name: index for index, name in enumerate(ROW_TABLE_FIELDS)}
ROW_TABLE_SLOTS = len(ROW_TABLE_FIELDS)

GEMM_TILE = 128


@dataclass(frozen=True)
class Family:
    name: str
    m: int
    n: int
    k: int
    a: str
    a_t: bool
    b: str
    d: str
    d_dtype: torch.dtype

    @property
    def tiles(self) -> int:
        return (self.m + GEMM_TILE - 1) // GEMM_TILE * ((self.n + GEMM_TILE - 1) // GEMM_TILE)

    @property
    def flops(self) -> int:
        return 2 * self.m * self.n * self.k

    def tile_split(self) -> tuple[int, int]:
        return divmod(self.tiles, PROGRAM_CTAS)


@cache
def families(sequence: int) -> tuple[Family, ...]:
    """The twelve GEMM families at S tokens per GPU.

    Order is ABI-significant: each family contributes three application arguments.
    """

    return (
        Family(
            "qkv_fwd",
            sequence,
            QKV_HIDDEN,
            HIDDEN,
            "norm1",
            False,
            "qkv_weight_t",
            "qkv_raw",
            torch.bfloat16,
        ),
        Family(
            "o_fwd",
            sequence,
            HIDDEN,
            HIDDEN,
            "attention_out",
            False,
            "o_weight_t",
            "residual_mid",
            torch.bfloat16,
        ),
        Family(
            "gate_up_fwd",
            sequence,
            TWO_I,
            HIDDEN,
            "norm2",
            False,
            "gate_up_weight_t",
            "gate_up",
            torch.bfloat16,
        ),
        Family(
            "down_fwd",
            sequence,
            HIDDEN,
            INTERMEDIATE,
            "mlp_intermediate",
            False,
            "down_weight_t",
            "layer_output",
            torch.bfloat16,
        ),
        Family(
            "down_dx",
            sequence,
            INTERMEDIATE,
            HIDDEN,
            "layer_output_dy",
            False,
            "down_weight",
            "mlp_intermediate_dy",
            torch.bfloat16,
        ),
        Family(
            "down_dw",
            HIDDEN,
            INTERMEDIATE,
            sequence,
            "layer_output_dy",
            True,
            "mlp_intermediate",
            "down_grad",
            torch.float32,
        ),
        Family(
            "gate_up_dx",
            sequence,
            HIDDEN,
            TWO_I,
            "gate_up_dy",
            False,
            "gate_up_weight",
            "norm2_dy",
            torch.bfloat16,
        ),
        Family(
            "gate_up_dw",
            TWO_I,
            HIDDEN,
            sequence,
            "gate_up_dy",
            True,
            "norm2",
            "gate_up_grad",
            torch.float32,
        ),
        Family(
            "o_dx",
            sequence,
            HIDDEN,
            HIDDEN,
            "residual_mid_dy",
            False,
            "o_weight",
            "attention_dout",
            torch.bfloat16,
        ),
        Family(
            "o_dw",
            HIDDEN,
            HIDDEN,
            sequence,
            "residual_mid_dy",
            True,
            "attention_out",
            "o_grad",
            torch.float32,
        ),
        Family(
            "qkv_dx",
            sequence,
            HIDDEN,
            QKV_HIDDEN,
            "dqkv_raw",
            False,
            "qkv_weight",
            "norm1_dy",
            torch.bfloat16,
        ),
        Family(
            "qkv_dw",
            QKV_HIDDEN,
            HIDDEN,
            sequence,
            "dqkv_raw",
            True,
            "norm1",
            "qkv_grad",
            torch.float32,
        ),
    )

NORM_ELEMENTS = 2 * HIDDEN + 2 * HEAD_DIM
FP32_BYTES = 4
BF16_BYTES = 2


@dataclass(frozen=True)
class ReductionSite:
    name: str
    rows: int
    columns: int

    @property
    def elements(self) -> int:
        return self.rows * self.columns

    @property
    def full_surface_bytes(self) -> int:
        return self.elements * FP32_BYTES

    @property
    def owner_elements(self) -> int:
        if self.elements % WORLD_SIZE:
            raise AssertionError(f"{self.name} is not divisible across the ranks")
        return self.elements // WORLD_SIZE

    @property
    def owner_shard_bytes(self) -> int:
        return self.owner_elements * FP32_BYTES

    @property
    def staged_owner_shape(self) -> tuple[int, int]:
        if self.rows % WORLD_SIZE:
            return (1, self.owner_elements)
        return (self.rows // WORLD_SIZE, self.columns)

    def owner_bounds(self, rank: int) -> tuple[int, int]:
        if not 0 <= rank < WORLD_SIZE:
            raise ValueError(rank)
        begin = rank * self.owner_elements
        return (begin, begin + self.owner_elements)


REDUCTION_SITES = (
    ReductionSite("down_dw", HIDDEN, INTERMEDIATE),
    ReductionSite("gate_up_dw", TWO_I, HIDDEN),
    ReductionSite("o_dw", HIDDEN, HIDDEN),
    ReductionSite("qkv_dw", QKV_HIDDEN, HIDDEN),
    ReductionSite("norm", 1, NORM_ELEMENTS),
)


@dataclass(frozen=True)
class WeightPanel:
    name: str
    rows: int
    columns: int

    @property
    def elements(self) -> int:
        return self.rows * self.columns

    @property
    def full_bytes(self) -> int:
        return self.elements * BF16_BYTES

    @property
    def owner_elements(self) -> int:
        if self.elements % WORLD_SIZE:
            raise AssertionError(f"{self.name} is not divisible across the ranks")
        return self.elements // WORLD_SIZE

    @property
    def owner_bytes(self) -> int:
        return self.owner_elements * BF16_BYTES

    def owner_bounds(self, rank: int) -> tuple[int, int]:
        if not 0 <= rank < WORLD_SIZE:
            raise ValueError(rank)
        begin = rank * self.owner_elements
        return (begin, begin + self.owner_elements)


WEIGHT_PANELS = (
    WeightPanel("norm", 1, NORM_ELEMENTS),
    WeightPanel("qkv_dw", QKV_HIDDEN, HIDDEN),
    WeightPanel("o_dw", HIDDEN, HIDDEN),
    WeightPanel("gate_up_dw", TWO_I, HIDDEN),
    WeightPanel("down_dw", HIDDEN, INTERMEDIATE),
)

EMBEDDING_OWNER_ROWS = VOCAB // WORLD_SIZE
# Each GPU's optimizer owns its eighth of every layer's five reduction sites,
# layer by layer, then its embedding and head rows.
LAYER_OWNER_ELEMENTS = sum(site.owner_elements for site in REDUCTION_SITES)


@dataclass(frozen=True)
class OwnerSegment:
    name: str
    begin: int
    end: int
    elements: int


@cache
def owner_segments(depth: int) -> tuple[OwnerSegment, ...]:
    """The optimizer's 5D + 2 owner segments at ``depth`` decoder layers."""

    rows: list[OwnerSegment] = []
    cursor = 0
    for layer in range(depth):
        for site in REDUCTION_SITES:
            begin = cursor
            cursor += site.owner_elements
            rows.append(
                OwnerSegment(
                    name=f"decoder.{layer}.{site.name}",
                    begin=begin,
                    end=cursor,
                    elements=site.owner_elements,
                )
            )
    shell_elements = EMBEDDING_OWNER_ROWS * HIDDEN
    for name in ("embedding", "head"):
        begin = cursor
        cursor += shell_elements
        rows.append(OwnerSegment(name=name, begin=begin, end=cursor, elements=shell_elements))
    return tuple(rows)


@cache
def segment_by_name(depth: int) -> dict[str, OwnerSegment]:
    """Owner segments by name; the returned mapping is shared and must not be mutated."""

    return {row.name: row for row in owner_segments(depth)}


SITE_OFFSET: dict[str, int] = {}
_site_cursor = 0
for _site in REDUCTION_SITES:
    SITE_OFFSET[_site.name] = _site_cursor
    _site_cursor += _site.owner_elements
