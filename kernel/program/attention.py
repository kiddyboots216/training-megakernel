"""The program's attention phases, their tile schedulers and control table, and the host plan.

`run_attention_forward` runs FA4's forward member (see `attention_members`) with its LSE stored
generation-major. `run_attention_backward` runs the backward in three phases separated by grid
barriers: `clear_dkv_accumulators` with `attention_backward_preprocess`, FA4's backward kernel,
then `attention_backward_postprocess`. `AttentionForwardScheduler` and
`AttentionBackwardScheduler` give FA4's kernels their tiles, and `SlotSeqlenInfoQK` places the
backward kernel's workspace rows. On the host, `AttentionPlan` describes the packed batch and
`AttentionTensors` allocates the tensors and stages the control table.

The batch is packed: a generation of the plan holds up to `MAX_DOCUMENTS` causal documents that
fill `SEQUENCE` tokens, and each document is one FA4 batch entry. The program stages one
generation per layer slot and, before each layer's attention, writes the slot (`route`) into the
control table's `CONTROL_ROUTE` word; the device code here calls that index `gen`. The control
table is one Int32 tensor: FA4 reads its first `CONTROL_ENTRIES` words as `seqused`, and the
schedulers and phases read the block tables, the slot table and the forward schedule after them.

`AttentionTileScheduler.__new_from_mlir_values__` is adapted from
`StaticPersistentTileScheduler.__new_from_mlir_values__` in flash_attn/cute/tile_scheduler.py
(FlashAttention 4, commit 890f238).  Copyright (c) 2025, Tri Dao, Siyu Wang, Shengbin Di, Yuxi
Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li,
Wei Lin; FlashAttention is distributed under the BSD 3-Clause License (see
THIRD_PARTY_NOTICES.md).  It constructs `type(self)` instead of the base class, so the DSL
rebuilds the subclass.  The schedulers' other methods implement FA4's tile-scheduler interface
over the control table.
"""

# ruff: noqa: I001

from __future__ import annotations

import math
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.runtime import from_dlpack
from flash_attn.cute.named_barrier import NamedBarrierBwd  # noqa: TID253
from flash_attn.cute.seqlen_info import SeqlenInfoQK  # noqa: TID253
from flash_attn.cute.tile_scheduler import (  # noqa: TID253
    ParamsBase,
    SchedulingMode,
    StaticPersistentTileScheduler,
    TileSchedulerArguments,
    WorkTileInfo,
)
from flash_attn.cute.utils import AuxData  # noqa: TID253
from training_megakernel.contract import PHYSICAL_SLOTS
from training_megakernel.schedule import build_lpt_schedule

from anchors import anchor_index, anchored_tensor
from attention_members import (
    BACKWARD_TILE_M,
    BACKWARD_TILE_N,
    DKV_POSTPROCESS_TILE_M,
    FORWARD_TILE_M,
    assume_pointer_aligned,
    run_attention_forward_member,
)
from fa4_postprocess import (
    postprocess_task_body,
    postprocess_task_body_direct_load,
    postprocess_task_body_two_outputs,
)
from model import (
    HEAD_DIM,
    KV_HEADS,
    PROGRAM_CTAS,
    PROGRAM_THREADS,
    QKV_HIDDEN,
    QUERY_HEADS,
    SEQUENCE,
    V_OFFSET_ELEMENTS,
)


# The block tables, padded offsets and slot table below assume 64 x 64 backward tiles.
assert BACKWARD_TILE_M == BACKWARD_TILE_N == 64, (BACKWARD_TILE_M, BACKWARD_TILE_N)

# The forward's query block in tokens. `pack_gqa` folds the query heads of one KV head into
# `tile_m`, so a block covers `tile_m / (QUERY_HEADS / KV_HEADS)` tokens.
FWD_BLOCK_TOKENS = FORWARD_TILE_M // (QUERY_HEADS // KV_HEADS)   # 32
# The workspace's padding granule, and the block of the backward kernel's K tiles, the
# preprocess and the dQ postprocess.
BWD_BLOCK_ROWS = BACKWARD_TILE_M                       # 64

# Blocks in a full-length (SEQUENCE-token) document.
BWD_BLOCKS_PER_SEQUENCE = SEQUENCE // BWD_BLOCK_ROWS
FWD_BLOCKS_PER_SEQUENCE = SEQUENCE // FWD_BLOCK_TOKENS
DKV_BLOCKS_PER_SEQUENCE = SEQUENCE // DKV_POSTPROCESS_TILE_M

assert BWD_BLOCKS_PER_SEQUENCE > 0 and FWD_BLOCKS_PER_SEQUENCE > 0


# The control table holds one generation per layer slot, of up to MAX_DOCUMENTS documents. A
# batch entry is one (generation, document) pair. Each document after the first can add one
# partial block, so a generation has at most MAX_DOCUMENTS - 1 more blocks than one full-length
# document.

CONTROL_GENERATIONS = PHYSICAL_SLOTS               # one generation per layer slot
MAX_DOCUMENTS = 4                                  # documents per generation
CONTROL_ENTRIES = CONTROL_GENERATIONS * MAX_DOCUMENTS   # batch entries
CUMULATIVE_EDGES = MAX_DOCUMENTS + 1

MAX_FWD_BLOCKS_PER_GENERATION = FWD_BLOCKS_PER_SEQUENCE + MAX_DOCUMENTS - 1
MAX_BWD_BLOCKS_PER_GENERATION = BWD_BLOCKS_PER_SEQUENCE + MAX_DOCUMENTS - 1
MAX_DKV_BLOCKS_PER_GENERATION = DKV_BLOCKS_PER_SEQUENCE + MAX_DOCUMENTS - 1

# The forward schedule, the control table's last region: one stride of
# MAX_FWD_TASKS_PER_GENERATION words per generation.
MAX_FWD_TASKS_PER_GENERATION = MAX_FWD_BLOCKS_PER_GENERATION * KV_HEADS
FWD_SCHEDULE_WORDS = CONTROL_GENERATIONS * MAX_FWD_TASKS_PER_GENERATION

# A slot holds the padded rows of one generation, which form one contiguous span. Relative to
# the generation's first document, FA4's padded offset puts document i at its token offset
# rounded down to 64 rows plus 64 * i rows, and the dK/dV postprocess reads the last document in
# whole 128-row tiles. A generation of MAX_DOCUMENTS documents therefore spans at most
# SEQUENCE + MAX_DOCUMENTS * BWD_BLOCK_ROWS rows (the second assert below). _MAX_GENERATION_SPAN
# is a looser bound; rounding it up to a multiple of DKV_POSTPROCESS_TILE_M makes every slot base
# a multiple of both tile sizes, which the `cute.assume(divby=...)` on slot-table reads relies
# on. `AttentionPlan.check_slot_layout` checks each plan's spans on the host.
_MAX_GENERATION_SPAN = (
    SEQUENCE + (MAX_DOCUMENTS - 1) * (BWD_BLOCK_ROWS - 1) + (DKV_POSTPROCESS_TILE_M - 1)
)
SLOT_ROWS = -(-_MAX_GENERATION_SPAN // DKV_POSTPROCESS_TILE_M) * DKV_POSTPROCESS_TILE_M  # S + 384
assert SLOT_ROWS % BWD_BLOCK_ROWS == 0 and SLOT_ROWS % DKV_POSTPROCESS_TILE_M == 0
assert SLOT_ROWS >= SEQUENCE + MAX_DOCUMENTS * BWD_BLOCK_ROWS

# The control table: one Int32 tensor, bound to FA4 as `mSeqUsedQ` and `mSeqUsedK` and read by
# the schedulers and phases. FA4 indexes `seqused` by batch entry, so the document lengths
# occupy [0, CONTROL_ENTRIES) and everything else follows. In order: the document lengths; the
# route word (the current generation); per generation, its first entry, its document count and
# the base and rows of its dK/dV span; per generation, the forward, backward (64-row) and dK/dV
# postprocess (128-row) cumulative block tables of CUMULATIVE_EDGES words each; per entry, the Q
# and K padded bases (the slot table) and the dQ and dK/dV postprocess shifts; then the forward
# schedule.

CONTROL_SEQUSED = 0
CONTROL_ROUTE = CONTROL_ENTRIES
CONTROL_ENTRY_BASE = CONTROL_ROUTE + 1
CONTROL_DOCUMENT_COUNT = CONTROL_ENTRY_BASE + CONTROL_GENERATIONS
CONTROL_DKV_SPAN_BASE = CONTROL_DOCUMENT_COUNT + CONTROL_GENERATIONS
CONTROL_DKV_SPAN_ROWS = CONTROL_DKV_SPAN_BASE + CONTROL_GENERATIONS
CONTROL_FWD_CUMULATIVE_BLOCKS = CONTROL_DKV_SPAN_ROWS + CONTROL_GENERATIONS
CONTROL_BWD_CUMULATIVE_BLOCKS = (
    CONTROL_FWD_CUMULATIVE_BLOCKS + CONTROL_GENERATIONS * CUMULATIVE_EDGES
)
CONTROL_DKV_CUMULATIVE_BLOCKS = (
    CONTROL_BWD_CUMULATIVE_BLOCKS + CONTROL_GENERATIONS * CUMULATIVE_EDGES
)
CONTROL_Q_PADDED_BASE = CONTROL_DKV_CUMULATIVE_BLOCKS + CONTROL_GENERATIONS * CUMULATIVE_EDGES
CONTROL_K_PADDED_BASE = CONTROL_Q_PADDED_BASE + CONTROL_ENTRIES
CONTROL_DQ_POSTPROCESS_SHIFT = CONTROL_K_PADDED_BASE + CONTROL_ENTRIES
CONTROL_DKV_POSTPROCESS_SHIFT = CONTROL_DQ_POSTPROCESS_SHIFT + CONTROL_ENTRIES
CONTROL_FWD_SCHEDULE = CONTROL_DKV_POSTPROCESS_SHIFT + CONTROL_ENTRIES
CONTROL_WORDS = CONTROL_FWD_SCHEDULE + FWD_SCHEDULE_WORDS


def block_count(length: int, rows: int) -> int:
    return math.ceil(length / rows)


def padded_offset(cu: list[int], entry: int, tile: int = BWD_BLOCK_ROWS) -> int:
    """FA4's padded offset of batch entry `entry`, as `SeqlenInfoQK` computes it at `tile`."""

    return (cu[entry] + entry * tile) // tile * tile


@dataclass(frozen=True)
class AttentionPlan:
    """The packed batch: each generation's document lengths, and the workspace slot count.

    Each generation's documents must sum to SEQUENCE. Generation `g` uses workspace slot
    `g % slot_count`. The methods derive what `AttentionTensors.stage_control_plane` writes: the
    packed `cu_seqlens`, the cumulative block tables and the slot table.
    """

    name: str
    document_lengths: list[list[int]]
    slot_count: int = 2          # physical workspace slots

    def __post_init__(self) -> None:
        if len(self.document_lengths) > CONTROL_GENERATIONS:
            raise ValueError("generations exceed the compiled control plane")
        for lengths in self.document_lengths:
            if not lengths or len(lengths) > MAX_DOCUMENTS:
                raise ValueError("documents per generation exceed the control plane")
            if sum(lengths) != SEQUENCE:
                raise ValueError("a generation's documents must pack exactly SEQUENCE tokens")
        if not (1 <= self.slot_count <= len(self.document_lengths)):
            raise ValueError("slot_count must be in [1, generations]")

    # The packed batch: one entry per (generation, document), in generation order.

    @property
    def generations(self) -> int:
        return len(self.document_lengths)

    @property
    def entry_lengths(self) -> list[int]:
        return [length for lengths in self.document_lengths for length in lengths]

    @property
    def entry_base(self) -> list[int]:
        bases, total = [], 0
        for lengths in self.document_lengths:
            bases.append(total)
            total += len(lengths)
        return bases

    @property
    def generation_of_entry(self) -> list[int]:
        table: list[int] = []
        for generation, lengths in enumerate(self.document_lengths):
            table.extend([generation] * len(lengths))
        return table

    @property
    def cu_seqlens(self) -> list[int]:
        out, total = [0], 0
        for length in self.entry_lengths:
            total += length
            out.append(total)
        return out

    @property
    def entry_count(self) -> int:
        return len(self.entry_lengths)

    def entry_index(self, generation: int, document: int) -> int:
        return self.entry_base[generation] + document

    def cumulative_blocks(self, generation: int, rows: int) -> list[int]:
        table = [0]
        for length in self.document_lengths[generation]:
            table.append(table[-1] + block_count(length, rows))
        table.extend([table[-1]] * (CUMULATIVE_EDGES - len(table)))
        return table

    def entry_padded_rows(self, entry: int) -> int:
        """Workspace rows an entry occupies: the larger of the ceil(L/64)*64 rows the backward
        kernel writes and the ceil(L/128)*128 rows the dK/dV postprocess reads."""

        length = self.entry_lengths[entry]
        return max(
            block_count(length, BWD_BLOCK_ROWS) * BWD_BLOCK_ROWS,
            block_count(length, DKV_POSTPROCESS_TILE_M) * DKV_POSTPROCESS_TILE_M,
        )

    # The slot layout: each generation's padded rows, moved to the base of its slot.

    def slot_of_generation(self, generation: int) -> int:
        return generation % self.slot_count

    def generation_padded_base(self, generation: int) -> int:
        """FA4's padded offset of the generation's first document in the packed batch."""

        return padded_offset(self.cu_seqlens, self.entry_index(generation, 0))

    def generation_span(self, generation: int) -> int:
        """Rows from the generation's first padded offset to the end of its last entry."""

        last = self.entry_index(generation, len(self.document_lengths[generation]) - 1)
        return (
            padded_offset(self.cu_seqlens, last)
            + self.entry_padded_rows(last)
            - self.generation_padded_base(generation)
        )

    def slot_padded_offset(self, entry: int) -> int:
        """The slot-table value of `entry`: its padded offset within its generation, plus the
        base row of the generation's slot."""

        generation = self.generation_of_entry[entry]
        rel = padded_offset(self.cu_seqlens, entry) - self.generation_padded_base(generation)
        return self.slot_of_generation(generation) * SLOT_ROWS + rel

    def slot_row_base(self, generation: int) -> int:
        """First workspace row of the generation's slot."""

        return self.slot_of_generation(generation) * SLOT_ROWS

    def workspace_rows(self) -> int:
        """Workspace rows the slots need."""

        return self.slot_count * SLOT_ROWS

    def check_slot_layout(self) -> None:
        """Checks the slot layout against the workspace allocation.

        Each generation's span must fit a slot, every entry's rows must end inside the
        workspace, and every slot-table value must be a multiple of BWD_BLOCK_ROWS.
        """

        span = max(self.generation_span(g) for g in range(self.generations))
        reach = max(
            self.slot_padded_offset(b) + self.entry_padded_rows(b)
            for b in range(self.entry_count)
        )
        aligned = all(
            self.slot_padded_offset(b) % BWD_BLOCK_ROWS == 0 for b in range(self.entry_count)
        )
        if span > SLOT_ROWS or reach > self.workspace_rows() or not aligned:
            raise ValueError(
                f"the slot layout does not fit: span {span} of {SLOT_ROWS} slot rows, reach "
                f"{reach} of {self.workspace_rows()} rows, tile-aligned {aligned}"
            )



# Control-table reads shared by the schedulers and the backward phases.


@cute.jit
def _cumulative_blocks(
    control: cute.Tensor, base: cutlass.Constexpr[int], gen: Int32, slot: Int32
):
    """Edge `slot` (0..MAX_DOCUMENTS) of generation `gen`'s cumulative block table at `base`."""

    return Int32(control[Int32(base) + gen * Int32(CUMULATIVE_EDGES) + slot])


@cute.jit
def _locate_block(
    control: cute.Tensor, base: cutlass.Constexpr[int], gen: Int32, block_ordinal: Int32
):
    """Maps a block ordinal of generation `gen` to (document, block within the document).

    It compares against every edge of the table rather than leaving the loop early.
    """

    document = Int32(0)
    for slot in cutlass.range(0, MAX_DOCUMENTS, 1, unroll=1):
        if block_ordinal >= _cumulative_blocks(control, base, gen, Int32(slot + 1)):
            document = Int32(slot + 1)
    start = _cumulative_blocks(control, base, gen, document)
    return document, block_ordinal - start


@cute.jit
def _generation_block_count(
    control: cute.Tensor, base: cutlass.Constexpr[int], gen: Int32
) -> Int32:
    return _cumulative_blocks(
        control, base, gen, Int32(control[Int32(CONTROL_DOCUMENT_COUNT) + gen])
    )


@cute.jit
def _entry_index(control: cute.Tensor, gen: Int32, document: Int32) -> Int32:
    return Int32(control[Int32(CONTROL_ENTRY_BASE) + gen]) + document


@dataclass
class AttentionSchedulerParams(ParamsBase):
    """The schedulers' kernel parameters: the control table, and the tile bound and cluster
    shape that `get_grid_shape`, inherited from `StaticPersistentTileScheduler`, reads."""

    num_head_divmod: FastDivmodDivisor
    total_blocks_cluster: Int32
    generation_control: cute.Tensor
    cluster_shape_m: cutlass.Constexpr[int] = 1


class AttentionTileScheduler(StaticPersistentTileScheduler):
    """A persistent FA4 tile scheduler over the current generation's documents.

    The generation is the control table's `CONTROL_ROUTE` word. Each CTA starts at its block
    index and strides by the grid size. The work tile carries the document's batch entry, so FA4
    finds the document through `cu_seqlens` and `seqused`. Subclasses set the head count, the
    cumulative block table and the per-head block bound.
    """

    num_head: int = 0
    cumulative_table: int = 0
    max_blocks_per_head: int = 0
    use_schedule: bool = False

    @classmethod
    def _make_params(cls, args: TileSchedulerArguments) -> AttentionSchedulerParams:
        assert args.mSeqUsedQ is not None
        assert const_expr(args.cluster_shape_mn[0] == 1)
        return AttentionSchedulerParams(
            FastDivmodDivisor(cls.num_head),
            Int32(cls.max_blocks_per_head * cls.num_head * CONTROL_ENTRIES),
            args.mSeqUsedQ,
            cluster_shape_m=1,
        )

    @classmethod
    def to_underlying_arguments(
        cls, args: TileSchedulerArguments, *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC, loc=None, ip=None,
    ) -> AttentionSchedulerParams:
        del loc, ip
        assert scheduling_mode == SchedulingMode.STATIC
        return cls._make_params(args)

    @classmethod
    @cute.jit
    def create(cls, params, clc=None, *, loc=None, ip=None):
        del clc
        assert const_expr(cute.size(params.cluster_shape_m) == 1)
        return cls(params, cute.arch.block_idx()[0], loc=loc, ip=ip)

    @cute.jit
    def _tile_count(self) -> Int32:
        control = self.params.generation_control
        gen = Int32(control[Int32(CONTROL_ROUTE)])
        return (
            _generation_block_count(control, self.cumulative_table, gen)
            * Int32(self.num_head)
        )

    @cute.jit
    def _permute_block(self, block_idx: Int32, head_idx: Int32, blocks_of_doc: Int32) -> Int32:
        return block_idx

    @cute.jit
    def _decode_tile(self, control: cute.Tensor, gen: Int32):
        """Maps the current tile index to (document, block, head) in generation `gen`.

        Tiles are ordered by document, then head, then block:

            ordinal = cum[doc]*num_head + head*blocks_of(doc) + block

        With `use_schedule`, the tile index is first mapped through the forward schedule in the
        control table.
        """

        ordinal = self._tile_idx
        if const_expr(self.use_schedule):
            ordinal = Int32(
                control[
                    Int32(self.schedule_base)
                    + gen * Int32(self.schedule_stride)
                    + ordinal
                ]
            )
        document = Int32(0)
        for slot in cutlass.range(0, MAX_DOCUMENTS, 1, unroll=1):
            edge = _cumulative_blocks(
                control, self.cumulative_table, gen, Int32(slot + 1)
            ) * Int32(self.num_head)
            if ordinal >= edge:
                document = Int32(slot + 1)
        # A tile index past the end scans past the last document. Its coordinates are never
        # used (`is_valid` is false), but the division below needs a nonzero block count, so
        # the document is clamped to the last one.
        last = Int32(control[Int32(CONTROL_DOCUMENT_COUNT) + gen]) - Int32(1)
        if document > last:
            document = last
        start = _cumulative_blocks(control, self.cumulative_table, gen, document)
        blocks_of_doc = _cumulative_blocks(
            control, self.cumulative_table, gen, document + Int32(1)
        ) - start
        local = Int32(ordinal) - start * Int32(self.num_head)
        head_idx = local // blocks_of_doc
        block_idx = local - head_idx * blocks_of_doc
        block_idx = self._permute_block(Int32(block_idx), Int32(head_idx), Int32(blocks_of_doc))
        return document, Int32(block_idx), Int32(head_idx)

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        del loc, ip
        control = self.params.generation_control
        gen = Int32(control[Int32(CONTROL_ROUTE)])
        document, block_idx, head_idx = self._decode_tile(control, gen)
        entry = _entry_index(control, gen, document)
        is_valid = self._tile_idx < self._tile_count()
        return WorkTileInfo((Int32(block_idx), Int32(head_idx), entry, Int32(0)), is_valid)

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None):
        del loc, ip
        return self.get_current_work()

    @cute.jit
    def advance_to_next_work(self, *, loc=None, ip=None):
        del loc, ip
        self._tile_idx = self._tile_idx + cute.arch.grid_dim()[0]
        return self.get_current_work()

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip((self.params, self._tile_idx), self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return type(self)(*tuple(obj_list), loc=self._loc)


class AttentionBackwardScheduler(AttentionTileScheduler):
    """The backward kernel's scheduler: one tile per (document, query head, 64-row K block).

    It walks the 64-row block table, `CONTROL_BWD_CUMULATIVE_BLOCKS`, which matches the kernel's
    K tile. Walking the dK/dV postprocess's 128-row table instead would skip the upper half of
    each document's K blocks and leave their dK and dV at zero.
    """

    num_head = QUERY_HEADS
    cumulative_table = CONTROL_BWD_CUMULATIVE_BLOCKS
    max_blocks_per_head = BWD_BLOCKS_PER_SEQUENCE

    @cute.jit
    def _permute_block(self, block_idx: Int32, head_idx: Int32, blocks_of_doc: Int32) -> Int32:
        """Reverses the block order within each document for odd query heads.

        The map is a bijection on each document's blocks, so it changes only which CTA runs a
        tile and when; every tile is still visited once.
        """

        reverse = Int32(head_idx & Int32(1))
        reversed_idx = blocks_of_doc - Int32(1) - block_idx
        return block_idx + reverse * (reversed_idx - block_idx)

    @cute.jit
    def advance_to_next_work(self, *, loc=None, ip=None):
        """Advances like the base scheduler, after draining the K and V shared memory.

        FA4's own scheduler gives each CTA one tile, so its load warp writes a tile's K and V
        into `sK` and `sV` without waiting for a previous tile, whose GEMMs read them and whose
        dK/dV epilogue stages its bulk reduce-adds in `sV`. Before a CTA moves on to another
        tile, warp 4 waits until its bulk reduce-adds have read their shared memory, the MMA
        warps (4 to 11) meet on FA4's epilogue barrier, and then they release the load warp
        (warp 0) through barrier 12, the first ID after FA4's backward named barriers.
        """

        del loc, ip
        next_tile = self._tile_idx + cute.arch.grid_dim()[0]
        if next_tile < self._tile_count():
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == 0:
                cute.arch.barrier(barrier_id=12, number_of_threads=32 + 256)
            elif warp_idx >= 4:
                if warp_idx == 4:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.Epilogue), number_of_threads=256
                )
                cute.arch.barrier(barrier_id=12, number_of_threads=32 + 256)
        self._tile_idx = next_tile
        return self.get_current_work()


class AttentionForwardScheduler(AttentionTileScheduler):
    """The forward member's scheduler: one tile per (document, KV head, query block).

    Tiles are visited in the order of the balanced schedule staged in the control table. With
    `pack_gqa` a tile holds all query heads of one KV head, so the head axis is the KV head.
    """

    num_head = KV_HEADS
    cumulative_table = CONTROL_FWD_CUMULATIVE_BLOCKS
    max_blocks_per_head = FWD_BLOCKS_PER_SEQUENCE
    use_schedule = True
    schedule_base = CONTROL_FWD_SCHEDULE
    schedule_stride = MAX_FWD_TASKS_PER_GENERATION


class SlotSeqlenInfoQK:
    """`SeqlenInfoQK` whose padded offsets come from the slot table.

    `padded_offset_q` and `padded_offset_k` are read from the control table at
    `CONTROL_Q_PADDED_BASE` and `CONTROL_K_PADDED_BASE` plus the batch entry, so the backward
    kernel's workspace rows (the log2 LSE, dPsum and the dQ, dK and dV accumulators) fall in the
    generation's slot. Every other field is FA4's, including the token offsets `offset_q` and
    `offset_k` into Q, K, V and dO. The slot table lies past `seqused`'s entries, so FA4's own
    `seqused[batch_idx]` reads are unaffected. The Q and K sides read separate regions; with
    64 x 64 tiles they hold equal values.
    """
    @staticmethod
    def create(
        batch_idx, seqlen_q_static, seqlen_k_static,
        mCuSeqlensQ=None, mCuSeqlensK=None, mSeqUsedQ=None, mSeqUsedK=None,
        mCuTotalMBlocks=None, mCuBlockIdxOffsets=None,
        tile_m: cutlass.Constexpr[int] = 128,
        tile_n: cutlass.Constexpr[int] = 128,
    ):
        native = SeqlenInfoQK.create(
            batch_idx, seqlen_q_static, seqlen_k_static,
            mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK,
            mCuTotalMBlocks, mCuBlockIdxOffsets, tile_m, tile_n,
        )
        padded_offset_q = native.padded_offset_q
        padded_offset_k = native.padded_offset_k
        if const_expr(mSeqUsedQ is not None):
            padded_offset_q = cute.assume(
                mSeqUsedQ[Int32(CONTROL_Q_PADDED_BASE) + batch_idx], divby=tile_m
            )
        if const_expr(mSeqUsedK is not None):
            padded_offset_k = cute.assume(
                mSeqUsedK[Int32(CONTROL_K_PADDED_BASE) + batch_idx], divby=tile_n
            )
        return SeqlenInfoQK(
            native.offset_q, native.offset_k,
            padded_offset_q, padded_offset_k,
            native.seqlen_q, native.seqlen_k,
            native.m_block_offset, native.block_idx_offset, native.num_n_blocks,
            has_cu_seqlens_q=native.has_cu_seqlens_q,
            has_cu_seqlens_k=native.has_cu_seqlens_k,
            has_seqused_q=native.has_seqused_q,
            has_seqused_k=native.has_seqused_k,
        )


# The backward phases.


@cute.jit
def clear_dkv_accumulators(
    dk_accum: cute.Tensor,
    dv_accum: cute.Tensor,
    row_base: Int32,
    row_count: Int32,
    workspace_rows: cutlass.Constexpr[int],
):
    """Zeroes rows `[row_base, row_base + row_count)` of the dK and dV accumulators of every KV
    head.

    The backward kernel accumulates dK and dV into these rows with bulk reduce-adds, so they
    must start at zero. The caller passes the generation's slot base and span, so the clear
    covers every document of the generation, including padding rows that the kernel does not
    write but the dK/dV postprocess reads in 128-row tiles. The grid barriers between the
    phases order the clear after the slot's previous postprocess and before the kernel.

    Every thread of the grid takes part, role 0 included, so the body must fit role 0's 56
    registers; it walks the KV heads in an outer runtime loop rather than dividing a flat index.
    """

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    start = bidx * PROGRAM_THREADS + tidx
    stride = grid_x * PROGRAM_THREADS
    span = row_count * Int32(HEAD_DIM)
    base = row_base * Int32(HEAD_DIM)
    for head in cutlass.range(0, KV_HEADS, 1, unroll=1):
        column = Int32(start)
        while column < span:
            offset = Int32(head * workspace_rows * HEAD_DIM) + base + column
            dk_accum[offset] = Float32(0.0)
            dv_accum[offset] = Float32(0.0)
            column += Int32(stride)


LOG2_E = math.log2(math.e)
# The preprocess reduces one row per warp: each lane sums PREPROCESS_COLUMNS_PER_LANE columns
# and a butterfly across the warp adds the lanes. A warpgroup holds PREPROCESS_ROWS_PER_WAVE
# rows at a time and covers a 64-row block in PREPROCESS_WAVES_PER_BLOCK waves.
PREPROCESS_LANES_PER_ROW = 32
PREPROCESS_SHUFFLE_STEPS = int(math.log2(PREPROCESS_LANES_PER_ROW))
PREPROCESS_ROWS_PER_WAVE = 128 // PREPROCESS_LANES_PER_ROW            # 4
PREPROCESS_WAVES_PER_BLOCK = BWD_BLOCK_ROWS // PREPROCESS_ROWS_PER_WAVE        # 16
PREPROCESS_COLUMNS_PER_LANE = HEAD_DIM // PREPROCESS_LANES_PER_ROW   # 4
# Upper bounds on the postprocess tasks one CTA runs, one task per wave.
MAX_DQ_POSTPROCESS_WAVES = math.ceil(MAX_BWD_BLOCKS_PER_GENERATION * QUERY_HEADS / PROGRAM_CTAS)
MAX_DKV_POSTPROCESS_WAVES = math.ceil(MAX_DKV_BLOCKS_PER_GENERATION * KV_HEADS / PROGRAM_CTAS)


@cute.jit
def _backward_preprocess_rows(
    control: cute.Tensor,
    cu_seqlens: cute.Tensor,
    out_flat: cute.Tensor,
    dout_flat: cute.Tensor,
    lse_flat: cute.Tensor,
    dpsum_flat: cute.Tensor,
    lse_log2_flat: cute.Tensor,
    dq_accum_flat: cute.Tensor,
    gen: Int32,
    workspace_rows: cutlass.Constexpr[int],
    worker_group: cutlass.Constexpr[int],
    worker_groups: cutlass.Constexpr[int],
):
    """The backward preprocess of generation `gen`, on warpgroup `worker_group` of every CTA.

    Tasks are (64-row block, query head) pairs, dealt in turn to the `worker_groups`
    warpgroups of the grid. Every row gets the values FA4's `FlashAttentionBackwardPreprocess`
    writes:

        row <  seqlen : dpsum = sum_d O*dO,  lse_log2 = lse*log2(e)  (-inf -> 0)
        row >= seqlen : dpsum = 0,           lse_log2 = +inf
        every row     : dq_accum = 0         over the whole 64-row block

    The +inf makes the backward kernel's `exp2(S - lse_log2)` zero on rows past the document's
    end, so those rows add nothing to dK and dV. Rows are addressed from the entry's slot-table
    base (`CONTROL_Q_PADDED_BASE`), the base `SlotSeqlenInfoQK` gives the kernel. The body also
    runs on role 0, so it must fit 56 registers.
    """

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    local_tidx = tidx - Int32(worker_group * 128)
    row_group = local_tidx // PREPROCESS_LANES_PER_ROW
    lane_in_row = local_tidx - row_group * PREPROCESS_LANES_PER_ROW
    total_blocks = _generation_block_count(control, CONTROL_BWD_CUMULATIVE_BLOCKS, gen)
    task_count = total_blocks * Int32(QUERY_HEADS)

    task = Int32(bidx + worker_group * grid_x)
    while task < Int32(task_count):
        block_ordinal = task // Int32(QUERY_HEADS)
        head = task - block_ordinal * Int32(QUERY_HEADS)
        document, m_block = _locate_block(
            control, CONTROL_BWD_CUMULATIVE_BLOCKS, gen, block_ordinal
        )
        entry = _entry_index(control, gen, document)
        token_base = Int32(cu_seqlens[entry])
        seqlen = Int32(control[Int32(CONTROL_SEQUSED) + entry])
        padded_base = Int32(control[Int32(CONTROL_Q_PADDED_BASE) + entry])   # the slot table

        # Waves cover disjoint rows. Each lane adds its columns in ascending order and the
        # butterfly below is unrolled at trace time, so the FP32 summation order is fixed.
        for row_wave in cutlass.range(0, PREPROCESS_WAVES_PER_BLOCK, 1, unroll=1):
            row_in_block = row_group + row_wave * PREPROCESS_ROWS_PER_WAVE
            row = Int32(m_block) * Int32(BWD_BLOCK_ROWS) + Int32(row_in_block)
            padded_row = padded_base + row
            inside = row < seqlen
            partial = Float32(0.0)
            for item in cutlass.range(0, PREPROCESS_COLUMNS_PER_LANE, 1, unroll=1):
                column = lane_in_row * PREPROCESS_COLUMNS_PER_LANE + item
                dq_accum_flat[
                    (head * Int32(workspace_rows) + padded_row) * Int32(HEAD_DIM)
                    + Int32(column)
                ] = Float32(0.0)
                if inside:
                    packed_row = token_base + row
                    dense = (
                        packed_row * Int32(QUERY_HEADS) + head
                    ) * Int32(HEAD_DIM) + Int32(column)
                    partial += out_flat[dense].to(Float32) * dout_flat[dense].to(Float32)
            for step in cutlass.range_constexpr(PREPROCESS_SHUFFLE_STEPS):
                partial += cute.arch.shuffle_sync_bfly(partial, 1 << step)
            if lane_in_row == 0:
                scratch = head * Int32(workspace_rows) + padded_row
                value = Float32(0.0)
                log2_value = Float32.inf
                if inside:
                    value = partial
                    lse = lse_flat[head * Int32(LSE_HEAD_STRIDE) + token_base + row]
                    log2_value = (
                        lse * Float32(LOG2_E) if lse != -Float32.inf else Float32(0.0)
                    )
                dpsum_flat[scratch] = value
                lse_log2_flat[scratch] = log2_value
        task += Int32(worker_groups * grid_x)


# The LSE is stored generation-major: generation g's LSE is a [QUERY_HEADS][SEQUENCE] slab at
# g * LSE_GENERATION_STRIDE, so the head stride is SEQUENCE whatever the number of generations.
# The forward (through `generation_major_lse`) and the preprocess address it as
# origin + head * LSE_HEAD_STRIDE + token, with the packed token index g * SEQUENCE + row;
# moving the origin by g * LSE_GENERATION_SHIFT lands that address in generation g's slab.
LSE_HEAD_STRIDE = SEQUENCE
LSE_GENERATION_STRIDE = QUERY_HEADS * SEQUENCE
LSE_GENERATION_SHIFT = LSE_GENERATION_STRIDE - SEQUENCE  # (QUERY_HEADS - 1) * SEQUENCE

# Positions in FA4's forward kernel arguments (`forward_setup`'s parameters). The program binds
# `mSeqUsedQ` to the control table. `generation_major_lse` checks the LSE's layout, so a change
# of signature that moves `mLSE` fails when the program is compiled.
FWD_ARG_LSE = 4                                   # mLSE
FWD_ARG_SEQUSED_Q = 7                             # mSeqUsedQ, the control table

assert LSE_GENERATION_SHIFT * 4 % 16 == 0, (
    "the per-generation pre-shift must preserve `lse`'s 16-byte alignment"
)


def check_generation_token_ranges(plan) -> None:
    """Checks the token layout the generation-major LSE relies on; raises `ValueError`.

    The LSE addressing works only if generation `g` holds the packed tokens
    `[g * SEQUENCE, (g + 1) * SEQUENCE)`. Otherwise the origin shift no longer cancels the
    token base and the forward writes the LSE to the wrong rows, which nothing else would
    notice, because every row is still written once. The program checks this on the host
    before compiling.
    """

    for generation in range(plan.generations):
        entry = plan.entry_base[generation]
        base = plan.cu_seqlens[entry]
        if base != generation * SEQUENCE:
            raise ValueError(
                f"generation {generation} starts at token {base}, not {generation * SEQUENCE}: "
                "the generation-major `lse` layout is not valid for this plan"
            )
        total = sum(plan.document_lengths[generation])
        if total != SEQUENCE:
            raise ValueError(
                f"generation {generation} packs {total} tokens, not {SEQUENCE}"
            )


def generation_major_lse(mLSE: cute.Tensor, gen: Int32, total_tokens: int) -> cute.Tensor:
    """FA4's recorded `mLSE`, rebuilt for the generation-major layout at generation `gen`.

    The recorded tensor already carries FA4's two host-side transforms, whose strides this
    checks (`TT` is `total_tokens`, `G` the query heads per KV head):

        select(mLSE, [1, 0])            (h, total_q):(TT, 1)  ->  (total_q, h):(1, TT)
        pack_gqa_layout(.., head_idx=1) ->  ((G, total_q), H_kv) : ((TT, 1), TT*G)

    The rebuild puts `SEQUENCE` in place of `TT` in every stride the head index reaches, as
    `pack_gqa_layout` does with its head stride, so the head numbering `h_kv*G + qh` is kept.
    """

    layout = mLSE.layout
    shape, stride = layout.shape, layout.stride
    packed = not isinstance(shape[0], int)

    if packed:
        # ((G, total_q), H_kv) : ((TT, 1), TT*G)
        group = int(shape[0][0])
        got = (int(stride[0][0]), int(stride[0][1]), int(stride[1]))
        want = (total_tokens, 1, total_tokens * group)
        if got != want:
            raise ValueError(
                f"FA4's packed-GQA LSE layout is not the one the LSE rebuild assumes: "
                f"strides {got}, expected {want}"
            )
        new_stride = ((LSE_HEAD_STRIDE, 1), LSE_HEAD_STRIDE * group)
    else:
        # (total_q, h) : (1, TT)
        got = (int(stride[0]), int(stride[1]))
        want = (1, total_tokens)
        if got != want:
            raise ValueError(
                f"FA4's LSE layout is not the one the LSE rebuild assumes: "
                f"strides {got}, expected {want}"
            )
        new_stride = (1, LSE_HEAD_STRIDE)

    # Move the origin to generation `gen`'s slab. The shape is unchanged, so FA4's predication
    # against it is too.
    origin = lse_generation_origin(mLSE.iterator, QUERY_HEADS * total_tokens, gen)
    return cute.make_tensor(origin, cute.make_layout(shape, stride=new_stride))


def lse_generation_origin(iterator, elements: int, gen: Int32):
    """`iterator` moved by `gen * LSE_GENERATION_SHIFT` elements, to generation `gen`'s slab.

    `cute.domain_offset` needs a coordinate with the layout's structure, so the shift is taken
    on a rank-1 `(elements,)` view: `make_layout(elements)` has a scalar shape and rejects a
    1-tuple coordinate. Only the resulting pointer is kept.
    """

    flat = cute.make_tensor(iterator, cute.make_layout((elements,)))
    return cute.domain_offset((gen * Int32(LSE_GENERATION_SHIFT),), flat).iterator


@cute.jit
def attention_backward_preprocess(
    control: cute.Tensor,
    cu_seqlens: cute.Tensor,
    out_flat: cute.Tensor,
    dout_flat: cute.Tensor,
    lse_flat: cute.Tensor,
    dpsum_flat: cute.Tensor,
    lse_log2_flat: cute.Tensor,
    dq_accum_flat: cute.Tensor,
    gen: Int32,
    workspace_rows: cutlass.Constexpr[int],
    total_tokens: cutlass.Constexpr[int],
    worker_group: cutlass.Constexpr[int],
    worker_groups: cutlass.Constexpr[int],
):
    """The backward preprocess of generation `gen`, reading the generation-major LSE.

    It runs `_backward_preprocess_rows` with the LSE origin moved to generation `gen`'s slab,
    so the body's `head * LSE_HEAD_STRIDE + token` reads the rows the forward wrote.
    """

    # Only the base pointer changes; the body sees the same rank-1 layout.
    origin = lse_generation_origin(lse_flat.iterator, QUERY_HEADS * total_tokens, gen)
    lse_gen = cute.make_tensor(origin, lse_flat.layout)
    _backward_preprocess_rows(
        control, cu_seqlens, out_flat, dout_flat, lse_gen,
        dpsum_flat, lse_log2_flat, dq_accum_flat,
        gen, workspace_rows, worker_group, worker_groups,
    )


@cute.jit
def attention_backward_postprocess(
    postprocess_dq,
    post_dq_args,
    postprocess_dkv,
    post_dkv_args,
    control: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdKaccum: cute.Tensor,
    mdVaccum: cute.Tensor,
    mdQ: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    mdVMirror: cute.Tensor,
    mCuSeqlensQ: cute.Tensor,
    softmax_scale: Float32,
    gen: Int32,
    role: cutlass.Constexpr[int],
):
    """FA4's dQ and dK/dV postprocess for generation `gen`, run by roles 1 and 2.

    CTA `bidx` takes tasks `bidx + wave * grid` from the dQ tasks (64-row block, query head),
    then from the dK/dV tasks (128-row block, KV head). Each task body finds its document
    through the batch entry and computes FA4's padded offset from `cu_seqlens` at its own tile
    size (64 for dQ, 128 for dK/dV); the bodies know nothing of slots. So the accumulator
    operand is shifted beforehand by the entry's slot-table value minus that offset
    (`CONTROL_DQ_POSTPROCESS_SHIFT`, `CONTROL_DKV_POSTPROCESS_SHIFT`), which lands each body on
    the rows the backward kernel wrote. dK uses the direct-load body; dV uses the two-output
    body, which also writes the V columns of the packed dQKV rows (`mdVMirror`).

    `bidx` goes through `anchor_index` inside this phase, so LLVM cannot hoist the task
    arithmetic out of the layer loop and keep it live across the backward kernel's
    224-register phase.
    """

    if const_expr(role > 0):
        raw_bidx, _, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()
        bidx = anchor_index(raw_bidx)

        dq_blocks = _generation_block_count(control, CONTROL_BWD_CUMULATIVE_BLOCKS, gen)
        dq_tasks = dq_blocks * Int32(QUERY_HEADS)
        # `task < dq_tasks` is uniform across the CTA's 256 threads, so they all reach the
        # barrier that ends each task, which keeps the next task body from overwriting shared
        # memory the previous one is still reading.
        for wave in cutlass.range(0, MAX_DQ_POSTPROCESS_WAVES, 1, unroll=1):
            task = Int32(bidx + wave * grid_x)
            if task < dq_tasks:
                block_ordinal = task // Int32(QUERY_HEADS)
                head_idx = task - block_ordinal * Int32(QUERY_HEADS)
                document, m_block = _locate_block(
                    control, CONTROL_BWD_CUMULATIVE_BLOCKS, gen, block_ordinal
                )
                entry = _entry_index(control, gen, document)
                # Shift the dQ accumulator so the body's padded offset lands on the slot rows.
                dq_shifted = cute.domain_offset(
                    (
                        0,
                        Int32(control[Int32(CONTROL_DQ_POSTPROCESS_SHIFT) + entry])
                        * Int32(HEAD_DIM),
                    ),
                    mdQaccum,
                )
                postprocess_task_body(
                    postprocess_dq, dq_shifted, mdQ, mCuSeqlensQ, None,
                    softmax_scale, *post_dq_args,
                    Int32(m_block), Int32(head_idx), entry,
                )
                cute.arch.barrier(barrier_id=2, number_of_threads=256)

        kv_blocks = _generation_block_count(control, CONTROL_DKV_CUMULATIVE_BLOCKS, gen)
        kv_tasks = kv_blocks * Int32(KV_HEADS)
        for wave in cutlass.range_constexpr(MAX_DKV_POSTPROCESS_WAVES):
            task = Int32(bidx + wave * grid_x)
            if task < kv_tasks:
                block_ordinal = task // Int32(KV_HEADS)
                head_idx = task - block_ordinal * Int32(KV_HEADS)
                document, m_block = _locate_block(
                    control, CONTROL_DKV_CUMULATIVE_BLOCKS, gen, block_ordinal
                )
                entry = _entry_index(control, gen, document)
                # One shift corrects both the body's 128-row padded offset and the slot base.
                shift = (
                    Int32(control[Int32(CONTROL_DKV_POSTPROCESS_SHIFT) + entry]) * Int32(HEAD_DIM)
                )
                dk_shifted = cute.domain_offset((0, shift), mdKaccum)
                dv_shifted = cute.domain_offset((0, shift), mdVaccum)
                postprocess_task_body_direct_load(
                    postprocess_dkv, dk_shifted, mdK, mCuSeqlensQ, None,
                    softmax_scale, *post_dkv_args,
                    Int32(m_block), Int32(head_idx), entry,
                )
                cute.arch.barrier(barrier_id=2, number_of_threads=256)
                postprocess_task_body_two_outputs(
                    postprocess_dkv, dv_shifted, mdV, mdVMirror,
                    mCuSeqlensQ, None, Float32(1.0), *post_dkv_args,
                    Int32(m_block), Int32(head_idx), entry,
                )
                cute.arch.barrier(barrier_id=2, number_of_threads=256)


@cute.jit
def record_attention_forward_args(self, *args):
    """Traces FA4's forward `__call__` (`fa4_forward_call.forward_call_setup`) for the member.

    The member's `kernel` records the 36 kernel arguments instead of launching; the program
    passes them to its kernel, whose code hands them to `run_attention_forward`.
    """

    (mQ, mK, mV, mO, mLSE, softmax_scale, cu_q, cu_k, seqused_q, seqused_k, stream) = args
    self._forward_setup(
        self.forward, mQ, mK, mV, mO, mLSE, softmax_scale,
        cu_q, cu_k, seqused_q, seqused_k,
        None, None, None, None, None, AuxData(), stream,
    )


@cute.jit
def run_attention_forward(
    self, role: cutlass.Constexpr[int], phase_counter, *forward_args
):
    """Runs the forward member on the current generation, with its LSE stored generation-major.

    `forward_args` are the member's recorded kernel arguments; the generation is the control
    table's `CONTROL_ROUTE` word.
    """

    # A freshly anchored page pointer keeps LLVM from hoisting the member's shared-memory
    # addresses to kernel entry, where they would stay live across the other members' phases.
    self.reanchor_smem_page()
    control = forward_args[FWD_ARG_SEQUSED_Q]
    gen = Int32(control[Int32(CONTROL_ROUTE)])
    args = list(forward_args)
    args[FWD_ARG_LSE] = generation_major_lse(
        forward_args[FWD_ARG_LSE], gen, self.total_tokens
    )
    # Called through its `cute.jit` wrapper, not `__wrapped__`, so the DSL's AST preprocessing
    # applies to it.
    run_attention_forward_member(self, role, phase_counter, *args)


# Positions in the backward member's recorded arguments: FA4's backward kernel parameters (44),
# then the dQ and the dK/dV postprocess arguments (7 each); the preprocess's two, last, are not
# used here. The last three constants index the kernel parameters (`backward_setup`'s).
BWD_MAIN_ARG_COUNT = 44
BWD_POST_DKV_ARGS_START = BWD_MAIN_ARG_COUNT + 7
BWD_POST_ARGS_END = BWD_POST_DKV_ARGS_START + 7
BWD_ARG_CU_SEQLENS_Q = 15
BWD_ARG_SEQUSED_Q = 17
BWD_ARG_SOFTMAX_SCALE = 31


@cute.jit
def run_attention_backward(
    self,
    role: cutlass.Constexpr[int],
    generation: Int32,
    phase_counter: cute.Tensor,
    ws_lse_log2: cute.Tensor,
    ws_dpsum: cute.Tensor,
    ws_dq_accum: cute.Tensor,
    ws_dk_accum: cute.Tensor,
    ws_dv_accum: cute.Tensor,
    prog_out: cute.Tensor,
    prog_dout: cute.Tensor,
    prog_lse: cute.Tensor,
    prog_dq: cute.Tensor,
    prog_dk: cute.Tensor,
    prog_dv: cute.Tensor,
    *backward_args,
):
    """Runs the backward attention of generation `generation` (the layer slot) in three phases.

    `backward_args` is the packed dQKV view followed by the backward member's recorded
    arguments. Grid barriers separate the phases: the dK/dV clear with the preprocess, FA4's
    backward kernel, and the postprocess (the caller adds the barrier after the last).
    """

    self.reanchor_smem_page()

    packed_dqkv = backward_args[0]
    backward_args = backward_args[1:]

    fa4_args = backward_args[:BWD_MAIN_ARG_COUNT]
    post_dq_args = backward_args[BWD_MAIN_ARG_COUNT:BWD_POST_DKV_ARGS_START]
    post_dkv_args = backward_args[BWD_POST_DKV_ARGS_START:BWD_POST_ARGS_END]
    control = fa4_args[BWD_ARG_SEQUSED_Q]          # the control table
    cu_seqlens = fa4_args[BWD_ARG_CU_SEQLENS_Q]
    rows = self.workspace_rows
    tokens = self.total_tokens

    # Phase 1: zero the generation's dK/dV accumulator rows in its slot, and the preprocess.
    dk_clear = cute.make_tensor(
        assume_pointer_aligned(ws_dk_accum, 16).iterator,
        cute.make_layout(KV_HEADS * rows * HEAD_DIM),
    )
    dv_clear = cute.make_tensor(
        assume_pointer_aligned(ws_dv_accum, 16).iterator,
        cute.make_layout(KV_HEADS * rows * HEAD_DIM),
    )
    clear_dkv_accumulators(
        dk_clear, dv_clear,
        Int32(control[Int32(CONTROL_DKV_SPAN_BASE) + generation]),
        Int32(control[Int32(CONTROL_DKV_SPAN_ROWS) + generation]),
        rows,
    )

    out_flat = cute.make_tensor(
        prog_out.iterator, cute.make_layout(tokens * QUERY_HEADS * HEAD_DIM)
    )
    dout_flat = cute.make_tensor(
        prog_dout.iterator, cute.make_layout(tokens * QUERY_HEADS * HEAD_DIM)
    )
    lse_flat = cute.make_tensor(
        prog_lse.iterator, cute.make_layout(QUERY_HEADS * tokens)
    )
    dpsum_flat = cute.make_tensor(
        ws_dpsum.iterator, cute.make_layout(QUERY_HEADS * rows)
    )
    lse_log2_flat = cute.make_tensor(
        ws_lse_log2.iterator, cute.make_layout(QUERY_HEADS * rows)
    )
    dq_accum_flat = cute.make_tensor(
        ws_dq_accum.iterator, cute.make_layout(QUERY_HEADS * rows * HEAD_DIM)
    )
    attention_backward_preprocess(
        control, cu_seqlens,
        out_flat, dout_flat, lse_flat,
        dpsum_flat, lse_log2_flat, dq_accum_flat,
        generation, rows, tokens, role, 3,
    )
    self.grid_phase_barrier(phase_counter)

    # Phase 2: FA4's backward kernel; SlotSeqlenInfoQK puts its workspace rows in the slot.
    state = self.backward_setup(self.backward, *fa4_args)
    self.backward_role(self.backward, *fa4_args, *state, role)
    cute.arch.barrier(barrier_id=15, number_of_threads=PROGRAM_THREADS)
    self.grid_phase_barrier(phase_counter)

    # Phase 3: the postprocess. The generation index and the tensor addresses are anchored
    # again after the kernel. Every control-table read and address below derives from them, so
    # none of that work can be moved above the kernel and kept live across its 224-register
    # phase.
    gen_post = anchor_index(generation)

    post_dq_accum = anchored_tensor(ws_dq_accum)
    post_dk_accum = anchored_tensor(ws_dk_accum)
    post_dv_accum = anchored_tensor(ws_dv_accum)
    post_dq_out = anchored_tensor(prog_dq)
    post_dk_out = anchored_tensor(prog_dk)
    post_dv_out = anchored_tensor(prog_dv)
    post_packed_dqkv = anchored_tensor(packed_dqkv)

    dq_accum = cute.make_tensor(
        assume_pointer_aligned(post_dq_accum, 16).iterator,
        cute.make_layout((QUERY_HEADS, rows * HEAD_DIM), stride=(rows * HEAD_DIM, 1)),
    )
    dk_accum = cute.make_tensor(
        assume_pointer_aligned(post_dk_accum, 16).iterator,
        cute.make_layout((KV_HEADS, rows * HEAD_DIM), stride=(rows * HEAD_DIM, 1)),
    )
    dv_accum = cute.make_tensor(
        assume_pointer_aligned(post_dv_accum, 16).iterator,
        cute.make_layout((KV_HEADS, rows * HEAD_DIM), stride=(rows * HEAD_DIM, 1)),
    )
    dq_view = cute.make_tensor(
        assume_pointer_aligned(post_dq_out, 16).iterator,
        cute.make_layout(
            (tokens, QUERY_HEADS, HEAD_DIM),
            stride=(QUERY_HEADS * HEAD_DIM, HEAD_DIM, 1),
        ),
    )
    dk_view = cute.make_tensor(
        assume_pointer_aligned(post_dk_out, 16).iterator,
        cute.make_layout(
            (tokens, KV_HEADS, HEAD_DIM), stride=(KV_HEADS * HEAD_DIM, HEAD_DIM, 1)
        ),
    )
    dv_view = cute.make_tensor(
        assume_pointer_aligned(post_dv_out, 16).iterator,
        cute.make_layout(
            (tokens, KV_HEADS, HEAD_DIM), stride=(KV_HEADS * HEAD_DIM, HEAD_DIM, 1)
        ),
    )
    # dV also goes to the V columns of the packed dQKV rows. Like the other outputs, the view
    # spans every layer slot from slot 0's first row, because the postprocess addresses rows by
    # packed token index.
    packed_v_pointer = cute.make_ptr(
        dtype=post_packed_dqkv.element_type,
        value=(
            post_packed_dqkv.iterator.toint()
            + Int64(2 * V_OFFSET_ELEMENTS)
        ),
        mem_space=post_packed_dqkv.iterator.memspace,
        assumed_align=16,
    )
    dv_mirror_view = cute.make_tensor(
        packed_v_pointer,
        cute.make_layout(
            (tokens, KV_HEADS, HEAD_DIM),
            stride=(QKV_HIDDEN, HEAD_DIM, 1),
        ),
    )
    attention_backward_postprocess(
        self.backward.postprocess_dq, post_dq_args,
        self.backward.postprocess_dkv, post_dkv_args,
        control,
        dq_accum, dk_accum, dv_accum,
        dq_view, dk_view, dv_view, dv_mirror_view,
        cu_seqlens, fa4_args[BWD_ARG_SOFTMAX_SCALE],
        gen_post, role,
    )



@dataclass
class AttentionTensors:
    """The attention tensors of every generation, the backward workspaces and the control table.

    `q`, `k`, `v`, `out`, `dout`, `dq`, `dk` and `dv` hold `generations * SEQUENCE` packed
    tokens; `decoder_layer`'s layer-slot slabs are views of them. The workspaces (`lse_log2`,
    `dpsum` and the FP32 accumulators) have `workspace_rows` rows per head.
    """

    plan: AttentionPlan
    workspace_rows: int
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    lse: torch.Tensor
    dout: torch.Tensor
    lse_log2: torch.Tensor
    dpsum: torch.Tensor
    dq_accum: torch.Tensor
    dk_accum: torch.Tensor
    dv_accum: torch.Tensor
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor
    cu_seqlens: torch.Tensor
    control: torch.Tensor
    phase_counter: torch.Tensor
    generation_slot: torch.Tensor

    @staticmethod
    def workspace_bound(plan: AttentionPlan) -> int:
        """Workspace rows the slots need, `slot_count * SLOT_ROWS`.

        SLOT_ROWS bounds any generation's span, so the bound does not depend on the plan's
        document lengths.
        """

        return plan.slot_count * SLOT_ROWS

    @classmethod
    def allocate(
        cls, plan: AttentionPlan, device: torch.device, *, canary: int = 4 * DKV_POSTPROCESS_TILE_M
    ) -> "AttentionTensors":
        capacity = plan.generations
        tokens = capacity * SEQUENCE
        plan.check_slot_layout()
        bound = cls.workspace_bound(plan)
        rows = bound + canary            # spare rows past the slots; no phase writes them
        bf16 = {"device": device, "dtype": torch.bfloat16}
        fp32 = {"device": device, "dtype": torch.float32}

        q = torch.empty(tokens, QUERY_HEADS, HEAD_DIM, **bf16)
        k = torch.empty(tokens, KV_HEADS, HEAD_DIM, **bf16)
        return cls(
            plan=plan,
            workspace_rows=rows,
            q=q,
            k=k,
            v=torch.zeros_like(k),   # FA4 may read V rows not produced yet; they must be finite
            out=torch.empty_like(q),
            lse=torch.empty(QUERY_HEADS, tokens, **fp32),
            dout=torch.empty_like(q),
            lse_log2=torch.empty(QUERY_HEADS, rows, **fp32),
            dpsum=torch.empty(QUERY_HEADS, rows, **fp32),
            dq_accum=torch.empty(QUERY_HEADS, rows * HEAD_DIM, **fp32),
            dk_accum=torch.empty(KV_HEADS, rows * HEAD_DIM, **fp32),
            dv_accum=torch.empty(KV_HEADS, rows * HEAD_DIM, **fp32),
            dq=torch.empty_like(q),
            dk=torch.empty_like(k),
            dv=torch.empty_like(k),
            cu_seqlens=torch.tensor(plan.cu_seqlens, device=device, dtype=torch.int32),
            control=torch.zeros(CONTROL_WORDS, device=device, dtype=torch.int32),
            phase_counter=torch.zeros(1, device=device, dtype=torch.int32),
            generation_slot=torch.zeros(4, device=device, dtype=torch.int32),
        )

    def stage_control_plane(self) -> None:
        """Writes the control table from the plan.

        It stages the document lengths, the per-generation tables, the slot table, the
        postprocess shifts and the balanced forward schedule. The route word starts at 0; the
        program writes the current layer slot into it before each layer's attention.
        """

        plan = self.plan
        config = torch.zeros(CONTROL_WORDS, dtype=torch.int32)
        cu = plan.cu_seqlens

        for entry, length in enumerate(plan.entry_lengths):
            config[CONTROL_SEQUSED + entry] = length
            # The slot table. With 64 x 64 backward tiles the Q-side and K-side padded offsets
            # are equal; SlotSeqlenInfoQK reads each side from its own region.
            config[CONTROL_Q_PADDED_BASE + entry] = plan.slot_padded_offset(entry)
            config[CONTROL_K_PADDED_BASE + entry] = plan.slot_padded_offset(entry)
            # The postprocess shifts: each task body computes FA4's padded offset from `cu` at
            # its own tile size, and the shift moves that to the entry's slot-table value.
            config[CONTROL_DQ_POSTPROCESS_SHIFT + entry] = (
                plan.slot_padded_offset(entry) - padded_offset(cu, entry, BWD_BLOCK_ROWS)
            )
            config[CONTROL_DKV_POSTPROCESS_SHIFT + entry] = (
                plan.slot_padded_offset(entry) - padded_offset(cu, entry, DKV_POSTPROCESS_TILE_M)
            )

        for generation in range(plan.generations):
            config[CONTROL_ENTRY_BASE + generation] = plan.entry_base[generation]
            config[CONTROL_DOCUMENT_COUNT + generation] = len(plan.document_lengths[generation])
            # The rows `clear_dkv_accumulators` zeroes: the generation's span from its slot's
            # base. The span fits the slot (check_slot_layout), so the clear stays inside it.
            config[CONTROL_DKV_SPAN_BASE + generation] = plan.slot_row_base(generation)
            config[CONTROL_DKV_SPAN_ROWS + generation] = plan.generation_span(generation)
            for slot, value in enumerate(plan.cumulative_blocks(generation, FWD_BLOCK_TOKENS)):
                config[CONTROL_FWD_CUMULATIVE_BLOCKS + generation * CUMULATIVE_EDGES + slot] = value
            for slot, value in enumerate(plan.cumulative_blocks(generation, BWD_BLOCK_ROWS)):
                config[CONTROL_BWD_CUMULATIVE_BLOCKS + generation * CUMULATIVE_EDGES + slot] = value
            for slot, value in enumerate(
                plan.cumulative_blocks(generation, DKV_POSTPROCESS_TILE_M)
            ):
                config[CONTROL_DKV_CUMULATIVE_BLOCKS + generation * CUMULATIVE_EDGES + slot] = value

        # Per generation, visit ordinal -> tile ordinal for AttentionForwardScheduler: a
        # longest-first assignment of the causal tiles to the CTAs' visits.
        for generation, lengths in enumerate(plan.document_lengths):
            schedule = build_lpt_schedule(
                lengths,
                kv_heads=KV_HEADS,
                tile_m=FWD_BLOCK_TOKENS,
                tile_n=BACKWARD_TILE_N,
                grid_ctas=PROGRAM_CTAS,
            )
            if len(schedule) > MAX_FWD_TASKS_PER_GENERATION:
                raise AssertionError("forward schedule exceeds its derived generation stride")
            base = CONTROL_FWD_SCHEDULE + generation * MAX_FWD_TASKS_PER_GENERATION
            config[base : base + len(schedule)] = torch.tensor(schedule, dtype=torch.int32)

        config[CONTROL_ROUTE] = 0
        self.control.copy_(config.to(self.control.device))


# CuTe views of the host tensors, for the program's launch arguments.


def dynamic_cute_tensor(tensor: torch.Tensor, alignment: int = 4):
    """A fully dynamic view, which `cu_seqlens` and the control table need.

    `from_dlpack` would bake `shape[0]` into the TVM-FFI signature, and FA4 derives
    `num_batch = cu_seqlens.shape[0] - 1` from it, so a static binding would reject a plan
    with a different number of documents.
    """

    from flash_attn.cute.cute_dsl_utils import to_cute_tensor  # noqa: TID253

    return to_cute_tensor(tensor.detach(), assumed_align=alignment, fully_dynamic=True)


def static_cute_tensor(tensor: torch.Tensor, alignment: int = 16):
    return from_dlpack(tensor.detach(), assumed_align=alignment, enable_tvm_ffi=True)


def ragged_cute_tensor(tensor: torch.Tensor):
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor  # noqa: TID253

    return to_cute_tensor(
        tensor.detach(), assumed_align=16, leading_dim=tensor.ndim - 1, fully_dynamic=False
    )
