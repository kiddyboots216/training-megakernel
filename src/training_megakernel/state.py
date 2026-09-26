"""Tensor ownership and launch tuples for one bundle's depth and sequence."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from . import schedule
from .geometry import Geometry
from .layout import (
    BWD_TILE_ROWS,
    CUM_SLOTS,
    FWD_TILE_ROWS,
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    KV_HEADS,
    KV_HIDDEN,
    KV_TILE_ROWS,
    MAX_DOCS,
    NORM_PARTIAL_BLOCK_ROWS,
    PAD_TILE,
    PHYSICAL_SLOTS,
    PRE_TILE_ROWS,
    PROGRAM_CTAS,
    PROGRAM_THREADS,  # noqa: F401 - retained state-module compatibility export
    QKV_HIDDEN,
    QUERY_HEADS,
    QUERY_HIDDEN,
    ROW_SLOT,
    ROW_TABLE_FIELDS,
    ROW_TABLE_SLOTS,
    TWO_I,
    VOCAB,
    control_layout,
    families,
)
from .shape import HEAD_CHUNKS


def incoming_gradient_slot(layer: int, depth: int) -> int:
    """Ring slot consumed by decoder layer ``layer``."""

    if not 0 <= layer < depth:
        raise ValueError("logical layer is outside the depth")
    return layer % PHYSICAL_SLOTS


def outgoing_gradient_slot(layer: int, depth: int) -> int:
    """Ring slot produced for the next lower decoder layer."""

    if not 0 <= layer < depth:
        raise ValueError("logical layer is outside the depth")
    return (layer - 1) % PHYSICAL_SLOTS


def _blocks(length: int, rows: int) -> int:
    return math.ceil(length / rows)


def _padded_offset(cu: list[int], entry: int, tile: int = PAD_TILE) -> int:
    return (cu[entry] + entry * tile) // tile * tile


@dataclass(frozen=True)
class PackedPlan:
    """Two physical FA4 slots carrying one identical packed document plan."""

    document_lengths: tuple[tuple[int, ...], tuple[int, ...]]
    geometry: Geometry

    @classmethod
    def from_documents(cls, lengths: Iterable[int], geometry: Geometry) -> PackedPlan:
        row = tuple(int(length) for length in lengths)
        if not row or len(row) > MAX_DOCS or any(length <= 0 for length in row):
            raise ValueError("one to four positive document lengths are required")
        if sum(row) != geometry.sequence:
            raise ValueError(f"documents must cover exactly {geometry.sequence} rows")
        return cls((row, row), geometry)

    @property
    def generations(self) -> int:
        return PHYSICAL_SLOTS

    @property
    def entries(self) -> list[int]:
        return [length for row in self.document_lengths for length in row]

    @property
    def entry_base(self) -> list[int]:
        return [0, len(self.document_lengths[0])]

    @property
    def gen_of_entry(self) -> list[int]:
        return [0] * len(self.document_lengths[0]) + [1] * len(self.document_lengths[1])

    @property
    def cu(self) -> list[int]:
        result = [0]
        for length in self.entries:
            result.append(result[-1] + length)
        return result

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    def entry_of(self, generation: int, document: int) -> int:
        return self.entry_base[generation] + document

    def cumulative_blocks(self, generation: int, rows: int) -> list[int]:
        result = [0]
        for length in self.document_lengths[generation]:
            result.append(result[-1] + _blocks(length, rows))
        result.extend([result[-1]] * (CUM_SLOTS - len(result)))
        return result

    def entry_rows(self, entry: int) -> int:
        length = self.entries[entry]
        return max(
            _blocks(length, PAD_TILE) * PAD_TILE,
            _blocks(length, KV_TILE_ROWS) * KV_TILE_ROWS,
        )

    def generation_pad_base(self, generation: int) -> int:
        return _padded_offset(self.cu, self.entry_of(generation, 0))

    def generation_span(self, generation: int) -> int:
        last = self.entry_of(generation, len(self.document_lengths[generation]) - 1)
        return (
            _padded_offset(self.cu, last)
            + self.entry_rows(last)
            - self.generation_pad_base(generation)
        )

    def slot_pad(self, entry: int) -> int:
        generation = self.gen_of_entry[entry]
        relative = _padded_offset(self.cu, entry) - self.generation_pad_base(generation)
        return generation * self.geometry.slot_rows + relative

    def validate(self) -> None:
        slot_rows = self.geometry.slot_rows
        spans = [self.generation_span(generation) for generation in range(PHYSICAL_SLOTS)]
        reach = max(
            self.slot_pad(entry) + self.entry_rows(entry) for entry in range(self.entry_count)
        )
        if max(spans) > slot_rows or reach > PHYSICAL_SLOTS * slot_rows:
            raise ValueError("packed documents exceed the two-slot workspace")


@dataclass
class AttentionState:
    plan: PackedPlan
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

    @classmethod
    def allocate(cls, plan: PackedPlan, device: torch.device) -> AttentionState:
        plan.validate()
        geometry = plan.geometry
        tokens = PHYSICAL_SLOTS * geometry.sequence
        workspace_rows = PHYSICAL_SLOTS * geometry.slot_rows + 4 * KV_TILE_ROWS
        q = torch.empty(tokens, QUERY_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        k = torch.empty(tokens, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        state = cls(
            plan=plan,
            q=q,
            k=k,
            v=torch.zeros_like(k),
            out=torch.empty_like(q),
            lse=torch.empty(QUERY_HEADS, tokens, dtype=torch.float32, device=device),
            dout=torch.empty_like(q),
            lse_log2=torch.empty(QUERY_HEADS, workspace_rows, dtype=torch.float32, device=device),
            dpsum=torch.empty(QUERY_HEADS, workspace_rows, dtype=torch.float32, device=device),
            dq_accum=torch.empty(
                QUERY_HEADS, workspace_rows * HEAD_DIM, dtype=torch.float32, device=device
            ),
            dk_accum=torch.empty(
                KV_HEADS, workspace_rows * HEAD_DIM, dtype=torch.float32, device=device
            ),
            dv_accum=torch.empty(
                KV_HEADS, workspace_rows * HEAD_DIM, dtype=torch.float32, device=device
            ),
            dq=torch.empty_like(q),
            dk=torch.empty_like(k),
            dv=torch.empty_like(k),
            cu_seqlens=torch.tensor(plan.cu, dtype=torch.int32, device=device),
            control=torch.zeros(geometry.control_elements, dtype=torch.int32, device=device),
            phase_counter=torch.zeros(1, dtype=torch.int32, device=device),
            # One forward and one reverse visit per layer.
            generation_slot=torch.zeros(4, dtype=torch.int32, device=device),
        )
        state.configure_forward_schedule()
        return state

    def configure_forward_schedule(self) -> None:
        self.control.copy_(self.control_for(self.plan).to(self.control.device))

    @staticmethod
    def control_for(plan: PackedPlan) -> torch.Tensor:
        geometry = plan.geometry
        layout = control_layout(geometry.depth, geometry.sequence)
        config = torch.zeros(layout.control_elems, dtype=torch.int32)
        for entry, length in enumerate(plan.entries):
            config[layout.seqused + entry] = length
            config[layout.padq + entry] = plan.slot_pad(entry)
            config[layout.padk + entry] = plan.slot_pad(entry)
            config[layout.dqsh + entry] = plan.slot_pad(entry) - _padded_offset(plan.cu, entry)
            config[layout.kvsh + entry] = plan.slot_pad(entry) - _padded_offset(
                plan.cu, entry, KV_TILE_ROWS
            )
        for generation in range(PHYSICAL_SLOTS):
            config[layout.ebase + generation] = plan.entry_base[generation]
            config[layout.ndocs + generation] = len(plan.document_lengths[generation])
            config[layout.kbase + generation] = generation * geometry.slot_rows
            config[layout.krows + generation] = plan.generation_span(generation)
            for slot, value in enumerate(plan.cumulative_blocks(generation, FWD_TILE_ROWS)):
                config[layout.fwdcum + generation * CUM_SLOTS + slot] = value
            for slot, value in enumerate(plan.cumulative_blocks(generation, PRE_TILE_ROWS)):
                config[layout.precum + generation * CUM_SLOTS + slot] = value
            for slot, value in enumerate(plan.cumulative_blocks(generation, KV_TILE_ROWS)):
                config[layout.kvcum + generation * CUM_SLOTS + slot] = value
            permutation = schedule.build_lpt_schedule(
                plan.document_lengths[generation],
                kv_heads=KV_HEADS,
                tile_m=FWD_TILE_ROWS,
                tile_n=BWD_TILE_ROWS,
                grid_ctas=PROGRAM_CTAS,
            )
            if len(permutation) > layout.max_gen_fwd_tasks:
                raise AssertionError("forward schedule exceeds the generation stride")
            begin = layout.fwdsched + generation * layout.max_gen_fwd_tasks
            config[begin : begin + len(permutation)] = torch.tensor(permutation, dtype=torch.int32)
        config[layout.route] = 0
        return config

    def reset_metadata(self) -> None:
        self.phase_counter.zero_()
        self.generation_slot.zero_()


@dataclass(frozen=True)
class Slab:
    shape: tuple[int, ...]
    dtype: torch.dtype
    staged: bool = False
    alias: str | None = None
    shared: bool = False


def slab_specs(sequence: int) -> dict[str, Slab]:
    """Every decoder slab at S tokens per GPU; the program allocates the same."""

    norm_blocks = sequence // NORM_PARTIAL_BLOCK_ROWS
    return {
        "residual_in": Slab((sequence, HIDDEN), torch.bfloat16, staged=True),
        "staging_dy": Slab((sequence, HIDDEN), torch.bfloat16, staged=True),
        "input_norm": Slab((HIDDEN,), torch.bfloat16, staged=True),
        "qkv_weight": Slab((QKV_HIDDEN, HIDDEN), torch.bfloat16, staged=True),
        "q_norm": Slab((HEAD_DIM,), torch.bfloat16, staged=True),
        "k_norm": Slab((HEAD_DIM,), torch.bfloat16, staged=True),
        "o_weight": Slab((HIDDEN, HIDDEN), torch.bfloat16, staged=True),
        "post_attention_norm": Slab((HIDDEN,), torch.bfloat16, staged=True),
        "gate_up_weight": Slab((TWO_I, HIDDEN), torch.bfloat16, staged=True),
        "down_weight": Slab((HIDDEN, INTERMEDIATE), torch.bfloat16, staged=True),
        "rotary_cos": Slab((sequence, HEAD_DIM), torch.bfloat16, staged=True, shared=True),
        "rotary_sin": Slab((sequence, HEAD_DIM), torch.bfloat16, staged=True, shared=True),
        "norm1": Slab((sequence, HIDDEN), torch.bfloat16),
        "rms1_rstd": Slab((sequence,), torch.float32),
        "qkv_raw": Slab((sequence, QKV_HIDDEN), torch.bfloat16),
        "v": Slab((sequence, KV_HIDDEN), torch.bfloat16, alias="v"),
        "q_rotary": Slab((sequence, QUERY_HIDDEN), torch.bfloat16, alias="q"),
        "k_rotary": Slab((sequence, KV_HIDDEN), torch.bfloat16, alias="k"),
        "attention_out": Slab((sequence, HIDDEN), torch.bfloat16, alias="out"),
        "residual_mid": Slab((sequence, HIDDEN), torch.bfloat16),
        "norm2": Slab((sequence, HIDDEN), torch.bfloat16),
        "rms2_rstd": Slab((sequence,), torch.float32),
        "gate_up": Slab((sequence, TWO_I), torch.bfloat16),
        "mlp_intermediate": Slab((sequence, INTERMEDIATE), torch.bfloat16),
        "layer_output": Slab((sequence, HIDDEN), torch.bfloat16),
        "layer_output_dy": Slab((sequence, HIDDEN), torch.bfloat16),
        "mlp_intermediate_dy": Slab((sequence, INTERMEDIATE), torch.bfloat16),
        "gate_up_dy": Slab((sequence, TWO_I), torch.bfloat16),
        "norm2_dy": Slab((sequence, HIDDEN), torch.bfloat16),
        "residual_mid_dy": Slab((sequence, HIDDEN), torch.bfloat16),
        "attention_dout": Slab((sequence, HIDDEN), torch.bfloat16, alias="dout"),
        "dq_rotary": Slab((sequence, QUERY_HIDDEN), torch.bfloat16, alias="dq"),
        "dk_rotary": Slab((sequence, KV_HIDDEN), torch.bfloat16, alias="dk"),
        "dv": Slab((sequence, KV_HIDDEN), torch.bfloat16, alias="dv"),
        "dqkv_raw": Slab((sequence, QKV_HIDDEN), torch.bfloat16),
        "norm1_dy": Slab((sequence, HIDDEN), torch.bfloat16),
        "layer_input_dx": Slab((sequence, HIDDEN), torch.bfloat16),
        "input_norm_partial": Slab((2 * PROGRAM_CTAS, HIDDEN), torch.float32),
        "post_attn_norm_partial": Slab((2 * PROGRAM_CTAS, HIDDEN), torch.float32),
        "q_norm_partial": Slab((QUERY_HEADS * norm_blocks, HEAD_DIM), torch.float32),
        "k_norm_partial": Slab((KV_HEADS * norm_blocks, HEAD_DIM), torch.float32),
        "input_norm_grad": Slab((HIDDEN,), torch.float32),
        "post_attn_norm_grad": Slab((HIDDEN,), torch.float32),
        "q_norm_grad": Slab((HEAD_DIM,), torch.float32),
        "k_norm_grad": Slab((HEAD_DIM,), torch.float32),
        "qkv_grad": Slab((QKV_HIDDEN, HIDDEN), torch.float32),
        "o_grad": Slab((HIDDEN, HIDDEN), torch.float32),
        "gate_up_grad": Slab((TWO_I, HIDDEN), torch.float32),
        "down_grad": Slab((HIDDEN, INTERMEDIATE), torch.float32),
    }


TRANSPOSE_VIEWS = {
    "qkv_weight_t": "qkv_weight",
    "o_weight_t": "o_weight",
    "gate_up_weight_t": "gate_up_weight",
    "down_weight_t": "down_weight",
}


@dataclass
class ShellState:
    final_norm_weight: torch.Tensor
    final_normalized_hidden: torch.Tensor
    final_rstd: torch.Tensor
    head_weight: torch.Tensor
    labels: torch.Tensor
    global_valid_tokens: torch.Tensor
    dlogits_slab: torch.Tensor
    per_token_loss: torch.Tensor
    loss: torch.Tensor
    head_dhidden_fp32: torch.Tensor
    head_dweight: torch.Tensor
    head_dweight_handle: object | None
    head_dweight_multicast_base: int
    task_records: torch.Tensor
    active_chunks: torch.Tensor
    generation_slot: torch.Tensor
    final_dnorm_bf16: torch.Tensor
    final_norm_partial: torch.Tensor
    final_norm_grad: torch.Tensor
    fa4_checkpoint_extra0: torch.Tensor
    fa4_checkpoint_extra1: torch.Tensor
    residual_mid_checkpoint: torch.Tensor
    geometry: Geometry

    @classmethod
    def allocate(cls, device: torch.device, geometry: Geometry) -> ShellState:
        sequence = geometry.sequence
        plan = geometry.checkpoint_plan
        head_dweight = torch.empty(VOCAB, HIDDEN, dtype=torch.float32, device=device)
        head_dweight_handle = None
        head_dweight_multicast_base = 0
        if device.type != "meta":
            import torch.distributed as dist

            if dist.is_initialized() and dist.get_world_size() == 8:
                from torch.distributed import _symmetric_memory as symm

                head_dweight = symm.empty(VOCAB * HIDDEN, dtype=torch.float32, device=device).view(
                    VOCAB, HIDDEN
                )
                head_dweight_handle = symm.rendezvous(head_dweight, dist.group.WORLD)
                head_dweight_multicast_base = int(
                    getattr(head_dweight_handle, "multicast_ptr", 0) or 0
                )
                if head_dweight_multicast_base == 0:
                    raise RuntimeError("head-dWeight NVLS multicast mapping is unavailable")
        # The checkpoint banks are allocated after the rest of the state, in
        # attach_checkpoint_storage, the way the build allocates them.
        checkpoint_sizes = (
            (plan.bank0_elements, plan.bank1_elements, plan.residual_storage_elements)
            if device.type == "meta"
            else (1, 1, 1)
        )
        return cls(
            final_norm_weight=torch.empty(HIDDEN, dtype=torch.bfloat16, device=device),
            final_normalized_hidden=torch.empty(
                sequence, HIDDEN, dtype=torch.bfloat16, device=device
            ),
            final_rstd=torch.empty(sequence, dtype=torch.float32, device=device),
            head_weight=torch.empty(VOCAB, HIDDEN, dtype=torch.bfloat16, device=device),
            labels=torch.empty(sequence, dtype=torch.int32, device=device),
            global_valid_tokens=torch.empty(1, dtype=torch.int32, device=device),
            dlogits_slab=torch.empty(sequence, VOCAB, dtype=torch.bfloat16, device=device),
            per_token_loss=torch.empty(sequence, dtype=torch.float32, device=device),
            loss=torch.empty(1, dtype=torch.float32, device=device),
            head_dhidden_fp32=torch.empty(sequence, HIDDEN, dtype=torch.float32, device=device),
            head_dweight=head_dweight,
            head_dweight_handle=head_dweight_handle,
            head_dweight_multicast_base=head_dweight_multicast_base,
            task_records=torch.arange(HEAD_CHUNKS, dtype=torch.int32, device=device),
            active_chunks=torch.tensor([HEAD_CHUNKS], dtype=torch.int32, device=device),
            generation_slot=torch.zeros(1, dtype=torch.int32, device=device),
            final_dnorm_bf16=torch.empty(sequence, HIDDEN, dtype=torch.bfloat16, device=device),
            final_norm_partial=torch.empty(
                2 * PROGRAM_CTAS, HIDDEN, dtype=torch.float32, device=device
            ),
            final_norm_grad=torch.empty(HIDDEN, dtype=torch.float32, device=device),
            fa4_checkpoint_extra0=torch.empty(
                checkpoint_sizes[0], dtype=torch.bfloat16, device=device
            ),
            fa4_checkpoint_extra1=torch.empty(
                checkpoint_sizes[1], dtype=torch.bfloat16, device=device
            ),
            residual_mid_checkpoint=torch.empty(
                checkpoint_sizes[2], dtype=torch.bfloat16, device=device
            ),
            geometry=geometry,
        )

    def attach_checkpoint_storage(self, residual_routes: torch.Tensor) -> None:
        if self.final_norm_weight.device.type == "meta":
            return
        device = self.final_norm_weight.device
        plan = self.geometry.checkpoint_plan
        self.fa4_checkpoint_extra0 = torch.empty(
            plan.bank0_elements, dtype=torch.bfloat16, device=device
        )
        self.fa4_checkpoint_extra1 = torch.empty(
            plan.bank1_elements, dtype=torch.bfloat16, device=device
        )
        self.residual_mid_checkpoint = torch.empty(
            plan.residual_storage_elements, dtype=torch.bfloat16, device=device
        )
        # The two ordinary residual_mid addresses follow the residual records.
        route_words = self.residual_mid_checkpoint.view(torch.int64)[
            plan.residual_bank_bytes // 8 :
        ]
        if route_words.numel() != 2:
            raise AssertionError("residual checkpoint route table extent drifted")
        route_words.copy_(
            torch.tensor(
                [residual_routes[0].data_ptr(), residual_routes[1].data_ptr()],
                dtype=torch.int64,
                device=device,
            )
        )

    def runtime_tensors(self) -> tuple[torch.Tensor, ...]:
        chunk_rows = self.geometry.head_chunk_rows
        hidden_chunks = self.final_normalized_hidden.view(HEAD_CHUNKS, chunk_rows, HIDDEN).permute(
            1, 2, 0
        )
        logits_broadcast = self.dlogits_slab.view(HEAD_CHUNKS, chunk_rows, VOCAB).permute(1, 2, 0)
        return (
            self.final_norm_weight,
            self.final_normalized_hidden,
            self.final_rstd,
            hidden_chunks,
            self.head_weight.unsqueeze(-1).expand(VOCAB, HIDDEN, HEAD_CHUNKS),
            logits_broadcast,
            self.dlogits_slab.unsqueeze(-1),
            self.head_weight.mT.unsqueeze(-1),
            self.head_dhidden_fp32.unsqueeze(-1),
            self.dlogits_slab.mT.unsqueeze(-1),
            self.final_normalized_hidden.mT.unsqueeze(-1),
            self.head_dweight.unsqueeze(-1),
            self.labels,
            self.global_valid_tokens,
            self.per_token_loss,
            self.loss,
            self.task_records,
            self.active_chunks,
            self.generation_slot,
            self.final_dnorm_bf16,
            self.final_norm_partial,
            self.final_norm_grad,
            self.fa4_checkpoint_extra0,
            self.fa4_checkpoint_extra1,
            self.residual_mid_checkpoint,
        )


@dataclass
class ModelState:
    attention: AttentionState
    slabs: dict[str, torch.Tensor]
    activation_chain: torch.Tensor
    gradient_ring: torch.Tensor
    row_table: torch.Tensor
    shell: ShellState
    document_lengths: tuple[int, ...]
    rotary_segment_extents: tuple[int, ...]
    geometry: Geometry

    @classmethod
    def allocate(
        cls,
        document_lengths: Iterable[int],
        device: torch.device,
        *,
        geometry: Geometry,
        rotary_segment_extents: Iterable[int] | None = None,
    ) -> ModelState:
        sequence, depth = geometry.sequence, geometry.depth
        documents = tuple(int(length) for length in document_lengths)
        rotary_segments = (
            documents
            if rotary_segment_extents is None
            else tuple(int(length) for length in rotary_segment_extents)
        )
        if not rotary_segments or any(length <= 0 for length in rotary_segments):
            raise ValueError("rotary segments must have positive extents")
        if sum(rotary_segments) != sequence:
            raise ValueError(f"rotary segments must cover exactly {sequence} rows")
        plan = PackedPlan.from_documents(documents, geometry)
        attention = AttentionState.allocate(plan, device)
        specs = slab_specs(sequence)
        slabs: dict[str, torch.Tensor] = {}
        for name, slab in specs.items():
            if slab.alias:
                source = getattr(attention, slab.alias)
                slabs[name] = source.view(PHYSICAL_SLOTS, sequence, -1)
            elif slab.shared:
                slabs[name] = torch.empty(slab.shape, dtype=slab.dtype, device=device)
            else:
                slabs[name] = torch.empty(
                    (PHYSICAL_SLOTS, *slab.shape), dtype=slab.dtype, device=device
                )
        for view_name, source_name in TRANSPOSE_VIEWS.items():
            slabs[view_name] = slabs[source_name].mT

        activation_chain = torch.empty(
            depth + 1, sequence, HIDDEN, dtype=torch.bfloat16, device=device
        )
        gradient_ring = torch.empty(
            PHYSICAL_SLOTS, sequence, HIDDEN, dtype=torch.bfloat16, device=device
        )
        projection_dy_slots = slabs["layer_output_dy"]
        slabs["layer_output"] = projection_dy_slots
        slabs["layer_output_dy"] = gradient_ring
        slabs["residual_in"] = activation_chain[:PHYSICAL_SLOTS]
        slabs["staging_dy"] = gradient_ring
        slabs["layer_input_dx"] = gradient_ring

        address_rows: list[int] = []
        for layer in range(depth):
            physical = layer % PHYSICAL_SLOTS
            for name in ROW_TABLE_FIELDS:
                if name == "residual_in":
                    target = activation_chain[layer]
                elif name == "layer_output":
                    target = activation_chain[layer + 1]
                elif name == "staging_dy":
                    target = gradient_ring[incoming_gradient_slot(layer, depth)]
                elif name == "layer_output_dy":
                    target = projection_dy_slots[physical]
                elif name == "layer_input_dx":
                    target = gradient_ring[outgoing_gradient_slot(layer, depth)]
                else:
                    slab = specs[name]
                    target = slabs[name] if slab.shared else slabs[name][physical]
                if target.data_ptr() % 16:
                    raise AssertionError(f"unaligned row-table target {layer}:{name}")
                address_rows.append(target.data_ptr())
        result = cls(
            attention=attention,
            slabs=slabs,
            activation_chain=activation_chain,
            gradient_ring=gradient_ring,
            row_table=torch.tensor(address_rows, dtype=torch.int64, device=device),
            shell=ShellState.allocate(device, geometry),
            document_lengths=documents,
            rotary_segment_extents=rotary_segments,
            geometry=geometry,
        )
        if device.type != "meta":
            result.validate_physical_aliases()
        return result

    def rebuild_row_table(self) -> None:
        depth = self.geometry.depth
        specs = slab_specs(self.geometry.sequence)
        addresses: list[int] = []
        for layer in range(depth):
            physical = layer % PHYSICAL_SLOTS
            for name in ROW_TABLE_FIELDS:
                if name == "residual_in":
                    target = self.activation_chain[layer]
                elif name == "layer_output":
                    target = self.activation_chain[layer + 1]
                elif name == "staging_dy":
                    target = self.gradient_ring[incoming_gradient_slot(layer, depth)]
                elif name == "layer_output_dy":
                    target = self.slabs["layer_output"][physical]
                elif name == "layer_input_dx":
                    target = self.gradient_ring[outgoing_gradient_slot(layer, depth)]
                else:
                    slab = specs[name]
                    target = self.slabs[name] if slab.shared else self.slabs[name][physical]
                addresses.append(target.data_ptr())
        self.row_table.copy_(
            torch.tensor(addresses, dtype=torch.int64, device=self.row_table.device)
        )
        if self.row_table.device.type != "meta":
            self.validate_physical_aliases()

    def validate_physical_aliases(self) -> None:
        """Enforce the donor's projection surface and two-slot dY ring law."""

        if self.slabs["layer_output_dy"].data_ptr() != self.gradient_ring.data_ptr():
            raise AssertionError("down-GEMM incoming-dY ring alias is broken")
        if self.slabs["layer_output"].data_ptr() == self.gradient_ring.data_ptr():
            raise AssertionError("down-forward projection storage was lost")
        depth = self.geometry.depth
        rows = self.row_table.view(depth, ROW_TABLE_SLOTS).cpu().tolist()
        for layer in range(depth):
            physical = layer % PHYSICAL_SLOTS
            expected = {
                "residual_in": self.activation_chain[layer],
                "layer_output": self.activation_chain[layer + 1],
                "staging_dy": self.gradient_ring[incoming_gradient_slot(layer, depth)],
                "layer_output_dy": self.slabs["layer_output"][physical],
                "layer_input_dx": self.gradient_ring[outgoing_gradient_slot(layer, depth)],
            }
            for name, target in expected.items():
                observed = int(rows[layer][ROW_SLOT[name]])
                if observed != target.data_ptr():
                    raise AssertionError(f"logical row route broken at {layer}:{name}")
            if layer + 1 < depth and outgoing_gradient_slot(
                layer + 1, depth
            ) != incoming_gradient_slot(layer, depth):
                raise AssertionError(f"reverse gradient edge broken at {layer + 1}")

    def initialize_rotary(self, theta: float = 1e6) -> None:
        positions = torch.cat(
            [
                torch.arange(length, dtype=torch.float32, device=self.row_table.device)
                for length in self.rotary_segment_extents
            ]
        )
        inverse = 1.0 / (
            theta
            ** (
                torch.arange(
                    0,
                    HEAD_DIM,
                    2,
                    dtype=torch.float32,
                    device=self.row_table.device,
                )
                / HEAD_DIM
            )
        )
        frequencies = torch.outer(positions, inverse)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        self.slabs["rotary_cos"].copy_(embedding.cos().to(torch.bfloat16))
        self.slabs["rotary_sin"].copy_(embedding.sin().to(torch.bfloat16))

    def runtime_prefix_suffix(self) -> tuple[tuple[object, ...], tuple[object, ...]]:
        t = self.attention
        scale = HEAD_DIM**-0.5
        family_arguments: list[torch.Tensor] = []
        for family in families(self.geometry.sequence):
            if family.name == "down_dw":
                family_arguments.extend(
                    (
                        self.slabs[family.b].permute(2, 1, 0),
                        self.slabs[family.a].permute(2, 1, 0),
                        self.slabs[family.d].permute(2, 1, 0),
                    )
                )
            else:
                a = self.slabs[family.a]
                family_arguments.extend(
                    (
                        a.permute(2, 1, 0) if family.a_t else a.permute(1, 2, 0),
                        self.slabs[family.b].permute(2, 1, 0),
                        self.slabs[family.d].permute(1, 2, 0),
                    )
                )
        prefix = (
            t.q,
            t.k,
            t.v,
            t.out,
            t.lse,
            scale,
            t.cu_seqlens,
            t.control,
            t.dout,
            t.lse_log2,
            t.dpsum,
            t.dq_accum,
            t.dk_accum,
            t.dv_accum,
            *family_arguments,
            self.activation_chain[:-1].permute(1, 2, 0),
            self.activation_chain[1:].permute(1, 2, 0),
            self.row_table,
            t.generation_slot,
        )
        suffix = (t.phase_counter, t.dq, t.dk, t.dv)
        if len(prefix) != 54 or len(suffix) != 4:
            raise AssertionError("model ABI drifted")
        return prefix, suffix
