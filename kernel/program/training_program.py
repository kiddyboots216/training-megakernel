"""The training program: its phase methods, the generator of its kernel, and its compile.

``TrainingProgram`` holds the program's members (FlashAttention 4's forward and backward, one
Quack GEMM per decoder projection family, Quack's RMSNorm and ``lm_head.HeadPhase``) and the
``@cute.jit`` phase methods the kernel calls: the decoder weight all-gather and gradient
reduce-scatter, the shell's valid-token all-reduce, embedding gather, head-weight all-gather and
head and embedding gradient reduce-scatters, the final RMSNorm, and the attention and residual
checkpoints.  Its ``__call__`` prepares every member's arguments and launches the kernel.

``_generate_training_kernel`` writes the source of ``training_step_kernel``, one training step
on 132 CTAs x 384 threads: the decoder forward, bottom layer first; the final RMSNorm, LM head
and loss; the decoder backward, top layer first, which recomputes each layer's forward in its
layer slot except what the checkpoints and retained outputs supply; the head and embedding
gradient reduce-scatters; and the optimizer.  ``resident_step`` supplies the step loop around
it.  The source is executed in this module's globals, so every name it uses is bound here.
``training_kernel`` caches the resulting class.  ``kernel/build.py`` calls
``allocate_checkpoint_banks`` and ``store_residual_mid_slot_addresses``, then
``compile_training_program`` with the step loop's operands from ``allocate_step_loop_operands``.
"""

# ruff: noqa: I001

from __future__ import annotations

import functools
import linecache
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from cutlass.cute.runtime import from_dlpack
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.utils import AuxData
from quack.cute_dsl_utils import ParamsBase
from quack.rmsnorm import RMSNorm, RMSNormBackward

import resident_step
from training_megakernel import resident_protocol as RP
from training_megakernel.shape import OWNER_GRADIENT_LAYER_BYTES
from model import SHAPE
from quack_rmsnorm_bodies import (
    resident_rms_backward_body_h4096,
    resident_rms_forward_body_h4096,
)
from attention_members import ATTENTION_BACKWARD_TENSOR_NAMES
from attention_members import FORWARD_TILE_N, AttentionBackwardMember, AttentionForwardMember
from fa4_backward_call import backward_call_setup
from fa4_backward_kernel import (
    backward_setup,
    backward_role,
)
from fa4_forward_call import forward_call_setup
from decoder_gemm_runners import (
    run_initial_gate_up_gemm,
    run_decoder_gemm,
    run_qkv_dx_gemm_without_cta0,
)
from projection_members import make_projection_member
from mlp_projections import (
    make_gate_up_forward_member,
)
from mlp_projections import make_down_dx_member
from mlp_projections import (
    make_down_forward_member,
)
from attention_projections import (
    make_attention_output_forward_member,
)
from attention_projections import (
    make_attention_output_checkpoint_member,
    run_attention_output_checkpoint_gemm,
)
from attention_projections import make_qkv_forward_member
from clipped_adamw import clipped_adamw_step
from communication.memory_ops import (  # noqa: F401 - the step exit is in generated source
    step_exit_branch,
    step_exit_label,
    atomic_cas_gpu_u32,
    copy_b128,
    copy_to_multicast_b128,
    global_timer_ns,
    load_b64,
    load_acquire_sys_u32,
    load_f32,
    load_u64,
    fence_sys,
    multicast_sum_2xf32,
    multicast_sum_f32,
    multicast_min_acquire_u32,
    multicast_store_b64,
    store_b64,
    store_f32,
    store_u64,
    store_relaxed_gpu_u32,
    store_release_sys_u32,
)
from quack_rmsnorm_bodies import (
    RMS_BACKWARD_CONFIG,
    RMS_FORWARD_CONFIG,
    restore_full_page,
    use_rms_shard_page,
)
from program_smem import ProgramSmemAllocator
from gemm_members import PROGRAM_SMEM_PAGE_BYTES
import model
import anchors
from training_megakernel import arenas
import attention
import attention_members
import clipped_adamw
import decoder_layer
import communication.embedding_route as embedding_route
import grid_barrier
import lm_head
import optimizer_state
import qk_norm_rope
import quack_rmsnorm_bodies
import communication.shell_fabric as shell_fabric

if TYPE_CHECKING:
    from training_tensors import TrainingTensors


# The layer's backward phases less dy_ready: the down GEMMs and RMS2 backward read the incoming
# gradient straight from the gradient ring (training_tensors), so there is nothing to stage.
BACKWARD_PHASES = tuple(
    phase
    for phase in decoder_layer.LAYER_PHASES[len(decoder_layer.FORWARD_LAYER_PHASES) :]
    if phase[0] != "dy_ready"
)

WEIGHT_PUBLICATIONS_PER_STEP = 2 * model.DEPTH
DECODER_WEIGHT_PANEL_COUNT = len(model.WEIGHT_PANELS)
DECODER_GRADIENT_SITE_COUNT = len(model.REDUCTION_SITES)

# The launch ABI's `fabric` group: the decoder's weight all-gather and gradient reduce-scatter
# (communication.decoder_fabric.DecoderFabric.runtime_tensors, in this order).
DECODER_FABRIC_ARGUMENT_NAMES = (
    "fabric_weight_arena",
    "fabric_grad_arena",
    "fabric_weight_source_table",
    "fabric_grad_destination_table",
    "fabric_control",
    "fabric_step",
    "fabric_status",
)

# The launch ABI's `optimizer` group (optimizer_state.OptimizerState.runtime_tensors).
OPTIMIZER_ARGUMENT_NAMES = (
    "optimizer_sr_arena",
    "optimizer_parameter",
    "optimizer_gradient",
    "optimizer_exp_avg",
    "optimizer_exp_avg_sq",
    "optimizer_bf16_parameter",
    "optimizer_final_parameter",
    "optimizer_final_exp_avg",
    "optimizer_final_exp_avg_sq",
    "optimizer_norm_partials",
    "optimizer_outputs",
    "optimizer_status",
    "optimizer_completed_steps",
    "optimizer_integer_control",
    "optimizer_hyperparameters",
)

# The shell's communication operands (communication.shell_fabric): the head weight and gradient
# rings, the valid-token all-reduce and the embedding route.
SHELL_FABRIC_ARGUMENT_NAMES = (
    "shell_weight_arena",
    "shell_gradient_arena",
    "shell_valid_arena",
    "shell_fabric_control",
    "shell_input_ids",
    "shell_local_valid_tokens",
    "shell_head_weight_raw",
    "shell_fabric_status",
    "shell_embedding_route_arena",
    "shell_embedding_route_peer_bases",
    "shell_embedding_route_unique_ids",
    "shell_embedding_route_inverse",
    "shell_embedding_route_owner_offsets",
    "shell_embedding_route_owner_counts",
    "shell_embedding_route_status",
)
# The launch ABI's `full_shell` group: the shell's communication operands, then one
# work-sharing scheduler state per dynamic GEMM body (ShellFabric.runtime_tensors).
SHELL_FABRIC_AND_SCHEDULER_ARGUMENT_NAMES = (
    *SHELL_FABRIC_ARGUMENT_NAMES,
    "shell_head_dx_scheduler_state",
    "shell_head_dw_scheduler_state",
    "shell_gate_up_dx_scheduler_state",
    "shell_gate_up_dw_scheduler_state",
    "shell_gate_up_fwd_scheduler_state",
    "shell_down_dx_scheduler_state",
    "shell_down_dw_scheduler_state",
    "shell_down_fwd_scheduler_state",
    "shell_qkv_dx_scheduler_state",
    "shell_qkv_dw_scheduler_state",
    "shell_qkv_fwd_scheduler_state",
    "shell_o_dx_scheduler_state",
    "shell_o_dw_scheduler_state",
    "shell_o_fwd_scheduler_state",
    "shell_head_fwd_scheduler_state",
)

# The decoder's working storage has two layer slots (training_tensors.py allocates them);
# layer L uses slot L % 2.
PHYSICAL_LAYER_SLOTS = 2

# In the forward, the top two layers' gate/up GEMM stores both the gate/up preactivation and
# the SwiGLU output in the layer slot (the other layers store only the SwiGLU output).  Nothing
# overwrites either before the backward's first two reverse visits, which use them in place and
# skip the gate/up recompute.
GATE_UP_RETAINED_TOP_LAYERS = 2
# The top twelve layers (all of them, below twelve) save residual_mid, the attention block's
# output plus its residual, from the attention-output GEMM into a checkpoint bank, top layer
# first.  Their reverse visits point residual_mid's row-table entry at the saved record and skip
# the attention-output recompute.
CHECKPOINT_PLAN = SHAPE.checkpoint_plan
RESIDUAL_MID_CHECKPOINT_LAYERS = CHECKPOINT_PLAN.residual
RESIDUAL_MID_CHECKPOINT_ELEMENTS = model.SEQUENCE * model.HIDDEN
RESIDUAL_MID_CHECKPOINT_BYTES = 2 * RESIDUAL_MID_CHECKPOINT_ELEMENTS
RESIDUAL_MID_CHECKPOINT_BANK_BYTES = CHECKPOINT_PLAN.residual_bank_bytes
# Two int64 words after the bank hold the two layer slots' residual_mid addresses, so a
# reverse visit can point the row-table entry at either a saved record or its slot; a record
# is never copied back into the slot.
RESIDUAL_MID_CHECKPOINT_STORAGE_BYTES = CHECKPOINT_PLAN.residual_storage_bytes
# The top layers save their forward attention output (O) and LSE, and their reverse visits
# restore them instead of recomputing FA4's forward (Q, K and V are still recomputed).  The
# plan (training_megakernel.shape.CheckpointPlan) puts the top layers, at most nine, in the
# optimizer gradient's low layers, which the backward writes last, and the rest in two banks
# (at 36 layers: layers 27-35 borrow, layers 1-26 use the banks).  Layer 0 is never saved.
# With nothing borrowed the checkpoints are off and every boundary below equals the depth.
ATTENTION_CHECKPOINT_BORROWED_LAYERS = CHECKPOINT_PLAN.prefix
ATTENTION_CHECKPOINT_BORROWED_FIRST_LAYER = CHECKPOINT_PLAN.prefix_first
ATTENTION_CHECKPOINT_BANK_LAYERS = CHECKPOINT_PLAN.extra
ATTENTION_CHECKPOINT_BANK0_LAYERS = CHECKPOINT_PLAN.bank0
ATTENTION_CHECKPOINT_BANK1_LAYERS = CHECKPOINT_PLAN.bank1
ATTENTION_CHECKPOINT_BANK0_FIRST_LAYER = CHECKPOINT_PLAN.split
ATTENTION_CHECKPOINT_FIRST_LAYER = CHECKPOINT_PLAN.first
ATTENTION_CHECKPOINT_OUTPUT_BYTES = (
    model.SEQUENCE * model.QUERY_HEADS * model.HEAD_DIM * 2
)
ATTENTION_CHECKPOINT_LAYER_BYTES = CHECKPOINT_PLAN.layer_bytes
ATTENTION_CHECKPOINT_BORROWED_BYTES = CHECKPOINT_PLAN.prefix_bytes
ATTENTION_CHECKPOINT_BANK0_BYTES = CHECKPOINT_PLAN.bank0_bytes
ATTENTION_CHECKPOINT_BANK1_BYTES = CHECKPOINT_PLAN.bank1_bytes
# The borrowed records must fit in the optimizer gradient of the layers the backward writes last.
assert (
    ATTENTION_CHECKPOINT_BORROWED_BYTES
    < CHECKPOINT_PLAN.owner_layers * OWNER_GRADIENT_LAYER_BYTES
    or not ATTENTION_CHECKPOINT_BORROWED_LAYERS
)
# Tensor views of the checkpoint stores keep at least one element when a store
# is empty (the checkpoint code is then never reached).
ATTENTION_CHECKPOINT_BORROWED_VIEW_BYTES = max(ATTENTION_CHECKPOINT_BORROWED_BYTES, 4)
ATTENTION_CHECKPOINT_BANK0_VIEW_BYTES = max(ATTENTION_CHECKPOINT_BANK0_BYTES, 4)
ATTENTION_CHECKPOINT_BANK1_VIEW_BYTES = max(ATTENTION_CHECKPOINT_BANK1_BYTES, 4)
# The CTAs that prefetch the next layer's weights before joining a GEMM, and how many copy each
# of the five panels (norm, qkv, o, gate_up, down), roughly in proportion to the panel sizes.
WEIGHT_PREFETCH_CTAS = 48
WEIGHT_PREFETCH_CTAS_PER_PANEL = (1, 6, 4, 25, 12)
assert sum(WEIGHT_PREFETCH_CTAS_PER_PANEL) == WEIGHT_PREFETCH_CTAS

# The decoder fabric's control words and failure kinds (communication.decoder_fabric).
DECODER_CONTROL_WEIGHT_MULTICAST_BASE = 0
DECODER_CONTROL_GRADIENT_MULTICAST_BASE = 1
DECODER_CONTROL_RANK = 2
DECODER_CONTROL_TIMEOUT_NS = 3

DECODER_STATUS_WEIGHT_REUSE_TIMEOUT = 1
DECODER_STATUS_WEIGHT_READY_TIMEOUT = 2
DECODER_STATUS_GRADIENT_READY_TIMEOUT = 3
DECODER_STATUS_GRADIENT_DONE_TIMEOUT = 4

# The embedding gather's request and response epochs, one word per rank each, sit in the route
# arena's owner-gradient region, which the embedding-gradient reduce-scatter does not use.
EMBEDDING_ROUTE_FORWARD_REQUEST = embedding_route.ROUTE_FORWARD_REQUEST_OFFSET
EMBEDDING_ROUTE_FORWARD_RESPONSE = EMBEDDING_ROUTE_FORWARD_REQUEST + model.WORLD * 4

# The page (gemm_members), from which Quack sizes every GEMM member's pipeline stages; the
# kernel allocates it 1024-byte aligned.
assert PROGRAM_SMEM_PAGE_BYTES % 1024 == 0


@cute.jit
def _record_timeout_status(
    status: cute.Tensor,
    kind: Int32,
    family: Int32,
    panel: Int32,
    expected: Int32,
    observed: Int32,
):
    """Record a failed wait in ``status`` (the SHELL_STATUS_* words of shell_fabric).

    The first failure claims the record by compare-and-swap on the kind word; only it writes
    the family, panel, expected and observed values.
    """

    won = atomic_cas_gpu_u32(
        status.iterator.toint() + Int64(shell_fabric.SHELL_STATUS_KIND * 4),
        Int32(shell_fabric.SHELL_STATUS_OK),
        kind,
    )
    if won == Int32(shell_fabric.SHELL_STATUS_OK):
        _ = store_relaxed_gpu_u32(
            status.iterator.toint() + Int64(shell_fabric.SHELL_STATUS_MATRIX * 4), family
        )
        _ = store_relaxed_gpu_u32(
            status.iterator.toint() + Int64(shell_fabric.SHELL_STATUS_PANEL * 4), panel
        )
        _ = store_relaxed_gpu_u32(
            status.iterator.toint() + Int64(shell_fabric.SHELL_STATUS_EXPECTED * 4), expected
        )
        _ = store_relaxed_gpu_u32(
            status.iterator.toint() + Int64(shell_fabric.SHELL_STATUS_OBSERVED * 4), observed
        )


@cute.jit
def _wait_for_epoch_on_all_ranks(
    address: Int64,
    expected: Int32,
    status: cute.Tensor,
    kind: Int32,
    family: Int32,
    panel: Int32,
    timeout_ns: Int64,
):
    """Spin until the epoch word at multicast ``address`` reaches ``expected`` on every rank.

    Each poll is a multimem minimum over the ranks.  After ``timeout_ns`` the wait records
    the failure in ``status`` and returns without waiting further.
    """

    observed = Int32(0)
    deadline = global_timer_ns() + timeout_ns
    waiting = Int32(1)
    while waiting == Int32(1):
        observed = multicast_min_acquire_u32(address)
        if observed >= expected:
            waiting = Int32(0)
        elif global_timer_ns() >= deadline:
            _record_timeout_status(
                status, kind, family, panel, expected, observed
            )
            waiting = Int32(0)


@cute.jit
def _weight_publication_epoch(step: cute.Tensor, service: Int32) -> Int32:
    """The epoch of weight publication ``service`` in decoder-fabric step ``step[0]``.

    Epochs count from 1 across steps, two publications per layer and step: service ``layer``
    in the forward and ``DEPTH + reverse_visit`` in the backward.
    """

    return (
        (Int32(step[0]) - Int32(1)) * Int32(WEIGHT_PUBLICATIONS_PER_STEP)
        + service
        + Int32(1)
    )


@cute.jit
def _weight_panel_owner_extent(panel_index: Int32):
    """One rank's share of a weight panel in 64-bit words, and the panel's byte offset.

    The share counts words of four BF16 values; the offset is inside a weight-ring slot.  The
    literals are model.WEIGHT_PANELS and arenas.DECODER_WEIGHT_PANEL_OFFSETS, which depend only
    on the model's width.
    """

    owner_words = Int32(264)
    relative_offset = Int64(0)
    if panel_index == Int32(1):
        owner_words = Int32(786432)
        relative_offset = Int64(24576)
    elif panel_index == Int32(2):
        owner_words = Int32(524288)
        relative_offset = Int64(50360320)
    elif panel_index == Int32(3):
        owner_words = Int32(3145728)
        relative_offset = Int64(83918848)
    elif panel_index == Int32(4):
        owner_words = Int32(1572864)
        relative_offset = Int64(285249536)
    return owner_words, relative_offset


@cute.jit
def _gradient_site_owner_extent(site_index: Int32):
    """One rank's share of a gradient site in pairs of FP32 values, and the site's byte offset.

    The offset is inside a gradient-ring slot.  The literals are model.REDUCTION_SITES and
    arenas.DECODER_GRADIENT_SITE_OFFSETS, which depend only on the model's width.
    """

    owner_pairs = Int32(3145728)
    relative_offset = Int64(0)
    if site_index == Int32(1):
        owner_pairs = Int32(6291456)
        relative_offset = Int64(201330688)
    elif site_index == Int32(2):
        owner_pairs = Int32(1048576)
        relative_offset = Int64(603987968)
    elif site_index == Int32(3):
        owner_pairs = Int32(1572864)
        relative_offset = Int64(671100928)
    elif site_index == Int32(4):
        owner_pairs = Int32(528)
        relative_offset = Int64(771768320)
    return owner_pairs, relative_offset


@cute.jit
def _gradient_publication_epoch(step: cute.Tensor, reverse_visit: Int32) -> Int32:
    """The epoch of reverse visit ``reverse_visit``'s gradient publication in step ``step[0]``.

    Epochs count from 1 across steps, one publication per layer and step.
    """

    return (
        (Int32(step[0]) - Int32(1)) * Int32(model.DEPTH)
        + reverse_visit
        + Int32(1)
    )


@cute.jit
def fp32_to_bf16(source_flat: cute.Tensor, destination_flat: cute.Tensor):
    """Convert SEQUENCE x HIDDEN FP32 values to BF16 with every thread of the grid.

    The head region uses it to turn the head's FP32 dX into the final RMSNorm backward's input.
    """

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    index = bidx * model.PROGRAM_THREADS + tidx
    stride = grid_x * model.PROGRAM_THREADS
    for element in cutlass.range(index, model.SEQUENCE * model.HIDDEN, stride, unroll=1):
        destination_flat[element] = source_flat[element].to(BFloat16)


@dataclass(frozen=True)
class ShellTensors:
    """The shell's device tensors (final RMSNorm, LM head and loss) and the checkpoint stores.

    The kernel takes them as its ``shell_*`` operands, in the order ``_shell_operand_pairs``
    gives.
    """

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

    @classmethod
    def allocate(cls, device: torch.device) -> "ShellTensors":

        # At run time head_dweight is symmetric memory.  The head dW GEMM writes it as an
        # ordinary contiguous tensor, and reduce_scatter_head_gradient sums every rank's copy
        # into this rank's owner rows through its multicast address, without a copy through a
        # ring.  The build's one GPU has no peers, so the tensor's own address stands in for the
        # multicast address.
        head_dweight = torch.empty(
            model.VOCAB * model.HIDDEN, device=device, dtype=torch.float32
        ).view(model.VOCAB, model.HIDDEN)
        head_dweight_multicast_base = head_dweight.data_ptr()
        return cls(
            final_norm_weight=torch.empty(model.HIDDEN, device=device, dtype=torch.bfloat16),
            final_normalized_hidden=torch.empty(
                model.SEQUENCE, model.HIDDEN, device=device, dtype=torch.bfloat16
            ),
            final_rstd=torch.empty(model.SEQUENCE, device=device, dtype=torch.float32),
            head_weight=torch.empty(model.VOCAB, model.HIDDEN, device=device, dtype=torch.bfloat16),
            labels=torch.empty(model.SEQUENCE, device=device, dtype=torch.int32),
            global_valid_tokens=torch.empty(1, device=device, dtype=torch.int32),
            dlogits_slab=torch.empty(
                model.SEQUENCE, model.VOCAB, device=device, dtype=torch.bfloat16
            ),
            per_token_loss=torch.empty(model.SEQUENCE, device=device, dtype=torch.float32),
            loss=torch.empty(1, device=device, dtype=torch.float32),
            head_dhidden_fp32=torch.empty(
                model.SEQUENCE, model.HIDDEN, device=device, dtype=torch.float32
            ),
            head_dweight=head_dweight,
            head_dweight_multicast_base=head_dweight_multicast_base,
            task_records=torch.arange(model.HEAD_CHUNKS, device=device, dtype=torch.int32),
            active_chunks=torch.tensor([model.HEAD_CHUNKS], device=device, dtype=torch.int32),
            generation_slot=torch.zeros(1, device=device, dtype=torch.int32),
            final_dnorm_bf16=torch.empty(
                model.SEQUENCE, model.HIDDEN, device=device, dtype=torch.bfloat16
            ),
            final_norm_partial=torch.empty(
                2 * model.PROGRAM_CTAS, model.HIDDEN, device=device, dtype=torch.float32
            ),
            final_norm_grad=torch.empty(model.HIDDEN, device=device, dtype=torch.float32),
            # Placeholders: allocate_checkpoint_banks allocates the checkpoint stores after
            # the rest of the training state.
            fa4_checkpoint_extra0=torch.empty(1, device=device, dtype=torch.bfloat16),
            fa4_checkpoint_extra1=torch.empty(1, device=device, dtype=torch.bfloat16),
            residual_mid_checkpoint=torch.empty(1, device=device, dtype=torch.bfloat16),
        )


def allocate_checkpoint_banks(
    shell: ShellTensors, device: torch.device
) -> None:
    """Replace ``shell``'s placeholders with the two attention banks and the residual storage.

    kernel/build.py calls it after allocating the rest of the training state.  An empty
    store keeps one element.
    """

    for name, bank_bytes in (
        ("fa4_checkpoint_extra0", ATTENTION_CHECKPOINT_BANK0_BYTES),
        ("fa4_checkpoint_extra1", ATTENTION_CHECKPOINT_BANK1_BYTES),
        ("residual_mid_checkpoint", RESIDUAL_MID_CHECKPOINT_STORAGE_BYTES),
    ):
        object.__setattr__(
            shell,
            name,
            torch.empty(
                max(1, bank_bytes // 2),
                device=device,
                dtype=torch.bfloat16,
            ),
        )


def store_residual_mid_slot_addresses(tensors: TrainingTensors) -> None:
    """Write both layer slots' residual_mid addresses into the two words after the residual bank.

    ``set_residual_mid_row`` reads them to point a layer's row-table entry back at its slot.
    """

    physical = tensors.slabs["residual_mid"]
    if physical.shape[0] != 2:
        raise AssertionError("direct residual checkpoint requires two physical slots")
    checkpoint = tensors.shell.residual_mid_checkpoint
    if checkpoint.numel() * checkpoint.element_size() != (
        RESIDUAL_MID_CHECKPOINT_STORAGE_BYTES
    ):
        raise AssertionError("residual checkpoint storage extent drifted")
    route_words = checkpoint.view(torch.int64)[
        RESIDUAL_MID_CHECKPOINT_BANK_BYTES // 8 :
    ]
    if route_words.numel() != 2:
        raise AssertionError("residual checkpoint route table extent drifted")
    route_words.copy_(
        torch.tensor(
            [physical[0].data_ptr(), physical[1].data_ptr()],
            device=checkpoint.device,
            dtype=torch.int64,
        )
    )


class TrainingProgram:
    """The program's members and phase methods.

    ``training_kernel`` subclasses it with the generated ``training_step_kernel`` and its three
    regions; ``__call__`` then prepares the members' arguments and launches the kernel.  The
    constructor takes the number of layer slots (``capacity``, two), the attention workspace's
    row bound and the decoder GEMM families.
    """

    reanchor_smem_page = anchors.reanchor_smem_page
    run_attention_backward = attention.run_attention_backward
    grid_phase_barrier = grid_barrier.grid_phase_barrier
    run_row_phase = decoder_layer.run_row_phase
    # The attention forward stores FA4's LSE as one [head][row] slab per layer slot
    # (attention.generation_major_lse), the layout the backward preprocess reads.
    run_attention_forward = attention.run_attention_forward

    run_initial_gate_up_gemm = run_initial_gate_up_gemm
    run_decoder_gemm = run_decoder_gemm
    run_qkv_dx_gemm_without_cta0 = (
        run_qkv_dx_gemm_without_cta0
    )
    run_attention_output_checkpoint_gemm = (
        run_attention_output_checkpoint_gemm
    )
    run_optimizer_step = clipped_adamw_step

    def __init__(
        self,
        capacity: int,
        *,
        workspace_rows: int,
        families: tuple[decoder_layer.GemmFamily, ...],
    ):
        if not workspace_rows:
            raise ValueError("the packed body needs its compile-time workspace row bound")
        self.capacity = capacity
        self.workspace_rows = workspace_rows
        self.total_tokens = capacity * model.SEQUENCE
        self.smem_page_origin = None
        self.families = tuple(families)
        index = {family.name: i for i, family in enumerate(self.families)}

        # The FlashAttention 4 members, and the functions the attention phases reach through the
        # program: FA4's `__call__` setup on the program's attention schedulers
        # (fa4_forward_call, fa4_backward_call) and the backward kernel's setup and role.
        self.backward = AttentionBackwardMember()
        self.forward = AttentionForwardMember()
        self._forward_setup = forward_call_setup
        # AttentionBackwardMember.__call__ calls the backward setup through this attribute.
        self.backward._call_setup = backward_call_setup
        self.backward_setup, self.backward_role = backward_setup, backward_role

        # The layer RMS norms, each run as two 128-thread shards.
        self.rms1_fwd = RMSNorm(
            BFloat16, model.HIDDEN, is_layernorm=False, config=RMS_FORWARD_CONFIG
        )
        self.rms2_fwd = RMSNorm(
            BFloat16, model.HIDDEN, is_layernorm=False, config=RMS_FORWARD_CONFIG
        )
        self.rms1_bwd = RMSNormBackward(
            BFloat16,
            model.HIDDEN,
            dout_dtype=BFloat16,
            T_hint=model.SEQUENCE,
            per_head=False,
            config=RMS_BACKWARD_CONFIG,
        )
        self.rms2_bwd = RMSNormBackward(
            BFloat16,
            model.HIDDEN,
            dout_dtype=BFloat16,
            T_hint=model.SEQUENCE,
            per_head=False,
            config=RMS_BACKWARD_CONFIG,
        )

        # One Quack GEMM member per decoder family.
        members: list = [None] * len(self.families)
        members[index["gate_up_dx"]] = make_projection_member()
        members[index["gate_up_dw"]] = make_projection_member()
        members[index["down_fwd"]] = make_down_forward_member()
        # Down dX runs the SwiGLU backward in its epilogue and stores dgate and dup, the two
        # halves of each gate_up_dy row, instead of its own output.
        members[index["down_dx"]] = make_down_dx_member()
        members[index["down_dw"]] = make_projection_member()
        members[index["qkv_fwd"]] = make_qkv_forward_member()
        members[index["qkv_dx"]] = make_projection_member()
        members[index["qkv_dw"]] = make_projection_member()
        members[index["o_fwd"]] = make_attention_output_forward_member()
        members[index["o_dx"]] = make_projection_member()
        members[index["o_dw"]] = make_projection_member()
        members[index["gate_up_fwd"]] = make_gate_up_forward_member()
        self._members = tuple(members)
        self.qkv_fwd_family_index = index["qkv_fwd"]
        self.gate_up_fwd_family_index = index["gate_up_fwd"]
        self.down_fwd_family_index = index["down_fwd"]
        # The forward's gate/up GEMM below the top GATE_UP_RETAINED_TOP_LAYERS layers, which
        # stores only the SwiGLU output.
        self.initial_gate_up_aux_member = make_gate_up_forward_member()
        self.down_dx_family_index = index["down_dx"]
        self.gate_up_dx_family_index = index["gate_up_dx"]
        self.o_fwd_family_index = index["o_fwd"]
        # The forward's attention-output GEMM for the residual-checkpoint layers, which also
        # stores residual_mid into the checkpoint bank.
        self.o_fwd_checkpoint_member = make_attention_output_checkpoint_member()
        # Down forward adds o_fwd's output, residual_mid, as its residual.
        self.down_fwd_residual_source_family_index = index["o_fwd"]
        # The LM head's members, and the final RMSNorm.
        self.head_phase = lm_head.HeadPhase()
        self.final_rms_fwd = RMSNorm(
            BFloat16, model.HIDDEN, is_layernorm=False, config=RMS_FORWARD_CONFIG
        )
        self.final_rms_bwd = RMSNormBackward(
            BFloat16,
            model.HIDDEN,
            dout_dtype=BFloat16,
            T_hint=model.SEQUENCE,
            per_head=False,
            config=RMS_BACKWARD_CONFIG,
        )

    def member_smem_bytes(self) -> dict:
        """Shared storage of each prepared decoder GEMM member, in bytes, by family name.

        It includes the activation-only gate/up member; compile_training_program checks each
        against the page.
        """

        census = {
            f.name: int(m.shared_storage.size_in_bytes())
            for f, m in zip(self.families, self._members)
        }
        member = self.initial_gate_up_aux_member
        if hasattr(member, "shared_storage"):
            census["gate_up_initial_aux_only"] = int(
                member.shared_storage.size_in_bytes()
            )
        return census

    @cute.jit
    def copy_attention_checkpoint(
        self,
        generation: Int32,
        route: Int32,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        row_table: cute.Tensor,
        destination_table: cute.Tensor,
        explicit_checkpoint0: cute.Tensor,
        explicit_checkpoint1: cute.Tensor,
        restore: cutlass.Constexpr[bool],
    ):
        """Save (``restore`` False) or restore layer ``generation``'s attention output and LSE.

        Layers from ATTENTION_CHECKPOINT_BORROWED_FIRST_LAYER up keep their records in the
        optimizer gradient's low layers (row 0 of ``destination_table``), top layer first;
        lower layers use bank 0 or bank 1.  The output is read through the row table (``mO``
        is unused).  The LSE copy indexes the LSE as [head][slot][row], while the forward
        stores it as [slot][head][row] (attention.generation_major_lse), so one copy moves the
        heads of one parity from both slots.  Restoring layer g + 1 and then layer g still
        rewrites all of layer g's heads with its forward values, because the saved layers are a
        contiguous range that ends at the top layer.
        """

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        global_thread = bidx * Int32(model.PROGRAM_THREADS) + tidx
        global_threads = Int32(model.PROGRAM_CTAS * model.PROGRAM_THREADS)
        o_route = anchors.row_table_view(
            row_table,
            row_table.iterator.memspace,
            generation * Int32(decoder_layer.ROW_TABLE_WIDTH)
            + Int32(decoder_layer.ROW_TABLE_INDEX["attention_out"]),
            BFloat16,
            cute.make_layout(model.SEQUENCE * model.HIDDEN),
        )
        checkpoint_bf16 = anchors.row_table_view(
            destination_table,
            destination_table.iterator.memspace,
            Int32(0),
            BFloat16,
            cute.make_layout(ATTENTION_CHECKPOINT_BORROWED_VIEW_BYTES // 2),
        )
        checkpoint_f32 = anchors.row_table_view(
            destination_table,
            destination_table.iterator.memspace,
            Int32(0),
            Float32,
            cute.make_layout(ATTENTION_CHECKPOINT_BORROWED_VIEW_BYTES // 4),
        )
        extra_checkpoint0_bf16 = cute.make_tensor(
            explicit_checkpoint0.iterator,
            cute.make_layout(ATTENTION_CHECKPOINT_BANK0_VIEW_BYTES // 2),
        )
        extra_checkpoint0_f32 = cute.make_tensor(
            cute.recast_ptr(explicit_checkpoint0.iterator, None, dtype=Float32),
            cute.make_layout(ATTENTION_CHECKPOINT_BANK0_VIEW_BYTES // 4),
        )
        extra_checkpoint1_bf16 = cute.make_tensor(
            explicit_checkpoint1.iterator,
            cute.make_layout(ATTENTION_CHECKPOINT_BANK1_VIEW_BYTES // 2),
        )
        extra_checkpoint1_f32 = cute.make_tensor(
            cute.recast_ptr(explicit_checkpoint1.iterator, None, dtype=Float32),
            cute.make_layout(ATTENTION_CHECKPOINT_BANK1_VIEW_BYTES // 4),
        )
        checkpoint_index = Int32(model.DEPTH - 1) - generation
        checkpoint_base = (
            checkpoint_bf16.iterator.toint()
            + Int64(checkpoint_index) * Int64(ATTENTION_CHECKPOINT_LAYER_BYTES)
        )
        checkpoint_lse_element = (
            checkpoint_index * Int32(ATTENTION_CHECKPOINT_LAYER_BYTES // 4)
            + Int32(ATTENTION_CHECKPOINT_OUTPUT_BYTES // 4)
        )
        if generation < Int32(ATTENTION_CHECKPOINT_BORROWED_FIRST_LAYER):
            checkpoint_index = (
                Int32(ATTENTION_CHECKPOINT_BORROWED_FIRST_LAYER - 1) - generation
            )
            if generation >= Int32(ATTENTION_CHECKPOINT_BANK0_FIRST_LAYER):
                checkpoint_base = (
                    extra_checkpoint0_bf16.iterator.toint()
                    + Int64(checkpoint_index)
                    * Int64(ATTENTION_CHECKPOINT_LAYER_BYTES)
                )
            else:
                checkpoint_index = (
                    Int32(
                        ATTENTION_CHECKPOINT_BANK0_FIRST_LAYER - 1
                    )
                    - generation
                )
                checkpoint_base = (
                    extra_checkpoint1_bf16.iterator.toint()
                    + Int64(checkpoint_index)
                    * Int64(ATTENTION_CHECKPOINT_LAYER_BYTES)
                )
            checkpoint_lse_element = (
                checkpoint_index * Int32(ATTENTION_CHECKPOINT_LAYER_BYTES // 4)
                + Int32(ATTENTION_CHECKPOINT_OUTPUT_BYTES // 4)
            )
        lse_flat = cute.make_tensor(
            mLSE.iterator,
            cute.make_layout(
                2 * model.SEQUENCE * model.QUERY_HEADS
            ),
        )

        o_words = Int32(ATTENTION_CHECKPOINT_OUTPUT_BYTES // 16)
        for word in cutlass.range(
            global_thread, o_words, global_threads, unroll=1
        ):
            byte_offset = Int64(word) * Int64(16)
            source = o_route.iterator.toint()
            destination = checkpoint_base
            if const_expr(restore):
                source = checkpoint_base
                destination = o_route.iterator.toint()
            _ = copy_b128(
                source + byte_offset, destination + byte_offset
            )

        lse_elements = Int32(model.SEQUENCE * model.QUERY_HEADS)
        for element in cutlass.range(
            global_thread, lse_elements, global_threads, unroll=1
        ):
            head = element // Int32(model.SEQUENCE)
            row = element - head * Int32(model.SEQUENCE)
            lse_source_element = (
                head * Int32(2 * model.SEQUENCE)
                + route * Int32(model.SEQUENCE)
                + row
            )
            if const_expr(restore):
                if generation < Int32(ATTENTION_CHECKPOINT_BORROWED_FIRST_LAYER):
                    if generation >= Int32(ATTENTION_CHECKPOINT_BANK0_FIRST_LAYER):
                        lse_flat[lse_source_element] = extra_checkpoint0_f32[
                            checkpoint_lse_element + element
                        ]
                    else:
                        lse_flat[lse_source_element] = extra_checkpoint1_f32[
                            checkpoint_lse_element + element
                        ]
                else:
                    lse_flat[lse_source_element] = checkpoint_f32[
                        checkpoint_lse_element + element
                    ]
            else:
                if generation < Int32(ATTENTION_CHECKPOINT_BORROWED_FIRST_LAYER):
                    if generation >= Int32(ATTENTION_CHECKPOINT_BANK0_FIRST_LAYER):
                        extra_checkpoint0_f32[
                            checkpoint_lse_element + element
                        ] = lse_flat[lse_source_element]
                    else:
                        extra_checkpoint1_f32[
                            checkpoint_lse_element + element
                        ] = lse_flat[lse_source_element]
                else:
                    checkpoint_f32[checkpoint_lse_element + element] = lse_flat[
                        lse_source_element
                    ]
        cute.arch.sync_threads()

    @cute.jit
    def set_residual_mid_row(
        self,
        generation: Int32,
        route: Int32,
        checkpoint_index: Int32,
        row_table: cute.Tensor,
        checkpoint_bank: cute.Tensor,
        checkpoint: cutlass.Constexpr[bool],
    ):
        """Point layer ``generation``'s residual_mid row-table entry at its record or its slot.

        RMS2's forward and backward read residual_mid through the row table, which has one row
        per layer.  With ``checkpoint`` the entry points at record ``checkpoint_index`` of the
        residual bank; otherwise at the layer slot ``route``'s own residual_mid, whose address
        ``store_residual_mid_slot_addresses`` keeps after the bank.  CTA 0's thread 0 writes
        it; the grid barrier that follows in the kernel makes it visible before RMS2 reads it.
        """

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if bidx == 0:
            if tidx == 0:
                if const_expr(checkpoint):
                    address = (
                        checkpoint_bank.iterator.toint()
                        + Int64(checkpoint_index)
                        * Int64(RESIDUAL_MID_CHECKPOINT_BYTES)
                    )
                else:
                    address = load_u64(
                        checkpoint_bank.iterator.toint()
                        + Int64(RESIDUAL_MID_CHECKPOINT_BANK_BYTES)
                        + Int64(route) * Int64(8)
                    )
                table_index = (
                    generation * Int32(decoder_layer.ROW_TABLE_WIDTH)
                    + Int32(decoder_layer.ROW_TABLE_INDEX["residual_mid"])
                )
                _ = store_u64(
                    row_table.iterator.toint()
                    + Int64(table_index) * Int64(8),
                    address,
                )

    @cute.jit
    def advance_decoder_fabric_step(
        self,
        phase_counter: cute.Tensor,
        fabric_step: cute.Tensor,
    ):
        """Increment the decoder fabric's step counter, from which this step's epochs derive."""

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if bidx == Int32(0) and tidx == Int32(0):
            fabric_step[0] = Int32(fabric_step[0]) + Int32(1)
        self.grid_phase_barrier(phase_counter)

    @cute.jit
    def publish_layer_weights(
        self,
        generation: Int32,
        service: Int32,
        phase_counter: cute.Tensor,
        weight_arena: cute.Tensor,
        source_table: cute.Tensor,
        control: cute.Tensor,
        step: cute.Tensor,
        status: cute.Tensor,
    ):
        """All-gather layer ``generation``'s five weight panels with the whole grid.

        Each rank copies its share from the optimizer's BF16 parameters (``source_table``) to
        slot ``generation % 2`` of every rank's weight ring.  Before that, CTA 0 waits until
        every rank has released the slot's previous occupant: CONSUMED must reach the READY
        epoch the slot holds.  Afterwards it stores this rank's READY epochs and does not wait
        for the other ranks'; ``wait_layer_weights_ready`` does.
        """

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        epoch = _weight_publication_epoch(step, service)
        slot = generation - (
            generation // Int32(arenas.LAYER_SLOTS)
        ) * Int32(arenas.LAYER_SLOTS)
        if bidx == Int32(0) and tidx == Int32(0):
            # At the forward-to-backward turn two consecutive publications use the top layer's
            # slot, so the previous occupant's epoch is not always epoch - LAYER_SLOTS; read it.
            prior_ready_index = slot * Int32(DECODER_WEIGHT_PANEL_COUNT)
            prior_ready = load_acquire_sys_u32(
                weight_arena.iterator.toint()
                + Int64(arenas.DECODER_WEIGHT_READY_OFFSET)
                + Int64(prior_ready_index) * Int64(4)
            )
            if prior_ready > Int32(0):
                _wait_for_epoch_on_all_ranks(
                    Int64(control[DECODER_CONTROL_WEIGHT_MULTICAST_BASE])
                    + Int64(arenas.DECODER_WEIGHT_CONSUMED_OFFSET)
                    + Int64(slot) * Int64(4),
                    prior_ready,
                    status,
                    Int32(DECODER_STATUS_WEIGHT_REUSE_TIMEOUT),
                    service,
                    Int32(-1),
                    Int64(control[DECODER_CONTROL_TIMEOUT_NS]),
                )
        self.grid_phase_barrier(phase_counter)

        memspace = source_table.iterator.memspace
        global_thread = bidx * Int32(model.PROGRAM_THREADS) + tidx
        global_threads = Int32(model.PROGRAM_CTAS * model.PROGRAM_THREADS)
        rank = Int32(control[DECODER_CONTROL_RANK])
        for panel_index in cutlass.range(0, DECODER_WEIGHT_PANEL_COUNT, 1, unroll=1):
            owner_words, relative_offset = _weight_panel_owner_extent(panel_index)
            source = anchors.row_table_view(
                source_table,
                memspace,
                generation * Int32(DECODER_WEIGHT_PANEL_COUNT) + panel_index,
                BFloat16,
                cute.make_layout(1),
            )
            destination = (
                Int64(control[DECODER_CONTROL_WEIGHT_MULTICAST_BASE])
                + Int64(arenas.DECODER_WEIGHT_PAYLOAD_OFFSET)
                + Int64(slot) * Int64(arenas.DECODER_WEIGHT_SLOT_STRIDE)
                + relative_offset
                + Int64(rank) * Int64(owner_words) * Int64(4 * model.BF16_BYTES)
            )
            for vector_word in cutlass.range(
                global_thread,
                owner_words // Int32(2),
                global_threads,
                unroll=1,
            ):
                _ = copy_to_multicast_b128(
                    source.iterator.toint() + Int64(vector_word) * Int64(16),
                    destination + Int64(vector_word) * Int64(16),
                )
        self.grid_phase_barrier(phase_counter)

        if bidx == Int32(0) and tidx == Int32(0):
            _ = fence_sys()
            for panel_index in cutlass.range(
                0, DECODER_WEIGHT_PANEL_COUNT, 1, unroll=1
            ):
                ready_index = (
                    slot * Int32(DECODER_WEIGHT_PANEL_COUNT) + Int32(panel_index)
                )
                _ = store_release_sys_u32(
                    weight_arena.iterator.toint()
                    + Int64(arenas.DECODER_WEIGHT_READY_OFFSET)
                    + Int64(ready_index) * Int64(4),
                    epoch,
                )

    @cute.jit
    def prefetch_layer_weights(
        self,
        generation: Int32,
        service: Int32,
        physical_cta_base: cutlass.Constexpr[int],
        weight_arena: cute.Tensor,
        source_table: cute.Tensor,
        control: cute.Tensor,
        step: cute.Tensor,
        status: cute.Tensor,
    ):
        """Copy this CTA's share of layer ``generation``'s weight panels, then return to compute.

        WEIGHT_PREFETCH_CTAS CTAs from ``physical_cta_base`` split the panels as
        WEIGHT_PREFETCH_CTAS_PER_PANEL gives and copy into slot ``generation % 2`` of every
        rank's weight ring; the kernel calls it with CTAs 0-47 before the forward's down_fwd
        GEMM and CTAs 1-48 during the backward's qkv dX, while CTA 0 reduces the q/k-norm
        gradients.  Each CTA first waits, as ``publish_layer_weights`` does, until every rank
        has released the slot's previous occupant.  ``mark_prefetched_weights_ready``
        publishes READY after the GEMM's grid barrier.
        """

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        service_bidx = bidx - Int32(physical_cta_base)
        slot = generation - (
            generation // Int32(arenas.LAYER_SLOTS)
        ) * Int32(arenas.LAYER_SLOTS)

        if tidx == Int32(0):
            prior_ready_index = slot * Int32(DECODER_WEIGHT_PANEL_COUNT)
            prior_ready = load_acquire_sys_u32(
                weight_arena.iterator.toint()
                + Int64(arenas.DECODER_WEIGHT_READY_OFFSET)
                + Int64(prior_ready_index) * Int64(4)
            )
            if prior_ready > Int32(0):
                _wait_for_epoch_on_all_ranks(
                    Int64(control[DECODER_CONTROL_WEIGHT_MULTICAST_BASE])
                    + Int64(arenas.DECODER_WEIGHT_CONSUMED_OFFSET)
                    + Int64(slot) * Int64(4),
                    prior_ready,
                    status,
                    Int32(DECODER_STATUS_WEIGHT_REUSE_TIMEOUT),
                    service,
                    Int32(-1),
                    Int64(control[DECODER_CONTROL_TIMEOUT_NS]),
                )
        cute.arch.sync_threads()

        panel_index = Int32(0)
        panel_begin = Int32(0)
        panel_ctas = Int32(WEIGHT_PREFETCH_CTAS_PER_PANEL[0])
        if service_bidx >= Int32(1):
            panel_index = Int32(1)
            panel_begin = Int32(1)
            panel_ctas = Int32(WEIGHT_PREFETCH_CTAS_PER_PANEL[1])
        if service_bidx >= Int32(7):
            panel_index = Int32(2)
            panel_begin = Int32(7)
            panel_ctas = Int32(WEIGHT_PREFETCH_CTAS_PER_PANEL[2])
        if service_bidx >= Int32(11):
            panel_index = Int32(3)
            panel_begin = Int32(11)
            panel_ctas = Int32(WEIGHT_PREFETCH_CTAS_PER_PANEL[3])
        if service_bidx >= Int32(36):
            panel_index = Int32(4)
            panel_begin = Int32(36)
            panel_ctas = Int32(WEIGHT_PREFETCH_CTAS_PER_PANEL[4])

        owner_words, relative_offset = _weight_panel_owner_extent(panel_index)
        source = anchors.row_table_view(
            source_table,
            source_table.iterator.memspace,
            generation * Int32(DECODER_WEIGHT_PANEL_COUNT) + panel_index,
            BFloat16,
            cute.make_layout(1),
        )
        rank = Int32(control[DECODER_CONTROL_RANK])
        destination = (
            Int64(control[DECODER_CONTROL_WEIGHT_MULTICAST_BASE])
            + Int64(arenas.DECODER_WEIGHT_PAYLOAD_OFFSET)
            + Int64(slot) * Int64(arenas.DECODER_WEIGHT_SLOT_STRIDE)
            + relative_offset
            + Int64(rank) * Int64(owner_words) * Int64(4 * model.BF16_BYTES)
        )
        local_thread = (
            (service_bidx - panel_begin) * Int32(model.PROGRAM_THREADS) + tidx
        )
        local_threads = panel_ctas * Int32(model.PROGRAM_THREADS)
        for vector_word in cutlass.range(
            local_thread,
            owner_words // Int32(2),
            local_threads,
            unroll=1,
        ):
            _ = copy_to_multicast_b128(
                source.iterator.toint() + Int64(vector_word) * Int64(16),
                destination + Int64(vector_word) * Int64(16),
            )
        # These CTAs go straight into a warp-specialized GEMM: fence each thread's multicast
        # stores and synchronize the CTA before its warps split into producer and consumer roles.
        _ = fence_sys()
        cute.arch.sync_threads()

    @cute.jit
    def mark_prefetched_weights_ready(
        self,
        generation: Int32,
        service: Int32,
        weight_arena: cute.Tensor,
        step: cute.Tensor,
    ):
        """Store this rank's READY epochs for the five panels ``prefetch_layer_weights`` copied.

        The kernel calls it after the GEMM's grid barrier, when every prefetching CTA is done.
        """

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        epoch = _weight_publication_epoch(step, service)
        slot = generation - (
            generation // Int32(arenas.LAYER_SLOTS)
        ) * Int32(arenas.LAYER_SLOTS)
        if bidx == Int32(0) and tidx == Int32(0):
            _ = fence_sys()
            for panel_index in cutlass.range(
                0, DECODER_WEIGHT_PANEL_COUNT, 1, unroll=1
            ):
                ready_index = (
                    slot * Int32(DECODER_WEIGHT_PANEL_COUNT) + Int32(panel_index)
                )
                _ = store_release_sys_u32(
                    weight_arena.iterator.toint()
                    + Int64(arenas.DECODER_WEIGHT_READY_OFFSET)
                    + Int64(ready_index) * Int64(4),
                    epoch,
                )

    @cute.jit
    def wait_layer_weights_ready(
        self,
        generation: Int32,
        service: Int32,
        phase_counter: cute.Tensor,
        control: cute.Tensor,
        step: cute.Tensor,
        status: cute.Tensor,
    ):
        """Wait until every rank has published READY for layer ``generation``'s five panels.

        A grid barrier follows.
        """

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        epoch = _weight_publication_epoch(step, service)
        slot = generation - (
            generation // Int32(arenas.LAYER_SLOTS)
        ) * Int32(arenas.LAYER_SLOTS)
        if bidx == Int32(0) and tidx == Int32(0):
            for panel_index in cutlass.range(
                0, DECODER_WEIGHT_PANEL_COUNT, 1, unroll=1
            ):
                ready_index = (
                    slot * Int32(DECODER_WEIGHT_PANEL_COUNT) + Int32(panel_index)
                )
                _wait_for_epoch_on_all_ranks(
                    Int64(control[DECODER_CONTROL_WEIGHT_MULTICAST_BASE])
                    + Int64(arenas.DECODER_WEIGHT_READY_OFFSET)
                    + Int64(ready_index) * Int64(4),
                    epoch,
                    status,
                    Int32(DECODER_STATUS_WEIGHT_READY_TIMEOUT),
                    service,
                    Int32(panel_index),
                    Int64(control[DECODER_CONTROL_TIMEOUT_NS]),
                )
        self.grid_phase_barrier(phase_counter)

    @cute.jit
    def release_layer_weights(
        self,
        generation: Int32,
        service: Int32,
        phase_counter: cute.Tensor,
        weight_arena: cute.Tensor,
        step: cute.Tensor,
    ):
        """Mark layer ``generation``'s weight slot consumed on this rank after its last read.

        Publishers wait for every rank's CONSUMED epoch before they overwrite the slot.
        """

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        epoch = _weight_publication_epoch(step, service)
        slot = generation - (
            generation // Int32(arenas.LAYER_SLOTS)
        ) * Int32(arenas.LAYER_SLOTS)
        if bidx == Int32(0) and tidx == Int32(0):
            _ = fence_sys()
            _ = store_release_sys_u32(
                weight_arena.iterator.toint()
                + Int64(arenas.DECODER_WEIGHT_CONSUMED_OFFSET)
                + Int64(slot) * Int64(4),
                epoch,
            )
        self.grid_phase_barrier(phase_counter)

    @cute.jit
    def publish_layer_gradients(
        self,
        generation: Int32,
        reverse_visit: Int32,
        phase_counter: cute.Tensor,
        grad_arena: cute.Tensor,
        step: cute.Tensor,
    ):
        """Mark layer ``generation``'s gradients in this rank's gradient-ring slot ready.

        The backward's dW GEMMs and norm reductions write the full gradients straight into slot
        ``generation % 2``; ``reduce_scatter_layer_gradients`` reduces them one reverse visit
        later.
        """

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        epoch = _gradient_publication_epoch(step, reverse_visit)
        slot = generation - (
            generation // Int32(arenas.LAYER_SLOTS)
        ) * Int32(arenas.LAYER_SLOTS)

        # Every thread fences its gradient stores before CTA 0 publishes READY.  No barrier
        # follows: the next reverse visit computes in the other slot while the other ranks
        # publish this layer.
        _ = fence_sys()
        self.grid_phase_barrier(phase_counter)
        if bidx == Int32(0) and tidx == Int32(0):
            for site_index in cutlass.range(
                0, DECODER_GRADIENT_SITE_COUNT, 1, unroll=1
            ):
                ready_index = slot * Int32(DECODER_GRADIENT_SITE_COUNT) + Int32(site_index)
                _ = store_release_sys_u32(
                    grad_arena.iterator.toint()
                    + Int64(arenas.DECODER_GRADIENT_READY_OFFSET)
                    + Int64(ready_index) * Int64(4),
                    epoch,
                )

    @cute.jit
    def reduce_scatter_layer_gradients(
        self,
        generation: Int32,
        reverse_visit: Int32,
        phase_counter: cute.Tensor,
        grad_arena: cute.Tensor,
        destination_table: cute.Tensor,
        control: cute.Tensor,
        step: cute.Tensor,
        status: cute.Tensor,
    ):
        """Reduce-scatter layer ``generation``'s gradients, published one reverse visit earlier.

        CTA 0 waits until every rank has marked the slot ready; the grid then sums this rank's
        share of each of the five sites over the ranks (a multimem load-reduce) into the
        optimizer's gradient (``destination_table``).  Last, CTA 0 marks the slot done and
        waits until every rank has, so no rank overwrites a slot another is still reading.
        """

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        epoch = _gradient_publication_epoch(step, reverse_visit)
        slot = generation - (
            generation // Int32(arenas.LAYER_SLOTS)
        ) * Int32(arenas.LAYER_SLOTS)

        if bidx == Int32(0) and tidx == Int32(0):
            for site_index in cutlass.range(
                0, DECODER_GRADIENT_SITE_COUNT, 1, unroll=1
            ):
                ready_index = slot * Int32(DECODER_GRADIENT_SITE_COUNT) + Int32(site_index)
                _wait_for_epoch_on_all_ranks(
                    Int64(control[DECODER_CONTROL_GRADIENT_MULTICAST_BASE])
                    + Int64(arenas.DECODER_GRADIENT_READY_OFFSET)
                    + Int64(ready_index) * Int64(4),
                    epoch,
                    status,
                    Int32(DECODER_STATUS_GRADIENT_READY_TIMEOUT),
                    reverse_visit,
                    Int32(site_index),
                    Int64(control[DECODER_CONTROL_TIMEOUT_NS]),
                )
        self.grid_phase_barrier(phase_counter)

        memspace = destination_table.iterator.memspace
        global_thread = bidx * Int32(model.PROGRAM_THREADS) + tidx
        global_threads = Int32(model.PROGRAM_CTAS * model.PROGRAM_THREADS)
        rank = Int32(control[DECODER_CONTROL_RANK])
        for site_index in cutlass.range(0, DECODER_GRADIENT_SITE_COUNT, 1, unroll=1):
            owner_pairs, relative_offset = _gradient_site_owner_extent(site_index)
            destination = anchors.row_table_view(
                destination_table,
                memspace,
                generation * Int32(DECODER_GRADIENT_SITE_COUNT) + site_index,
                Float32,
                cute.make_layout(1),
            )
            owner_begin = Int64(rank) * Int64(owner_pairs) * Int64(2)
            for pair in cutlass.range(
                global_thread,
                owner_pairs,
                global_threads,
                unroll=1,
            ):
                source_element = owner_begin + Int64(pair) * Int64(2)
                source_address = (
                    Int64(control[DECODER_CONTROL_GRADIENT_MULTICAST_BASE])
                    + Int64(arenas.DECODER_GRADIENT_PAYLOAD_OFFSET)
                    + Int64(slot) * Int64(arenas.DECODER_GRADIENT_SLOT_STRIDE)
                    + relative_offset
                    + source_element * Int64(model.FP32_BYTES)
                )
                value = multicast_sum_2xf32(source_address)
                _ = store_b64(
                    destination.iterator.toint()
                    + Int64(pair) * Int64(2 * model.FP32_BYTES),
                    value,
                )
        self.grid_phase_barrier(phase_counter)

        if bidx == Int32(0) and tidx == Int32(0):
            _ = fence_sys()
            _ = store_release_sys_u32(
                grad_arena.iterator.toint()
                + Int64(arenas.DECODER_GRADIENT_DONE_OFFSET)
                + Int64(slot) * Int64(4),
                epoch,
            )
            _wait_for_epoch_on_all_ranks(
                Int64(control[DECODER_CONTROL_GRADIENT_MULTICAST_BASE])
                + Int64(arenas.DECODER_GRADIENT_DONE_OFFSET)
                + Int64(slot) * Int64(4),
                epoch,
                status,
                Int32(DECODER_STATUS_GRADIENT_DONE_TIMEOUT),
                reverse_visit,
                Int32(-1),
                Int64(control[DECODER_CONTROL_TIMEOUT_NS]),
            )
        self.grid_phase_barrier(phase_counter)

    @cute.jit
    def all_reduce_valid_tokens(
        self,
        phase_counter: cute.Tensor,
        valid_arena: cute.Tensor,
        control: cute.Tensor,
        local_valid_tokens: cute.Tensor,
        global_valid_tokens: cute.Tensor,
        status: cute.Tensor,
        integer_control: cute.Tensor,
    ):
        """All-reduce the ranks' valid-token counts into ``global_valid_tokens`` for the loss.

        The all-reduce arena's two scalar slots alternate by step; before reusing one, CTA 0
        waits until every rank has consumed the value published there two steps earlier.
        """

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        epoch = Int32(integer_control[clipped_adamw.CONTROL_STEP])
        slot = epoch - (epoch // Int32(arenas.ALL_REDUCE_SLOTS)) * Int32(arenas.ALL_REDUCE_SLOTS)
        if bidx == Int32(0) and tidx == Int32(0):
            if epoch > Int32(arenas.ALL_REDUCE_SLOTS):
                _wait_for_epoch_on_all_ranks(
                    Int64(control[shell_fabric.SHELL_CONTROL_VALID_TOKENS_MULTICAST])
                    + Int64(arenas.ALL_REDUCE_SCALAR_CONSUMED_OFFSET)
                    + Int64(slot) * Int64(4),
                    epoch - Int32(arenas.ALL_REDUCE_SLOTS),
                    status,
                    Int32(shell_fabric.SHELL_STATUS_VALID_TOKENS_REUSE_TIMEOUT),
                    Int32(0),
                    Int32(-1),
                    Int64(control[shell_fabric.SHELL_CONTROL_TIMEOUT_NS]),
                )
            _ = store_f32(
                valid_arena.iterator.toint()
                + Int64(arenas.ALL_REDUCE_SCALAR_LOCAL_OFFSET)
                + Int64(slot) * Int64(4),
                Float32(local_valid_tokens[0]),
            )
            _ = fence_sys()
            _ = store_release_sys_u32(
                valid_arena.iterator.toint()
                + Int64(arenas.ALL_REDUCE_SCALAR_READY_OFFSET)
                + Int64(slot) * Int64(4),
                epoch,
            )
        self.grid_phase_barrier(phase_counter)

        if bidx == Int32(0) and tidx == Int32(0):
            _wait_for_epoch_on_all_ranks(
                Int64(control[shell_fabric.SHELL_CONTROL_VALID_TOKENS_MULTICAST])
                + Int64(arenas.ALL_REDUCE_SCALAR_READY_OFFSET)
                + Int64(slot) * Int64(4),
                epoch,
                status,
                Int32(shell_fabric.SHELL_STATUS_VALID_TOKENS_READY_TIMEOUT),
                Int32(0),
                Int32(-1),
                Int64(control[shell_fabric.SHELL_CONTROL_TIMEOUT_NS]),
            )
            reduced = multicast_sum_f32(
                Int64(control[shell_fabric.SHELL_CONTROL_VALID_TOKENS_MULTICAST])
                + Int64(arenas.ALL_REDUCE_SCALAR_LOCAL_OFFSET)
                + Int64(slot) * Int64(4)
            )
            global_valid_tokens[0] = Int32(reduced)
            _ = fence_sys()
            _ = store_release_sys_u32(
                valid_arena.iterator.toint()
                + Int64(arenas.ALL_REDUCE_SCALAR_CONSUMED_OFFSET)
                + Int64(slot) * Int64(4),
                epoch,
            )
            _ = store_release_sys_u32(
                valid_arena.iterator.toint() + Int64(arenas.ALL_REDUCE_SCALAR_DONE_OFFSET),
                epoch,
            )
        self.grid_phase_barrier(phase_counter)

    @cute.jit
    def all_gather_head_weight(
        self,
        kind: cutlass.Constexpr[int],
        phase_counter: cute.Tensor,
        row_table: cute.Tensor,
        weight_arena: cute.Tensor,
        control: cute.Tensor,
        input_ids: cute.Tensor,
        head_weight: cute.Tensor,
        optimizer_bf16: cute.Tensor,
        status: cute.Tensor,
        integer_control: cute.Tensor,
    ):
        """All-gather a vocabulary matrix from the ranks' owner rows through the head weight ring.

        Panel by panel, each rank copies its owner rows of the BF16 parameter into the ring's
        slot on every rank; after every rank's READY, each rank reads the panel back.  ``kind``
        1 gathers the head weight into ``head_weight``; ``kind`` 0 would gather the embedding
        rows of ``input_ids`` into layer 0's input, but the kernel calls only kind 1 and
        gathers the embedding with ``gather_embedding_rows``.  Each step numbers the ring's
        epochs with the embedding's panels first, then the head's.
        """

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        global_thread = bidx * Int32(model.PROGRAM_THREADS) + tidx
        global_threads = Int32(model.PROGRAM_CTAS * model.PROGRAM_THREADS)
        rank = Int32(control[shell_fabric.SHELL_CONTROL_RANK])
        step = Int32(integer_control[clipped_adamw.CONTROL_STEP])
        owner_begin = rank * Int32(arenas.VOCAB_ROWS_PER_RANK)
        owner_end = owner_begin + Int32(arenas.VOCAB_ROWS_PER_RANK)
        if const_expr(kind == 0):
            segment_begin = Int64(optimizer_state.OWNER_SEGMENT_BY_NAME["embedding"].begin)
        else:
            segment_begin = Int64(optimizer_state.OWNER_SEGMENT_BY_NAME["head"].begin)

        memspace = row_table.iterator.memspace
        embedding_out = anchors.row_table_view(
            row_table,
            memspace,
            Int32(decoder_layer.ROW_TABLE_INDEX["residual_in"]),
            BFloat16,
            cute.make_layout(model.SEQUENCE * model.HIDDEN),
        )
        for panel in cutlass.range(0, arenas.VOCAB_PANELS, 1, unroll=1):
            position = Int32(kind * arenas.VOCAB_PANELS) + panel
            epoch = (
                (step - Int32(1)) * Int32(shell_fabric.VOCAB_MATRICES * arenas.VOCAB_PANELS)
                + position
                + Int32(1)
            )
            slot = epoch - (epoch // Int32(arenas.HEAD_WEIGHT_SLOTS)) * Int32(
                arenas.HEAD_WEIGHT_SLOTS
            )
            if bidx == Int32(0) and tidx == Int32(0):
                if epoch > Int32(arenas.HEAD_WEIGHT_SLOTS):
                    _wait_for_epoch_on_all_ranks(
                        Int64(control[shell_fabric.SHELL_CONTROL_HEAD_WEIGHT_MULTICAST])
                        + Int64(arenas.HEAD_WEIGHT_CONSUMED_OFFSET)
                        + Int64(slot) * Int64(4),
                        epoch - Int32(arenas.HEAD_WEIGHT_SLOTS),
                        status,
                        Int32(shell_fabric.SHELL_STATUS_WEIGHT_REUSE_TIMEOUT),
                        Int32(kind),
                        panel,
                        Int64(control[shell_fabric.SHELL_CONTROL_TIMEOUT_NS]),
                    )
            self.grid_phase_barrier(phase_counter)

            panel_begin = panel * Int32(arenas.VOCAB_PANEL_ROWS)
            panel_end = panel_begin + Int32(arenas.VOCAB_PANEL_ROWS)
            if panel_end > Int32(model.VOCAB):
                panel_end = Int32(model.VOCAB)
            begin = panel_begin
            if owner_begin > begin:
                begin = owner_begin
            end = panel_end
            if owner_end < end:
                end = owner_end
            if begin < end:
                rows = end - begin
                source_row = begin - owner_begin
                panel_row = begin - panel_begin
                for word in cutlass.range(
                    global_thread,
                    rows * Int32(model.HIDDEN // 4),
                    global_threads,
                    unroll=1,
                ):
                    source_element = (
                        segment_begin
                        + Int64(source_row) * Int64(model.HIDDEN)
                        + Int64(word) * Int64(4)
                    )
                    destination = (
                        Int64(control[shell_fabric.SHELL_CONTROL_HEAD_WEIGHT_MULTICAST])
                        + Int64(arenas.HEAD_WEIGHT_PAYLOAD_OFFSET)
                        + Int64(slot) * Int64(arenas.HEAD_WEIGHT_SLOT_STRIDE)
                        + (
                            Int64(panel_row) * Int64(model.HIDDEN)
                            + Int64(word) * Int64(4)
                        )
                        * Int64(model.BF16_BYTES)
                    )
                    value = load_b64(
                        optimizer_bf16.iterator.toint()
                        + source_element * Int64(model.BF16_BYTES)
                    )
                    _ = multicast_store_b64(destination, value)
            self.grid_phase_barrier(phase_counter)

            if bidx == Int32(0) and tidx == Int32(0):
                _ = fence_sys()
                _ = store_release_sys_u32(
                    weight_arena.iterator.toint()
                    + Int64(arenas.HEAD_WEIGHT_READY_OFFSET)
                    + Int64(slot) * Int64(4),
                    epoch,
                )
                _wait_for_epoch_on_all_ranks(
                    Int64(control[shell_fabric.SHELL_CONTROL_HEAD_WEIGHT_MULTICAST])
                    + Int64(arenas.HEAD_WEIGHT_READY_OFFSET)
                    + Int64(slot) * Int64(4),
                    epoch,
                    status,
                    Int32(shell_fabric.SHELL_STATUS_WEIGHT_READY_TIMEOUT),
                    Int32(kind),
                    panel,
                    Int64(control[shell_fabric.SHELL_CONTROL_TIMEOUT_NS]),
                )
            self.grid_phase_barrier(phase_counter)

            if const_expr(kind == 0):
                words_per_token = Int32(model.HIDDEN // 4)
                for word in cutlass.range(
                    global_thread,
                    Int32(model.SEQUENCE * (model.HIDDEN // 4)),
                    global_threads,
                    unroll=1,
                ):
                    token = word // words_per_token
                    hidden_word = word - token * words_per_token
                    token_id = Int32(input_ids[token])
                    if token_id >= panel_begin and token_id < panel_end:
                        panel_row = token_id - panel_begin
                        value = load_b64(
                            weight_arena.iterator.toint()
                            + Int64(arenas.HEAD_WEIGHT_PAYLOAD_OFFSET)
                            + Int64(slot) * Int64(arenas.HEAD_WEIGHT_SLOT_STRIDE)
                            + (
                                Int64(panel_row) * Int64(model.HIDDEN)
                                + Int64(hidden_word) * Int64(4)
                            )
                            * Int64(model.BF16_BYTES)
                        )
                        _ = store_b64(
                            embedding_out.iterator.toint()
                            + Int64(word) * Int64(4 * model.BF16_BYTES),
                            value,
                        )
            else:
                panel_rows = panel_end - panel_begin
                for word in cutlass.range(
                    global_thread,
                    panel_rows * Int32(model.HIDDEN // 4),
                    global_threads,
                    unroll=1,
                ):
                    value = load_b64(
                        weight_arena.iterator.toint()
                        + Int64(arenas.HEAD_WEIGHT_PAYLOAD_OFFSET)
                        + Int64(slot) * Int64(arenas.HEAD_WEIGHT_SLOT_STRIDE)
                        + Int64(word) * Int64(4 * model.BF16_BYTES)
                    )
                    _ = store_b64(
                        head_weight.iterator.toint()
                        + (
                            Int64(panel_begin) * Int64(model.HIDDEN)
                            + Int64(word) * Int64(4)
                        )
                        * Int64(model.BF16_BYTES),
                        value,
                    )
            self.grid_phase_barrier(phase_counter)

            if bidx == Int32(0) and tidx == Int32(0):
                _ = fence_sys()
                _ = store_release_sys_u32(
                    weight_arena.iterator.toint()
                    + Int64(arenas.HEAD_WEIGHT_CONSUMED_OFFSET)
                    + Int64(slot) * Int64(4),
                    epoch,
                )
            self.grid_phase_barrier(phase_counter)

    @cute.jit
    def gather_embedding_rows(
        self,
        phase_counter: cute.Tensor,
        row_table: cute.Tensor,
        weight_arena: cute.Tensor,
        control: cute.Tensor,
        input_ids: cute.Tensor,
        optimizer_bf16: cute.Tensor,
        integer_control: cute.Tensor,
        route_arena: cute.Tensor,
        route_peer_bases: cute.Tensor,
        route_unique_ids: cute.Tensor,
        route_inverse: cute.Tensor,
        route_owner_offsets: cute.Tensor,
        route_owner_counts: cute.Tensor,
        route_status: cute.Tensor,
    ):
        """Gather this step's embedding rows into layer 0's input through the embedding route.

        Each rank sends every owner rank the distinct token ids it needs from that owner
        (``route_unique_ids``, one sorted slice per owner), serves the other ranks' requests
        from its own rows of the BF16 parameter, then expands the received rows to one row per
        token (``route_inverse``).  Last, it advances the head weight ring past the embedding's
        panel epochs, which the route replaces.
        """

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        global_thread = bidx * Int32(model.PROGRAM_THREADS) + tidx
        global_threads = Int32(model.PROGRAM_CTAS * model.PROGRAM_THREADS)
        rank = Int32(control[shell_fabric.SHELL_CONTROL_RANK])
        step = Int32(integer_control[clipped_adamw.CONTROL_STEP])
        timeout_ns = Int64(control[shell_fabric.SHELL_CONTROL_TIMEOUT_NS])

        # Write each owner the sorted row ids this rank needs from it.  The grid barrier
        # completes every store before CTA 0 publishes the counts and the request epoch.
        for owner_index in cutlass.range_constexpr(model.WORLD):
            owner_base = Int64(route_peer_bases[owner_index])
            source_offset = Int32(route_owner_offsets[owner_index])
            count = Int32(route_owner_counts[owner_index])
            for record in cutlass.range(
                global_thread, count, global_threads, unroll=1
            ):
                row_id = load_u64(
                    route_unique_ids.iterator.toint()
                    + Int64(source_offset + record) * Int64(8)
                )
                _ = store_u64(
                    owner_base
                    + Int64(embedding_route.ROUTE_ROW_IDS_OFFSET)
                    + (
                        Int64(rank) * Int64(embedding_route.ROUTE_RECORDS) + Int64(record)
                    )
                    * Int64(8),
                    row_id,
                )
        self.grid_phase_barrier(phase_counter)

        if bidx == Int32(0) and tidx == Int32(0):
            _ = fence_sys()
            for owner_index in cutlass.range_constexpr(model.WORLD):
                owner_base = Int64(route_peer_bases[owner_index])
                _ = store_release_sys_u32(
                    owner_base
                    + Int64(embedding_route.ROUTE_COUNTS_OFFSET)
                    + Int64(rank) * Int64(4),
                    Int32(route_owner_counts[owner_index]),
                )
                _ = store_release_sys_u32(
                    owner_base
                    + Int64(EMBEDDING_ROUTE_FORWARD_REQUEST)
                    + Int64(rank) * Int64(4),
                    step,
                )
            # Every rank publishes its requests before it waits for the others', so the
            # exchange cannot deadlock.
            for source_index in cutlass.range_constexpr(model.WORLD):
                embedding_route.wait_route_epoch_equal(
                    route_arena.iterator.toint()
                    + Int64(EMBEDDING_ROUTE_FORWARD_REQUEST)
                    + Int64(source_index) * Int64(4),
                    step,
                    route_status,
                    Int32(0),
                    Int32(source_index),
                    timeout_ns,
                )
        self.grid_phase_barrier(phase_counter)

        # Serve every requester from this rank's owner rows, four BF16 values per 64-bit store,
        # into the requester's route payload.  BF16 rows fill only the first half of the
        # payload, which is sized for the gradient's FP32 rows.
        embedding_segment = Int64(optimizer_state.OWNER_SEGMENT_BY_NAME["embedding"].begin)
        owner_begin = rank * Int32(arenas.VOCAB_ROWS_PER_RANK)
        for source_index in cutlass.range_constexpr(model.WORLD):
            source_base = Int64(route_peer_bases[source_index])
            count = load_acquire_sys_u32(
                route_arena.iterator.toint()
                + Int64(embedding_route.ROUTE_COUNTS_OFFSET)
                + Int64(source_index) * Int64(4)
            )
            word_count = count * Int32(model.HIDDEN // 4)
            for word in cutlass.range(
                global_thread, word_count, global_threads, unroll=1
            ):
                record = word // Int32(model.HIDDEN // 4)
                hidden_word = word - record * Int32(model.HIDDEN // 4)
                row_id = Int32(
                    load_u64(
                        route_arena.iterator.toint()
                        + Int64(embedding_route.ROUTE_ROW_IDS_OFFSET)
                        + (
                            Int64(source_index) * Int64(embedding_route.ROUTE_RECORDS)
                            + Int64(record)
                        )
                        * Int64(8)
                    )
                )
                local_row = row_id - owner_begin
                value = load_b64(
                    optimizer_bf16.iterator.toint()
                    + (
                        embedding_segment
                        + Int64(local_row) * Int64(model.HIDDEN)
                        + Int64(hidden_word) * Int64(4)
                    )
                    * Int64(model.BF16_BYTES)
                )
                _ = store_b64(
                    source_base
                    + Int64(embedding_route.ROUTE_PAYLOAD_OFFSET)
                    + (
                        Int64(rank) * Int64(embedding_route.ROUTE_PAYLOAD_ELEMENTS_PER_SOURCE)
                        + Int64(record) * Int64(model.HIDDEN)
                        + Int64(hidden_word) * Int64(4)
                    )
                    * Int64(model.BF16_BYTES),
                    value,
                )
        self.grid_phase_barrier(phase_counter)

        if bidx == Int32(0) and tidx == Int32(0):
            _ = fence_sys()
            for source_index in cutlass.range_constexpr(model.WORLD):
                source_base = Int64(route_peer_bases[source_index])
                _ = store_release_sys_u32(
                    source_base
                    + Int64(EMBEDDING_ROUTE_FORWARD_RESPONSE)
                    + Int64(rank) * Int64(4),
                    step,
                )
            for owner_index in cutlass.range_constexpr(model.WORLD):
                embedding_route.wait_route_epoch_equal(
                    route_arena.iterator.toint()
                    + Int64(EMBEDDING_ROUTE_FORWARD_RESPONSE)
                    + Int64(owner_index) * Int64(4),
                    step,
                    route_status,
                    Int32(0),
                    Int32(owner_index),
                    timeout_ns,
                )
        self.grid_phase_barrier(phase_counter)

        memspace = row_table.iterator.memspace
        embedding_out = anchors.row_table_view(
            row_table,
            memspace,
            Int32(decoder_layer.ROW_TABLE_INDEX["residual_in"]),
            BFloat16,
            cute.make_layout(model.SEQUENCE * model.HIDDEN),
        )
        words_per_token = Int32(model.HIDDEN // 4)
        for word in cutlass.range(
            global_thread,
            Int32(model.SEQUENCE * (model.HIDDEN // 4)),
            global_threads,
            unroll=1,
        ):
            token = word // words_per_token
            hidden_word = word - token * words_per_token
            token_id = Int32(input_ids[token])
            owner = token_id // Int32(arenas.VOCAB_ROWS_PER_RANK)
            record = Int32(route_inverse[token]) - Int32(
                route_owner_offsets[owner]
            )
            value = load_b64(
                route_arena.iterator.toint()
                + Int64(embedding_route.ROUTE_PAYLOAD_OFFSET)
                + (
                    Int64(owner) * Int64(embedding_route.ROUTE_PAYLOAD_ELEMENTS_PER_SOURCE)
                    + Int64(record) * Int64(model.HIDDEN)
                    + Int64(hidden_word) * Int64(4)
                )
                * Int64(model.BF16_BYTES)
            )
            _ = store_b64(
                embedding_out.iterator.toint()
                + Int64(word) * Int64(4 * model.BF16_BYTES),
                value,
            )
        self.grid_phase_barrier(phase_counter)

        # The route replaces the head weight ring's VOCAB_PANELS embedding epochs, which come
        # first in each step.  Store their READY and CONSUMED epochs as if the panels had gone
        # through the ring, so all_gather_head_weight's reuse waits hold.
        if bidx == Int32(0) and tidx == Int32(0):
            for panel_index in cutlass.range_constexpr(arenas.VOCAB_PANELS):
                epoch = (
                    (step - Int32(1))
                    * Int32(shell_fabric.VOCAB_MATRICES * arenas.VOCAB_PANELS)
                    + Int32(panel_index)
                    + Int32(1)
                )
                slot = epoch - (epoch // Int32(arenas.HEAD_WEIGHT_SLOTS)) * Int32(
                    arenas.HEAD_WEIGHT_SLOTS
                )
                _ = store_release_sys_u32(
                    weight_arena.iterator.toint()
                    + Int64(arenas.HEAD_WEIGHT_READY_OFFSET)
                    + Int64(slot) * Int64(4),
                    epoch,
                )
                _ = store_release_sys_u32(
                    weight_arena.iterator.toint()
                    + Int64(arenas.HEAD_WEIGHT_CONSUMED_OFFSET)
                    + Int64(slot) * Int64(4),
                    epoch,
                )

    @cute.jit
    def reduce_scatter_head_gradient(
        self,
        phase_counter: cute.Tensor,
        gradient_arena: cute.Tensor,
        control: cute.Tensor,
        optimizer_gradient: cute.Tensor,
        status: cute.Tensor,
        integer_control: cute.Tensor,
    ):
        """Sum every rank's head dW over this rank's owner rows into the optimizer's gradient.

        The head dW output is symmetric memory, read through its multicast address.  The head
        gradient ring's READY and CONSUMED words, at the epoch of the step's last head panel,
        serve as the rendezvous before and after the sum; the ring's payload is not used.
        """

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        global_thread = bidx * Int32(model.PROGRAM_THREADS) + tidx
        global_threads = Int32(model.PROGRAM_CTAS * model.PROGRAM_THREADS)
        rank = Int32(control[shell_fabric.SHELL_CONTROL_RANK])
        step = Int32(integer_control[clipped_adamw.CONTROL_STEP])
        final_position = Int32(arenas.VOCAB_PANELS - 1)
        final_epoch = (
            (step - Int32(1))
            * Int32(shell_fabric.VOCAB_MATRICES * arenas.VOCAB_PANELS)
            + Int32(arenas.VOCAB_PANELS)
        )
        final_slot = final_epoch - (
            final_epoch // Int32(arenas.HEAD_GRADIENT_SLOTS)
        ) * Int32(arenas.HEAD_GRADIENT_SLOTS)

        # Wait until every rank has finished the previous step's ring epochs, then until every
        # rank's head dW is complete (READY at this step's last head panel epoch).
        if bidx == Int32(0) and tidx == Int32(0):
            if step > Int32(1):
                _wait_for_epoch_on_all_ranks(
                    Int64(control[shell_fabric.SHELL_CONTROL_HEAD_GRADIENT_MULTICAST])
                    + Int64(arenas.HEAD_GRADIENT_CONSUMED_OFFSET)
                    + Int64(final_slot) * Int64(4),
                    (step - Int32(1))
                    * Int32(shell_fabric.VOCAB_MATRICES * arenas.VOCAB_PANELS),
                    status,
                    Int32(shell_fabric.SHELL_STATUS_GRADIENT_REUSE_TIMEOUT),
                    Int32(0),
                    final_position,
                    Int64(control[shell_fabric.SHELL_CONTROL_TIMEOUT_NS]),
                )
            _ = fence_sys()
            _ = store_release_sys_u32(
                gradient_arena.iterator.toint()
                + Int64(arenas.HEAD_GRADIENT_READY_OFFSET)
                + Int64(final_slot) * Int64(4),
                final_epoch,
            )
            _wait_for_epoch_on_all_ranks(
                Int64(control[shell_fabric.SHELL_CONTROL_HEAD_GRADIENT_MULTICAST])
                + Int64(arenas.HEAD_GRADIENT_READY_OFFSET)
                + Int64(final_slot) * Int64(4),
                final_epoch,
                status,
                Int32(shell_fabric.SHELL_STATUS_GRADIENT_READY_TIMEOUT),
                Int32(0),
                final_position,
                Int64(control[shell_fabric.SHELL_CONTROL_TIMEOUT_NS]),
            )
        self.grid_phase_barrier(phase_counter)

        owner_elements = Int64(arenas.VOCAB_ROWS_PER_RANK * model.HIDDEN)
        source_element = Int64(rank) * owner_elements
        segment_begin = Int64(optimizer_state.OWNER_SEGMENT_BY_NAME["head"].begin)
        for pair in cutlass.range(
            global_thread,
            Int32(arenas.VOCAB_ROWS_PER_RANK * model.HIDDEN // 2),
            global_threads,
            unroll=4,
        ):
            value = multicast_sum_2xf32(
                Int64(control[shell_fabric.SHELL_CONTROL_HEAD_DW_MULTICAST])
                + (source_element + Int64(pair) * Int64(2))
                * Int64(model.FP32_BYTES)
            )
            _ = store_b64(
                optimizer_gradient.iterator.toint()
                + (segment_begin + Int64(pair) * Int64(2))
                * Int64(model.FP32_BYTES),
                value,
            )
        self.grid_phase_barrier(phase_counter)

        # Wait until every rank has read every rank's head dW, so no next head dW overwrites
        # one early.  Then advance the ring's READY and CONSUMED words through each head panel
        # epoch, as if the panels had gone through the ring.
        if bidx == Int32(0) and tidx == Int32(0):
            _ = fence_sys()
            _ = store_release_sys_u32(
                gradient_arena.iterator.toint()
                + Int64(arenas.HEAD_GRADIENT_CONSUMED_OFFSET)
                + Int64(final_slot) * Int64(4),
                final_epoch,
            )
            _wait_for_epoch_on_all_ranks(
                Int64(control[shell_fabric.SHELL_CONTROL_HEAD_GRADIENT_MULTICAST])
                + Int64(arenas.HEAD_GRADIENT_CONSUMED_OFFSET)
                + Int64(final_slot) * Int64(4),
                final_epoch,
                status,
                Int32(shell_fabric.SHELL_STATUS_GRADIENT_REUSE_TIMEOUT),
                Int32(0),
                final_position,
                Int64(control[shell_fabric.SHELL_CONTROL_TIMEOUT_NS]),
            )
            for panel_index in cutlass.range_constexpr(arenas.VOCAB_PANELS):
                position = Int32(panel_index)
                virtual_epoch = (
                    (step - Int32(1))
                    * Int32(shell_fabric.VOCAB_MATRICES * arenas.VOCAB_PANELS)
                    + position
                    + Int32(1)
                )
                slot = virtual_epoch - (
                    virtual_epoch // Int32(arenas.HEAD_GRADIENT_SLOTS)
                ) * Int32(arenas.HEAD_GRADIENT_SLOTS)
                _ = store_release_sys_u32(
                    gradient_arena.iterator.toint()
                    + Int64(arenas.HEAD_GRADIENT_READY_OFFSET)
                    + Int64(slot) * Int64(4),
                    virtual_epoch,
                )
                _ = store_release_sys_u32(
                    gradient_arena.iterator.toint()
                    + Int64(arenas.HEAD_GRADIENT_CONSUMED_OFFSET)
                    + Int64(slot) * Int64(4),
                    virtual_epoch,
                )
        self.grid_phase_barrier(phase_counter)

    @cute.jit
    def reduce_scatter_embedding_gradient(
        self,
        phase_counter: cute.Tensor,
        row_table: cute.Tensor,
        gradient_scratch: cute.Tensor,
        optimizer_gradient: cute.Tensor,
        gradient_arena: cute.Tensor,
        shell_fabric_control: cute.Tensor,
        optimizer_integer_control: cute.Tensor,
        route_arena: cute.Tensor,
        route_peer_bases: cute.Tensor,
        route_unique_ids: cute.Tensor,
        route_inverse: cute.Tensor,
        route_owner_offsets: cute.Tensor,
        route_owner_counts: cute.Tensor,
        route_status: cute.Tensor,
        route_lookup_scratch: cute.Tensor,
    ):
        """Reduce-scatter the embedding gradient through the embedding route.

        Each token's dX row (layer 0's input gradient) is added into one FP32 row per distinct
        token id in ``gradient_scratch``, the head dW buffer, free once
        ``reduce_scatter_head_gradient`` has run.  Each owner rank then receives the rows it
        owns, and sums every rank's copy of each of its embedding rows, in source-rank order,
        into the optimizer gradient's embedding segment.  ``route_lookup_scratch`` is shared
        memory for one record index per source rank.
        """

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        global_thread = bidx * Int32(model.PROGRAM_THREADS) + tidx
        global_threads = Int32(model.PROGRAM_CTAS * model.PROGRAM_THREADS)
        rank = Int32(shell_fabric_control[shell_fabric.SHELL_CONTROL_RANK])
        epoch = Int32(optimizer_integer_control[clipped_adamw.CONTROL_STEP])
        timeout_ns = Int64(shell_fabric_control[shell_fabric.SHELL_CONTROL_TIMEOUT_NS])
        unique_count = Int32(0)
        for owner_index in cutlass.range_constexpr(model.WORLD):
            unique_count += Int32(route_owner_counts[owner_index])

        # The records below are addressed as a flat row-major array from the
        # scratch's base address.  The scratch arrives as the head dWeight's
        # (VOCAB, HIDDEN, 1) tensor, whose own layout would map a linear index
        # to a column-major walk, so zero it through a flat view.
        scratch = cute.make_tensor(
            gradient_scratch.iterator, cute.make_layout(model.SEQUENCE * model.HIDDEN)
        )
        for element in cutlass.range(
            global_thread,
            unique_count * Int32(model.HIDDEN),
            global_threads,
            unroll=1,
        ):
            scratch[element] = Float32(0.0)
        self.grid_phase_barrier(phase_counter)

        memspace = row_table.iterator.memspace
        embedding_dx = anchors.row_table_view(
            row_table,
            memspace,
            Int32(decoder_layer.ROW_TABLE_INDEX["layer_input_dx"]),
            BFloat16,
            cute.make_layout(model.SEQUENCE * model.HIDDEN),
        )
        for element in cutlass.range(
            global_thread,
            Int32(model.SEQUENCE * model.HIDDEN),
            global_threads,
            unroll=1,
        ):
            token = element // Int32(model.HIDDEN)
            hidden = element - token * Int32(model.HIDDEN)
            record = Int32(route_inverse[token])
            destination = record * Int32(model.HIDDEN) + hidden
            _ = cute.arch.atomic_add(
                gradient_scratch.iterator + destination,
                embedding_dx[element].to(Float32),
                sem="relaxed",
                scope="gpu",
            )
        self.grid_phase_barrier(phase_counter)

        # Send each owner its sorted slice, row ids and FP32 rows, into this rank's inbox in the
        # owner's route arena once the owner has consumed the previous step's.  The grid
        # barrier after the next owner's reuse wait also closes this owner's publication.
        for owner_index in cutlass.range_constexpr(model.WORLD):
            owner = Int32(owner_index)
            source_offset = Int32(route_owner_offsets[owner_index])
            count = Int32(route_owner_counts[owner_index])
            destination_base = Int64(route_peer_bases[owner_index])
            if bidx == Int32(0) and tidx == Int32(0):
                embedding_route.wait_route_epoch_at_least(
                    destination_base
                    + Int64(embedding_route.ROUTE_CONSUMED_OFFSET)
                    + Int64(rank) * Int64(4),
                    epoch - Int32(1),
                    route_status,
                    Int32(embedding_route.ROUTE_STATUS_REUSE_TIMEOUT),
                    Int32(0),
                    owner,
                    timeout_ns,
                )
            self.grid_phase_barrier(phase_counter)

            for record in cutlass.range(
                global_thread, count, global_threads, unroll=1
            ):
                row_id = load_u64(
                    route_unique_ids.iterator.toint()
                    + Int64(source_offset + record) * Int64(8)
                )
                _ = store_u64(
                    destination_base
                    + Int64(embedding_route.ROUTE_ROW_IDS_OFFSET)
                    + (
                        Int64(rank) * Int64(embedding_route.ROUTE_RECORDS) + Int64(record)
                    )
                    * Int64(8),
                    row_id,
                )

            pair_count = count * Int32(model.HIDDEN // 2)
            for pair in cutlass.range(
                global_thread, pair_count, global_threads, unroll=4
            ):
                value = load_b64(
                    gradient_scratch.iterator.toint()
                    + (
                        Int64(source_offset) * Int64(model.HIDDEN)
                        + Int64(pair) * Int64(2)
                    )
                    * Int64(model.FP32_BYTES)
                )
                _ = store_b64(
                    destination_base
                    + Int64(embedding_route.ROUTE_PAYLOAD_OFFSET)
                    + (
                        Int64(rank) * Int64(embedding_route.ROUTE_PAYLOAD_ELEMENTS_PER_SOURCE)
                        + Int64(pair) * Int64(2)
                    )
                    * Int64(model.FP32_BYTES),
                    value,
                )
            self.grid_phase_barrier(phase_counter)

            if bidx == Int32(0) and tidx == Int32(0):
                _ = fence_sys()
                _ = store_release_sys_u32(
                    destination_base
                    + Int64(embedding_route.ROUTE_COUNTS_OFFSET)
                    + Int64(rank) * Int64(4),
                    count,
                )
                _ = store_release_sys_u32(
                    destination_base
                    + Int64(embedding_route.ROUTE_ARRIVAL_OFFSET)
                    + Int64(rank) * Int64(4),
                    epoch,
                )

        # Wait until every rank has delivered its rows to this rank's inboxes.
        if bidx == Int32(0) and tidx == Int32(0):
            for source_index in cutlass.range_constexpr(model.WORLD):
                embedding_route.wait_route_epoch_equal(
                    route_arena.iterator.toint()
                    + Int64(embedding_route.ROUTE_ARRIVAL_OFFSET)
                    + Int64(source_index) * Int64(4),
                    epoch,
                    route_status,
                    Int32(0),
                    Int32(source_index),
                    timeout_ns,
                )
        self.grid_phase_barrier(phase_counter)

        embedding_segment = Int64(optimizer_state.OWNER_SEGMENT_BY_NAME["embedding"].begin)
        for local_row in cutlass.range(
            bidx, arenas.VOCAB_ROWS_PER_RANK, model.PROGRAM_CTAS, unroll=1
        ):
            global_row = rank * Int32(arenas.VOCAB_ROWS_PER_RANK) + local_row

            # Each source's row ids are sorted: one thread per source binary-searches for this
            # row and leaves the record index (or -1) in shared scratch.  The column loop then
            # adds the sources' rows in source-rank order, so the FP32 sum is deterministic.
            if tidx < Int32(model.WORLD):
                source = tidx
                count = load_acquire_sys_u32(
                    route_arena.iterator.toint()
                    + Int64(embedding_route.ROUTE_COUNTS_OFFSET)
                    + Int64(source) * Int64(4)
                )
                lo = Int32(0)
                hi = count
                while lo < hi:
                    mid = (lo + hi) // Int32(2)
                    observed = Int32(
                        load_u64(
                            route_arena.iterator.toint()
                            + Int64(embedding_route.ROUTE_ROW_IDS_OFFSET)
                            + (
                                Int64(source) * Int64(embedding_route.ROUTE_RECORDS)
                                + Int64(mid)
                            )
                            * Int64(8)
                        )
                    )
                    if observed < global_row:
                        lo = mid + Int32(1)
                    else:
                        hi = mid
                match = Int32(-1)
                if lo < count:
                    observed = Int32(
                        load_u64(
                            route_arena.iterator.toint()
                            + Int64(embedding_route.ROUTE_ROW_IDS_OFFSET)
                            + (
                                Int64(source) * Int64(embedding_route.ROUTE_RECORDS)
                                + Int64(lo)
                            )
                            * Int64(8)
                        )
                    )
                    if observed == global_row:
                        match = lo
                route_lookup_scratch[source] = Float32(match)
            cute.arch.sync_threads()

            for column in cutlass.range(
                tidx, model.HIDDEN, model.PROGRAM_THREADS, unroll=1
            ):
                reduced = Float32(0.0)
                for source_index in cutlass.range_constexpr(model.WORLD):
                    source = Int32(source_index)
                    match = Int32(route_lookup_scratch[source_index])
                    if match >= Int32(0):
                        reduced += load_f32(
                            route_arena.iterator.toint()
                            + Int64(embedding_route.ROUTE_PAYLOAD_OFFSET)
                            + (
                                Int64(source)
                                * Int64(embedding_route.ROUTE_PAYLOAD_ELEMENTS_PER_SOURCE)
                                + Int64(match) * Int64(model.HIDDEN)
                                + Int64(column)
                            )
                            * Int64(model.FP32_BYTES)
                        )
                _ = store_f32(
                    optimizer_gradient.iterator.toint()
                    + (
                        embedding_segment
                        + Int64(local_row) * Int64(model.HIDDEN)
                        + Int64(column)
                    )
                    * Int64(model.FP32_BYTES),
                    reduced,
                )
            cute.arch.sync_threads()
        self.grid_phase_barrier(phase_counter)

        if bidx == Int32(0) and tidx == Int32(0):
            _ = fence_sys()
            for source_index in cutlass.range_constexpr(model.WORLD):
                _ = store_release_sys_u32(
                    route_arena.iterator.toint()
                    + Int64(embedding_route.ROUTE_CONSUMED_OFFSET)
                    + Int64(source_index) * Int64(4),
                    epoch,
                )
            _ = store_release_sys_u32(
                route_arena.iterator.toint() + Int64(embedding_route.ROUTE_DONE_OFFSET), epoch
            )
            # The route replaces the head gradient ring's VOCAB_PANELS embedding epochs, the
            # last in each step.  Advance the ring's READY and CONSUMED words through them: the
            # next step's reduce_scatter_head_gradient waits for this step's last epoch in
            # CONSUMED.
            for panel_index in cutlass.range_constexpr(arenas.VOCAB_PANELS):
                position = Int32(arenas.VOCAB_PANELS + panel_index)
                virtual_epoch = (
                    (epoch - Int32(1))
                    * Int32(shell_fabric.VOCAB_MATRICES * arenas.VOCAB_PANELS)
                    + position
                    + Int32(1)
                )
                slot = virtual_epoch - (
                    virtual_epoch // Int32(arenas.HEAD_GRADIENT_SLOTS)
                ) * Int32(arenas.HEAD_GRADIENT_SLOTS)
                _ = store_release_sys_u32(
                    gradient_arena.iterator.toint()
                    + Int64(arenas.HEAD_GRADIENT_READY_OFFSET)
                    + Int64(slot) * Int64(4),
                    virtual_epoch,
                )
                _ = store_release_sys_u32(
                    gradient_arena.iterator.toint()
                    + Int64(arenas.HEAD_GRADIENT_CONSUMED_OFFSET)
                    + Int64(slot) * Int64(4),
                    virtual_epoch,
                )
        self.grid_phase_barrier(phase_counter)

    @cute.jit
    def run_final_rmsnorm(
        self,
        backward: cutlass.Constexpr[bool],
        role: cutlass.Constexpr[int],
        generation: Int32,
        phase_counter: cute.Tensor,
        row_table: cute.Tensor,
        shell_weight: cute.Tensor,
        shell_out: cute.Tensor,
        shell_rstd: cute.Tensor,
        shell_dnorm: cute.Tensor,
        shell_partial: cute.Tensor,
        fwd_tiler_mn,
        fwd_tiled_copy,
        fwd_threads_per_row: cutlass.Constexpr[int],
        bwd_tiler_mn,
        bwd_tiled_copy,
        bwd_threads_per_row: cutlass.Constexpr[int],
    ):
        """The final RMSNorm, forward or backward, on the output of layer ``generation`` (the top).

        Roles 1 and 2 each run one 128-thread shard on half of the page; role 0 only joins the
        closing grid barrier.  The backward writes dX into the top layer's incoming gradient
        (its ``staging_dy`` row entry) and the weight gradient's partial rows into
        ``shell_partial``.
        """

        self.reanchor_smem_page()
        memspace = row_table.iterator.memspace
        base = generation * Int32(decoder_layer.ROW_TABLE_WIDTH)
        matrix_h = cute.make_layout((model.SEQUENCE, model.HIDDEN), stride=(model.HIDDEN, 1))
        weight_h = cute.make_layout((1, model.HIDDEN), stride=(0, 1))
        fwd_rstd_h = cute.make_layout((model.SEQUENCE, model.HIDDEN), stride=(1, 0))
        bwd_rstd_h = cute.make_layout(model.SEQUENCE)
        partial_h = cute.make_layout(
            (2 * model.PROGRAM_CTAS, model.HIDDEN), stride=(model.HIDDEN, 1)
        )

        def row_view(name: cutlass.Constexpr[str], dtype, layout):
            return anchors.row_table_view(
                row_table,
                memspace,
                base + Int32(decoder_layer.ROW_TABLE_INDEX[name]),
                dtype,
                layout,
            )

        weight = cute.make_tensor(shell_weight.iterator, weight_h)
        out = cute.make_tensor(shell_out.iterator, matrix_h)
        if const_expr(backward):
            member = self.final_rms_bwd
            x = row_view("layer_output", BFloat16, matrix_h)
            dout = cute.make_tensor(shell_dnorm.iterator, matrix_h)
            rstd = cute.make_tensor(shell_rstd.iterator, bwd_rstd_h)
            dx = row_view("staging_dy", BFloat16, matrix_h)
            partial = cute.make_tensor(shell_partial.iterator, partial_h)
        else:
            member = self.final_rms_fwd
            x = row_view("layer_output", BFloat16, matrix_h)
            rstd = cute.make_tensor(shell_rstd.iterator, fwd_rstd_h)

        full_page = ProgramSmemAllocator.page
        if const_expr(role == 1):
            use_rms_shard_page(full_page, 0)
            if const_expr(backward):
                resident_rms_backward_body_h4096(
                    member,
                    x,
                    weight,
                    dout,
                    None,
                    rstd,
                    dx,
                    partial,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    bwd_tiler_mn,
                    bwd_tiled_copy,
                    bwd_threads_per_row,
                    shard=0,
                )
            else:
                resident_rms_forward_body_h4096(
                    member,
                    x,
                    weight,
                    None,
                    None,
                    out,
                    None,
                    rstd,
                    None,
                    Float32(model.RMS_EPSILON),
                    fwd_tiler_mn,
                    fwd_tiled_copy,
                    fwd_threads_per_row,
                    quack_rmsnorm_bodies.RMS_FORWARD_WAVES,
                    shard=0,
                )
            restore_full_page(full_page)
        elif const_expr(role == 2):
            use_rms_shard_page(full_page, 1)
            if const_expr(backward):
                resident_rms_backward_body_h4096(
                    member,
                    x,
                    weight,
                    dout,
                    None,
                    rstd,
                    dx,
                    partial,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    bwd_tiler_mn,
                    bwd_tiled_copy,
                    bwd_threads_per_row,
                    shard=1,
                )
            else:
                resident_rms_forward_body_h4096(
                    member,
                    x,
                    weight,
                    None,
                    None,
                    out,
                    None,
                    rstd,
                    None,
                    Float32(model.RMS_EPSILON),
                    fwd_tiler_mn,
                    fwd_tiled_copy,
                    fwd_threads_per_row,
                    quack_rmsnorm_bodies.RMS_FORWARD_WAVES,
                    shard=1,
                )
            restore_full_page(full_page)
        self.grid_phase_barrier(phase_counter)

    @cute.jit
    def reduce_final_norm_gradient(
        self,
        phase_counter: cute.Tensor,
        shell_partial: cute.Tensor,
        shell_grad: cute.Tensor,
    ):
        """Sum the final RMSNorm backward's partial rows, in row order, into its weight gradient."""

        qk_norm_rope.sum_partial_rows(
            cute.make_tensor(
                shell_partial.iterator,
                cute.make_layout(2 * model.PROGRAM_CTAS * model.HIDDEN),
            ),
            cute.make_tensor(shell_grad.iterator, cute.make_layout(model.HIDDEN)),
            2 * model.PROGRAM_CTAS,
            model.HIDDEN,
        )
        self.grid_phase_barrier(phase_counter)

    @cute.jit
    def reduce_qk_norm_gradients(
        self,
        generation: Int32,
        row_table: cute.Tensor,
    ):
        """Sum layer ``generation``'s q-norm and k-norm gradient partials into their gradients.

        The sums run in row order.  The kernel calls this from CTA 0 alone while the other CTAs
        run qkv dX; it has no grid barrier of its own, and the qkv dX phase's barrier follows.
        """

        self.reanchor_smem_page()
        memspace = row_table.iterator.memspace
        base = generation * Int32(decoder_layer.ROW_TABLE_WIDTH)
        flat_d = cute.make_layout(model.HEAD_DIM)
        flat_qpartial = cute.make_layout(
            model.QUERY_HEADS * model.NORM_BLOCKS * model.HEAD_DIM
        )
        flat_kpartial = cute.make_layout(
            model.KV_HEADS * model.NORM_BLOCKS * model.HEAD_DIM
        )
        q_partial = anchors.row_table_view(
            row_table,
            memspace,
            base + Int32(decoder_layer.ROW_TABLE_INDEX["q_norm_partial"]),
            Float32,
            flat_qpartial,
        )
        q_grad = anchors.row_table_view(
            row_table,
            memspace,
            base + Int32(decoder_layer.ROW_TABLE_INDEX["q_norm_grad"]),
            Float32,
            flat_d,
        )
        k_partial = anchors.row_table_view(
            row_table,
            memspace,
            base + Int32(decoder_layer.ROW_TABLE_INDEX["k_norm_partial"]),
            Float32,
            flat_kpartial,
        )
        k_grad = anchors.row_table_view(
            row_table,
            memspace,
            base + Int32(decoder_layer.ROW_TABLE_INDEX["k_norm_grad"]),
            Float32,
            flat_d,
        )
        qk_norm_rope.sum_q_norm_partials(q_partial, q_grad)
        qk_norm_rope.sum_k_norm_partials(k_partial, k_grad)

    @cute.jit
    def __call__(self, *args):
        """Prepare every member's kernel arguments and launch ``training_step_kernel``.

        ``args`` follows ``compile_training_program``'s argument order and ends with the
        stream.  The launch is cooperative, on the program grid.
        """

        nfam = len(self.families)
        (
            q, k, v, out, lse, softmax_scale, cu_seqlens, control,
            dout, lse_log2, dpsum, dq_accum, dk_accum, dv_accum,
        ) = args[0:14]
        # FA4 takes the control table as both seqused operands, which it never reads.
        forward_entry = (
            q, k, v, out, lse, softmax_scale, cu_seqlens, cu_seqlens, control, control,
        )
        backward_entry = (
            q, k, v, dout, lse_log2, dpsum, dq_accum, dk_accum, dv_accum,
            softmax_scale, cu_seqlens, cu_seqlens, control, control,
        )
        cursor = 14
        family_operands = args[cursor : cursor + 3 * nfam]
        cursor += 3 * nfam
        # Two activation-chain operands, indexed by the layer (generation_slot word 1): O
        # forward's residual input and down forward's second output.  The GEMMs' A, B and D
        # stay on the layer slots (word 0).
        o_fwd_residual_c = args[cursor]
        down_fwd_logical_output = args[cursor + 1]
        cursor += 2
        row_table = args[cursor]
        generation_slot = args[cursor + 1]
        generation_count = args[cursor + 2]
        phase_counter = args[cursor + 3]
        dq, dk, dv = args[cursor + 4 : cursor + 7]
        # FA4's forward binds `out` with dynamic extents; the backward phases read it through
        # q's static layout, which it shares.
        workspace_operands = (
            lse_log2, dpsum, dq_accum, dk_accum, dv_accum,
            cute.make_tensor(out.iterator, q.layout), dout, lse, dq, dk, dv,
        )
        shell = args[cursor + 7 : cursor + 32]
        trailing = args[cursor + 32 : -1]
        fabric_count = len(DECODER_FABRIC_ARGUMENT_NAMES)
        optimizer_count = len(OPTIMIZER_ARGUMENT_NAMES)
        full_shell_count = len(SHELL_FABRIC_AND_SCHEDULER_ARGUMENT_NAMES)
        fabric = trailing[:fabric_count]
        optimizer = trailing[fabric_count : fabric_count + optimizer_count]
        full_shell_begin = fabric_count + optimizer_count
        full_shell = trailing[full_shell_begin : full_shell_begin + full_shell_count]
        # The step loop's control words and per-step records.
        nstep = trailing[full_shell_begin + full_shell_count :]
        stream = args[-1]

        attention.record_attention_forward_args.__wrapped__(
            self, *forward_entry, stream
        )
        fargs = self.forward.recorded_args

        AttentionBackwardMember.__call__.__wrapped__(
            self.backward, *backward_entry, stream
        )

        gargs: list = []
        for index in cutlass.range_constexpr(nfam):
            member = self._members[index]
            a, b, d = family_operands[3 * index : 3 * index + 3]
            if const_expr(index == self.o_fwd_family_index):
                logical_generation_slot = cute.make_tensor(
                    generation_slot.iterator + 1,
                    cute.make_layout(1),
                )
                prepared = member.prepare_generation_kernel_arguments(
                    a,
                    b,
                    d,
                    o_fwd_residual_c,
                    generation_slot,
                    Int32(model.PROGRAM_CTAS),
                    logical_generation_slot,
                )
            elif const_expr(index == self.down_fwd_family_index):
                physical_residual_mid = family_operands[
                    3 * self.down_fwd_residual_source_family_index + 2
                ]
                logical_generation_slot = cute.make_tensor(
                    generation_slot.iterator + 1,
                    cute.make_layout(1),
                )
                prepared = member.prepare_generation_kernel_arguments(
                    a,
                    b,
                    d,
                    physical_residual_mid,
                    down_fwd_logical_output,
                    generation_slot,
                    Int32(model.PROGRAM_CTAS),
                    logical_generation_slot,
                )
            elif const_expr(index == self.down_dx_family_index):
                saved_preactivation = family_operands[
                    3 * self.gate_up_fwd_family_index + 2
                ]
                dgateup_physical = family_operands[
                    3 * self.gate_up_dx_family_index
                ]
                # The gate/up forward saves the preactivation as interleaved [gate, up] pairs,
                # and the down dX epilogue reads them at columns 2j and 2j + 1 itself: a TMA
                # tile load cannot express a stride-2 leading dimension, so it gets a flat
                # view rather than gate and up halves.  dgate and dup are stored as the two
                # halves of each gate_up_dy row.
                gate_up_flat = cute.make_tensor(
                    saved_preactivation.iterator,
                    cute.make_layout(
                        self.capacity * model.SEQUENCE * model.GATE_UP
                    ),
                )
                concat_output_half_layout = cute.make_layout(
                    (model.SEQUENCE, model.INTERMEDIATE, self.capacity),
                    stride=(
                        model.GATE_UP,
                        1,
                        model.SEQUENCE * model.GATE_UP,
                    ),
                )
                dgate_half = cute.make_tensor(
                    dgateup_physical.iterator,
                    concat_output_half_layout,
                )
                dup_half = cute.make_tensor(
                    dgateup_physical.iterator + model.INTERMEDIATE,
                    concat_output_half_layout,
                )
                prepared = member.prepare_generation_kernel_arguments(
                    a,
                    b,
                    d,
                    generation_slot,
                    Int32(model.PROGRAM_CTAS),
                    mGateUpFlat=gate_up_flat,
                    mDGateHalf=dgate_half,
                    mDUpHalf=dup_half,
                )
            elif const_expr(index == self.gate_up_fwd_family_index):
                mlp_intermediate = family_operands[
                    3 * self.down_fwd_family_index
                ]
                prepared = member.prepare_dual_output_arguments(
                    a,
                    b,
                    d,
                    mlp_intermediate,
                    generation_slot,
                    Int32(model.PROGRAM_CTAS),
                )
            elif const_expr(index == self.qkv_fwd_family_index):
                v_publish = cute.make_tensor(
                    v.iterator,
                    cute.make_layout(
                        (model.SEQUENCE, model.KV_HIDDEN, self.capacity),
                        stride=(
                            model.KV_HIDDEN,
                            1,
                            model.SEQUENCE * model.KV_HIDDEN,
                        ),
                    ),
                )
                prepared = member.prepare_value_publish_arguments(
                    a,
                    b,
                    d,
                    v_publish,
                    generation_slot,
                    Int32(model.PROGRAM_CTAS),
                )
            else:
                prepared = member.prepare_generation_kernel_arguments(
                    a, b, d, generation_slot, Int32(model.PROGRAM_CTAS)
                )
            gargs.extend(prepared)

        o_index = self.o_fwd_family_index
        o_a, o_b, o_d = family_operands[
            3 * o_index : 3 * o_index + 3
        ]
        logical_generation_slot = cute.make_tensor(
            generation_slot.iterator + 1,
            cute.make_layout(1),
        )
        checkpoint_generation_slot = cute.make_tensor(
            generation_slot.iterator + 2,
            cute.make_layout(1),
        )
        checkpoint_output = cute.make_tensor(
            shell[24].iterator,
            cute.make_layout(
                (
                    model.SEQUENCE,
                    model.HIDDEN,
                    RESIDUAL_MID_CHECKPOINT_LAYERS,
                ),
                stride=(
                    model.HIDDEN,
                    1,
                    model.SEQUENCE * model.HIDDEN,
                ),
            ),
        )
        o_fwd_checkpoint_args = (
            self.o_fwd_checkpoint_member.prepare_checkpoint_arguments(
                o_a,
                o_b,
                o_d,
                o_fwd_residual_c,
                checkpoint_output,
                generation_slot,
                Int32(model.PROGRAM_CTAS),
                logical_generation_slot,
                checkpoint_generation_slot,
            )
        )

        gate_index = self.gate_up_fwd_family_index
        down_index = self.down_fwd_family_index
        gate_a = family_operands[3 * gate_index]
        gate_b_physical = family_operands[3 * gate_index + 1]
        mlp_intermediate = family_operands[3 * down_index]
        initial_gate_up_aux_args = (
            self.initial_gate_up_aux_member.prepare_activation_only_arguments(
                gate_a,
                gate_b_physical,
                mlp_intermediate,
                generation_slot,
                Int32(model.PROGRAM_CTAS),
            )
        )

        # Quack's RMSNorm host setup: every member's cluster_n, which the RMS bodies read, and
        # the tiled copies of the layer norms (rms1's serve rms2) and the final norm.
        vecsize = math.gcd(model.HIDDEN, 128 // 16)
        self.rms1_fwd._set_cluster_n()
        self.rms1_fwd._cap_cluster_n(vecsize)
        fwd_copy, fwd_tiler, fwd_tpr = self.rms1_fwd._get_tiled_copy(vecsize=vecsize)
        self.rms2_fwd._set_cluster_n()
        self.rms2_fwd._cap_cluster_n(vecsize)
        self.rms1_bwd._set_cluster_n()
        self.rms1_bwd._cap_cluster_n(vecsize)
        bwd_copy, bwd_tiler, bwd_tpr = self.rms1_bwd._get_tiled_copy(vecsize=vecsize)
        self.rms2_bwd._set_cluster_n()
        self.rms2_bwd._cap_cluster_n(vecsize)
        self.final_rms_fwd._set_cluster_n()
        self.final_rms_fwd._cap_cluster_n(vecsize)
        sf_copy, sf_tiler, sf_tpr = self.final_rms_fwd._get_tiled_copy(vecsize=vecsize)
        self.final_rms_bwd._set_cluster_n()
        self.final_rms_bwd._cap_cluster_n(vecsize)
        sb_copy, sb_tiler, sb_tpr = self.final_rms_bwd._get_tiled_copy(vecsize=vecsize)

        head_fwd_args = self.head_phase.head_forward.prepare_generation_kernel_arguments(
            shell[3], shell[4], shell[5], shell[18], Int32(model.PROGRAM_CTAS)
        )
        head_dx_args = self.head_phase.head_dx.prepare_kernel_arguments(
            shell[6], shell[7], shell[8], Int32(model.PROGRAM_CTAS)
        )
        head_dw_args = self.head_phase.head_dw.prepare_kernel_arguments(
            shell[9], shell[10], shell[11], Int32(model.PROGRAM_CTAS)
        )

        self.training_step_kernel(
            *fargs,
            *self.backward.recorded_args,
            *gargs,
            row_table,
            fwd_tiler,
            fwd_copy,
            fwd_tpr,
            bwd_tiler,
            bwd_copy,
            bwd_tpr,
            sf_tiler,
            sf_copy,
            sf_tpr,
            sb_tiler,
            sb_copy,
            sb_tpr,
            generation_slot,
            control,
            generation_count,
            phase_counter,
            *workspace_operands,
            *head_fwd_args,
            *head_dx_args,
            *head_dw_args,
            shell[0],
            shell[1],
            shell[2],
            shell[6],
            shell[8],
            *shell[11:25],
            *fabric,
            *optimizer,
            *full_shell,
            *initial_gate_up_aux_args,
            *o_fwd_checkpoint_args,
            *nstep,
        ).launch(
            grid=[model.PROGRAM_CTAS, 1, 1],
            block=[model.PROGRAM_THREADS, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
            cooperative=True,
        )


_TENSOR = cute.Tensor
_OPTIONAL_TENSOR = Optional[cute.Tensor]
_CONSTEXPR_CALLABLE = cutlass.Constexpr[Callable]

# Annotations of training_step_kernel's FA4 parameters (None: unannotated), in the order the
# forward and backward entries record their native arguments.  The forward's are FA4's
# forward kernel parameters; the backward's are FA4's backward kernel parameters followed
# by the postprocess and preprocess operands.
FA4_FORWARD_KERNEL_ANNOTATIONS = (
    _TENSOR, _TENSOR, _TENSOR, _TENSOR,  # mQ mK mV mO
    _OPTIONAL_TENSOR,  # mLSE
    _OPTIONAL_TENSOR, _OPTIONAL_TENSOR, _OPTIONAL_TENSOR, _OPTIONAL_TENSOR,  # cu_seqlens, seqused
    _OPTIONAL_TENSOR,  # mPageTable
    *(Optional[cute.CopyAtom],) * 4,  # tma_atom_Q K V O
    Float32, Optional[Float32],  # softmax_scale_log2, softmax_scale
    Optional[Int32], Optional[Int32],  # window_size_left, window_size_right
    _OPTIONAL_TENSOR,  # learnable_sink
    Optional[BlockSparseTensors],
    *(cute.ComposedLayout,) * 4,  # sQ sK sV sO layouts
    cute.ComposedLayout | None,  # sP_layout
    *(cute.TiledCopy,) * 4,  # gmem_tiled_copy_Q K V O
    cute.TiledMma, cute.TiledMma,  # tiled_mma_qk, tiled_mma_pv
    None,  # tile_sched_params
    _CONSTEXPR_CALLABLE, _CONSTEXPR_CALLABLE,  # TileScheduler, SharedStorage
    AuxData,
    None,  # fastdiv_mods
)
FA4_BACKWARD_KERNEL_ANNOTATIONS = (
    *(_TENSOR,) * 6,  # mQ mK mV mdO mdK mdV
    *(cute.CopyAtom,) * 6,  # tma_atom_Q K V dO dK dV
    _TENSOR, _TENSOR, _TENSOR,  # mLSE mdPsum mdQaccum
    *(_OPTIONAL_TENSOR,) * 4,  # cu_seqlens Q K, seqused Q K
    *(cute.ComposedLayout,) * 5,  # sQ sK sV sPdS sdO layouts
    cute.Layout,  # sdQaccum_layout
    cute.TiledCopy,  # r2s_tiled_copy_dQaccum
    *(cute.TiledMma,) * 4,  # tiled_mma_SdP dK dV dQ
    None, None,  # softmax_scale_log2, softmax_scale
    ParamsBase,  # tile_sched_params
    _CONSTEXPR_CALLABLE, _CONSTEXPR_CALLABLE,  # TileScheduler, SharedStorage
    AuxData,
    None,  # fastdiv_mods
    Optional[BlockSparseTensors],
    Optional[cute.FastDivmodDivisor],  # qhead_per_kvhead_divmod
    *(_OPTIONAL_TENSOR,) * 3,  # dQ dK dV semaphores
    Optional[Int32], Optional[Int32],  # window_size_left, window_size_right
    # the dQ postprocess, then the dK/dV postprocess
    cute.TiledMma, cutlass.Constexpr, cute.Layout, cute.ComposedLayout,
    cute.TiledCopy, cute.TiledCopy, cute.TiledCopy,
    cute.TiledMma, cutlass.Constexpr, cute.Layout, cute.ComposedLayout,
    cute.TiledCopy, cute.TiledCopy, cute.TiledCopy,
    cute.TiledCopy, cute.TiledCopy,  # preprocess gmem copies of O and dQaccum
)


def _kernel_signature(families: tuple[decoder_layer.GemmFamily, ...]) -> str:
    """``training_step_kernel``'s decorator, name and parameters before the fabric operands.

    The generated source names the FA4 parameters' annotations through aliases that this
    function binds in the module's globals, where the source is executed.
    """

    declarations: list[str] = []
    for prefix, alias_prefix, annotations in (
        ("f", "_FA4_FORWARD_ANNOTATION", FA4_FORWARD_KERNEL_ANNOTATIONS),
        ("b", "_FA4_BACKWARD_ANNOTATION", FA4_BACKWARD_KERNEL_ANNOTATIONS),
    ):
        for index, annotation in enumerate(annotations):
            name = f"{prefix}_{index}"
            if annotation is None:
                declarations.append(name)
            else:
                alias = f"{alias_prefix}_{index}"
                globals()[alias] = annotation
                declarations.append(f"{name}: {alias}")

    for family in range(len(families)):
        declarations.extend(f"g{family}_{index}" for index in range(17))
    declarations.extend(
        (
            "row_table: cute.Tensor",
            "rms_fwd_tiler_mn: cute.Shape",
            "rms_fwd_tiled_copy: cute.TiledCopy",
            "rms_fwd_threads_per_row: cutlass.Constexpr[int]",
            "rms_bwd_tiler_mn: cute.Shape",
            "rms_bwd_tiled_copy: cute.TiledCopy",
            "rms_bwd_threads_per_row: cutlass.Constexpr[int]",
            "shell_fwd_tiler_mn: cute.Shape",
            "shell_fwd_tiled_copy: cute.TiledCopy",
            "shell_fwd_threads_per_row: cutlass.Constexpr[int]",
            "shell_bwd_tiler_mn: cute.Shape",
            "shell_bwd_tiled_copy: cute.TiledCopy",
            "shell_bwd_threads_per_row: cutlass.Constexpr[int]",
            "generation_slot: cute.Tensor",
            "generation_control: cute.Tensor",
            "generation_count: Int32",
            "phase_counter: cute.Tensor",
        )
    )
    ws_names = ATTENTION_BACKWARD_TENSOR_NAMES
    declarations.extend(f"{name}: cute.Tensor" for name in ws_names)
    for prefix in ("hf", "hx", "hw"):
        declarations.extend(f"{prefix}{index}" for index in range(17))
    declarations.extend(
        (
            "shell_weight: cute.Tensor",
            "shell_out: cute.Tensor",
            "shell_rstd: cute.Tensor",
            "shell_dlogits: cute.Tensor",
            "shell_head_dhidden: cute.Tensor",
            "shell_head_dweight: cute.Tensor",
            "shell_labels: cute.Tensor",
            "shell_global_valid_tokens: cute.Tensor",
            "shell_per_token_loss: cute.Tensor",
            "shell_loss: cute.Tensor",
            "shell_task_records: cute.Tensor",
            "shell_active_chunks: cute.Tensor",
            "shell_generation_slot: cute.Tensor",
            "shell_final_dnorm: cute.Tensor",
            "shell_final_partial: cute.Tensor",
            "shell_final_grad: cute.Tensor",
            "shell_fa4_checkpoint_extra0: cute.Tensor",
            "shell_fa4_checkpoint_extra1: cute.Tensor",
            "shell_residual_mid_checkpoint: cute.Tensor",
        )
    )

    return "".join(
        (
            "@cute.kernel\n",
            "def training_step_kernel(\n",
            "    self,\n",
            *(f"    {declaration},\n" for declaration in declarations),
        )
    )


# The step loop (resident_step.py) adds two operands to training_step_kernel, the runtime's
# `nstep` group: the int64 control words and the float32 per-step records.
STEP_LOOP_ARGUMENT_NAMES = ("nstep_control", "nstep_records")
# Work-sharing scheduler states the step loop resets before the next step, in order; it resets
# the head forward's per-chunk states separately.
PER_STEP_SCHEDULER_STATES = (
    "shell_head_dx_scheduler_state",
    "shell_head_dw_scheduler_state",
    "shell_gate_up_dx_scheduler_state",
    "shell_gate_up_dw_scheduler_state",
    "shell_gate_up_fwd_scheduler_state",
    "shell_down_dx_scheduler_state",
    "shell_down_dw_scheduler_state",
    "shell_down_fwd_scheduler_state",
    "shell_qkv_dx_scheduler_state",
    "shell_qkv_dw_scheduler_state",
    "shell_qkv_fwd_scheduler_state",
    "shell_o_dx_scheduler_state",
    "shell_o_dw_scheduler_state",
    "shell_o_fwd_scheduler_state",
)


def _embedding_route_slot_views(direction: str) -> list[str]:
    """Source lines viewing the four embedding-route tables at this step's token slot."""

    return [
        resident_step.slot_view(f"nstep_route_{name}_{direction}", source, extent, 4)
        for name, source, extent in (
            ("unique", "shell_embedding_route_unique_ids", "model.SEQUENCE"),
            ("inverse", "shell_embedding_route_inverse", "model.SEQUENCE"),
            ("offsets", "shell_embedding_route_owner_offsets", "model.WORLD"),
            ("counts", "shell_embedding_route_owner_counts", "model.WORLD"),
        )
    ]


def _generate_training_kernel(families: tuple[decoder_layer.GemmFamily, ...]):
    """Generate the source of ``training_step_kernel``, one training step, and its three regions.

    The kernel allocates the page, runs the step loop's prologue, the valid-token all-reduce and
    the embedding gather, then splits the warps into the three roles.  Each role calls
    ``decoder_forward``, gathers the head weight, and calls ``final_norm_head_and_loss`` and
    ``decoder_backward``; after the roles rejoin, the kernel reduce-scatters the head and
    embedding gradients, runs the optimizer and ends with the step loop's epilogue.

    The regions are ``@cute.jit`` functions, which the DSL inlines, so they are not call
    boundaries: they decide where in the kernel the values derived from their operands are
    computed.  Each region takes only the operands it uses, and each GEMM rebuilds its anchored
    scheduler parameters where it runs, so no prepared GEMM argument tuple or scheduler value
    stays live from one region into the next.

    Returns the kernel, the three region functions and the source.
    """

    # Attention checkpoints are on when the plan both borrows layers and uses the banks.
    fa4_checkpoint_top9 = bool(
        ATTENTION_CHECKPOINT_BORROWED_LAYERS > 0 and ATTENTION_CHECKPOINT_BANK_LAYERS > 0
    )
    # The residual adds run in the attention-output and down GEMMs' epilogues, and the SwiGLU
    # backward in down dX's, so those row phases are left out.  swiglu_fwd, which runs in the
    # gate/up GEMM's epilogue, is skipped where the phases are emitted.
    selected_forward_phases = tuple(
        phase
        for phase in decoder_layer.FORWARD_LAYER_PHASES
        if not (
            phase[0] == "o_fwd_residual"
            or phase[0] == "down_fwd_residual"
        )
    )
    assert len(selected_forward_phases) == len(decoder_layer.FORWARD_LAYER_PHASES) - 2
    selected_backward_phases = tuple(
        phase
        for phase in BACKWARD_PHASES
        if phase[0] != "swiglu_bwd"
    )
    assert len(selected_backward_phases) == len(BACKWARD_PHASES) - 1
    backward_names = tuple(name for name, _kind in selected_backward_phases)
    # The q/k-norm reductions run inside qkv dX (emit_layer_phase), which they directly precede.
    q_reduce = backward_names.index("q_norm_reduce")
    assert backward_names[q_reduce : q_reduce + 3] == (
        "q_norm_reduce",
        "k_norm_reduce",
        "qkv_dx",
    )
    initial_gate_up_aux_names = [
        f"initial_gate_up_aux_{index}" for index in range(17)
    ]
    o_fwd_checkpoint_names = [
        f"o_fwd_checkpoint_{index}" for index in range(17)
    ]
    route_prelude = (f"        route = gen % Int32({PHYSICAL_LAYER_SLOTS})\n",)

    kernel_header = _kernel_signature(families)
    kernel_header += "".join(
        f"    {name}: cute.Tensor,\n" for name in DECODER_FABRIC_ARGUMENT_NAMES
    )
    kernel_header += "".join(
        f"    {name}: cute.Tensor,\n" for name in OPTIMIZER_ARGUMENT_NAMES
    )
    kernel_header += "".join(
        f"    {name}: cute.Tensor,\n"
        for name in SHELL_FABRIC_AND_SCHEDULER_ARGUMENT_NAMES
    )
    kernel_header += "".join(
        f"    {name},\n" for name in initial_gate_up_aux_names
    )
    kernel_header += "".join(
        f"    {name},\n" for name in o_fwd_checkpoint_names
    )
    kernel_header += "".join(
        f"    {name}: cute.Tensor,\n" for name in STEP_LOOP_ARGUMENT_NAMES
    )
    kernel_header += "):\n"

    family_index = {family.name: index for index, family in enumerate(families)}
    forward_family_indices = tuple(
        family_index[name]
        for name, kind in decoder_layer.FORWARD_LAYER_PHASES
        if kind == "gemm"
    )
    backward_family_indices = tuple(
        family_index[name]
        for name, kind in selected_backward_phases
        if kind == "gemm"
    )
    assert forward_family_indices == (0, 1, 2, 3)
    assert backward_family_indices == tuple(range(4, len(families)))

    f_names = [f"f_{index}" for index in range(attention_members.FORWARD_RECORDED_ARG_COUNT)]
    b_names = [f"b_{index}" for index in range(attention_members.BACKWARD_RECORDED_ARG_COUNT)]
    g_names = {
        family: [f"g{family}_{index}" for index in range(17)]
        for family in range(len(families))
    }
    rms_names = [
        "rms_fwd_tiler_mn",
        "rms_fwd_tiled_copy",
        "rms_fwd_threads_per_row",
        "rms_bwd_tiler_mn",
        "rms_bwd_tiled_copy",
        "rms_bwd_threads_per_row",
    ]
    shell_rms_names = [
        "shell_fwd_tiler_mn",
        "shell_fwd_tiled_copy",
        "shell_fwd_threads_per_row",
        "shell_bwd_tiler_mn",
        "shell_bwd_tiled_copy",
        "shell_bwd_threads_per_row",
    ]
    ws_names = list(ATTENTION_BACKWARD_TENSOR_NAMES)
    hf_names = [f"hf{index}" for index in range(17)]
    hx_names = [f"hx{index}" for index in range(17)]
    hw_names = [f"hw{index}" for index in range(17)]
    shell_names = [
        "shell_weight",
        "shell_out",
        "shell_rstd",
        "shell_dlogits",
        "shell_head_dhidden",
        "shell_labels",
        "shell_global_valid_tokens",
        "shell_per_token_loss",
        "shell_loss",
        "shell_task_records",
        "shell_active_chunks",
        "shell_generation_slot",
        "shell_final_dnorm",
        "shell_final_partial",
        "shell_final_grad",
    ]
    fabric_names = list(DECODER_FABRIC_ARGUMENT_NAMES)
    optimizer_names = list(OPTIMIZER_ARGUMENT_NAMES)
    # clipped_adamw_step's operands: the optimizer group, with the final norm's gradient and
    # BF16 weight, which are shell operands, and the grid barrier's word in their places.
    optimizer_call_names = [
        *optimizer_names[:7],
        "shell_final_grad",
        *optimizer_names[7:9],
        "shell_weight",
        *optimizer_names[9:13],
        "phase_counter",
        *optimizer_names[13:],
    ]

    def jit_signature(name: str, args_: list[str]) -> list[str]:
        out = ["@cute.jit\n", f"def {name}(\n", "    self,\n"]
        out.extend(f"    {arg},\n" for arg in args_)
        out.append("):\n")
        return out

    def argument_lines(names: list[str], indent: str) -> list[str]:
        return [f"{indent}{name},\n" for name in names]

    def emit_layer_phase(
        out: list[str],
        name: str,
        kind: str,
        indent: str,
        *,
        backward: bool,
        initial_forward: bool = False,
    ) -> None:
        out.append(f"{indent}# {name}\n")
        if kind == "gemm":
            index = family_index[name]
            weight_forward_late_join = bool(
                not backward
                and initial_forward
                and name == "down_fwd"
            )
            # qkv dX: CTA 0 sums the q/k-norm gradients; CTAs 1-48 first prefetch the next lower
            # layer's weights, and CTAs 1-131 run the GEMM.  One grid barrier closes the phase.
            if (
                backward
                and name == "qkv_dx"
            ):
                state = "shell_qkv_dx_scheduler_state"
                out.extend(
                    (
                        f"{indent}if bidx == 0:\n",
                        f"{indent}    self.reduce_qk_norm_gradients(\n",
                        f"{indent}        gen, row_table,\n",
                        f"{indent}    )\n",
                        f"{indent}else:\n",
                    )
                )
                out.extend(
                    (
                        f"{indent}    if rev + Int32(1) < generation_count and bidx <= Int32(WEIGHT_PREFETCH_CTAS):\n",
                        f"{indent}        self.prefetch_layer_weights(\n",
                        f"{indent}            gen - Int32(1), service + Int32(1), 1,\n",
                        f"{indent}            fabric_weight_arena, fabric_weight_source_table,\n",
                        f"{indent}            fabric_control, fabric_step, fabric_status,\n",
                        f"{indent}        )\n",
                    )
                )
                out.extend(
                    (
                        f"{indent}    self.run_qkv_dx_gemm_without_cta0(\n",
                        f"{indent}        {index}, role, phase_counter, {state},\n",
                    )
                )
                out.extend(argument_lines(g_names[index], indent + "        "))
                out.append(f"{indent}    )\n")
                out.append(f"{indent}self.grid_phase_barrier(phase_counter)\n")
                out.extend(
                    (
                        f"{indent}if rev + Int32(1) < generation_count:\n",
                        f"{indent}    self.mark_prefetched_weights_ready(\n",
                        f"{indent}        gen - Int32(1), service + Int32(1),\n",
                        f"{indent}        fabric_weight_arena, fabric_step,\n",
                        f"{indent}    )\n",
                    )
                )
                return
            # Forward gate/up: only the top GATE_UP_RETAINED_TOP_LAYERS layers use the member
            # that also stores the preactivation.
            if (
                initial_forward
                and name == "gate_up_fwd"
            ):
                state = "shell_gate_up_fwd_scheduler_state"
                out.extend(
                    (
                        f"{indent}if gen + Int32(GATE_UP_RETAINED_TOP_LAYERS) >= generation_count:\n",
                        f"{indent}    self.run_decoder_gemm(\n",
                        f"{indent}        {index}, role, phase_counter, {state},\n",
                    )
                )
                out.extend(argument_lines(g_names[index], indent + "        "))
                out.extend(
                    (
                        f"{indent}    )\n",
                        f"{indent}else:\n",
                        f"{indent}    self.run_initial_gate_up_gemm(\n",
                        f"{indent}        {index}, role, phase_counter, {state},\n",
                    )
                )
                out.extend(
                    argument_lines(initial_gate_up_aux_names, indent + "        ")
                )
                out.append(f"{indent}    )\n")
                return
            state = f"shell_{name}_scheduler_state"
            runner = "run_decoder_gemm"
            # Forward down_fwd: CTAs 0-47 first prefetch the next layer's weights.
            if weight_forward_late_join:
                out.extend(
                    (
                        f"{indent}if gen + Int32(1) < generation_count and bidx < Int32(WEIGHT_PREFETCH_CTAS):\n",
                        f"{indent}    self.prefetch_layer_weights(\n",
                        f"{indent}        gen + Int32(1), gen + Int32(1), 0,\n",
                        f"{indent}        fabric_weight_arena, fabric_weight_source_table,\n",
                        f"{indent}        fabric_control, fabric_step, fabric_status,\n",
                        f"{indent}    )\n",
                    )
                )
            out.append(
                f"{indent}self.{runner}(\n"
                f"{indent}    {index}, role, phase_counter, {state},\n"
            )
            gemm_arg_names = g_names[index]
            out.extend(argument_lines(gemm_arg_names, indent + "    "))
            out.append(f"{indent})\n")
            if weight_forward_late_join:
                out.extend(
                    (
                        f"{indent}if gen + Int32(1) < generation_count:\n",
                        f"{indent}    self.mark_prefetched_weights_ready(\n",
                        f"{indent}        gen + Int32(1), gen + Int32(1),\n",
                        f"{indent}        fabric_weight_arena, fabric_step,\n",
                        f"{indent}    )\n",
                    )
                )
        elif kind == "fa4f":
            out.append(f"{indent}self.run_attention_forward(\n")
            out.append(f"{indent}    role, phase_counter,\n")
            out.extend(argument_lines(f_names, indent + "    "))
            out.append(f"{indent})\n")
        elif kind == "fa4b":
            out.append(f"{indent}self.run_attention_backward(\n")
            out.append(
                f"{indent}    role, route, phase_counter,\n"
            )
            out.extend(argument_lines(ws_names, indent + "    "))
            # The attention backward's postprocess also stores dV into dqkv_raw.  It gets the
            # base of both layer slots' dqkv_raw (row 0's entry), because its token offsets
            # already include the slot; this layer's own entry would add the slot offset twice.
            # The qkv dX GEMM's prepared A operand cannot serve: it is a TMA tensor whose
            # iterator is a coordinate tuple, not a global-memory pointer.
            out.append(
                f"{indent}    anchors.row_table_view(\n"
                f"{indent}        row_table, row_table.iterator.memspace,\n"
                f"{indent}        Int32(decoder_layer.ROW_TABLE_INDEX['dqkv_raw']),\n"
                f"{indent}        BFloat16, cute.make_layout(model.SEQUENCE * model.QKV_HIDDEN),\n"
                f"{indent}    ),\n"
            )
            out.extend(argument_lines(b_names, indent + "    "))
            out.append(f"{indent})\n")
            out.append(f"{indent}self.grid_phase_barrier(phase_counter)\n")
        elif (
            backward
            and name in ("q_norm_reduce", "k_norm_reduce")
        ):
            # CTA 0 runs both reductions during qkv dX, so these phases keep only their grid
            # barriers.
            out.append(f"{indent}self.grid_phase_barrier(phase_counter)\n")
        elif name == "v_extract":
            # The QKV forward's epilogue writes V (QkvForwardMember), so this phase keeps only
            # its grid barrier.
            out.append(f"{indent}self.grid_phase_barrier(phase_counter)\n")
        elif name == "dv_insert":
            # The attention backward's postprocess stores dV into dqkv_raw, so this phase keeps
            # only its grid barrier.
            out.append(f"{indent}self.grid_phase_barrier(phase_counter)\n")
        else:
            out.append(
                f"{indent}self.run_row_phase(\n"
                f"{indent}    {name!r}, role, gen, phase_counter, row_table,\n"
            )
            out.extend(argument_lines(rms_names, indent + "    "))
            out.append(f"{indent})\n")

    # decoder_forward: every layer's forward, bottom layer first.  It takes no backward GEMM or
    # head operand.
    forward_args = [
        "role: cutlass.Constexpr[int]",
        "generation_count",
        "phase_counter",
        "generation_control",
        "generation_slot",
        "row_table",
        *rms_names,
        *f_names,
        *[name for index in forward_family_indices for name in g_names[index]],
        "shell_gate_up_fwd_scheduler_state",
        "shell_down_fwd_scheduler_state",
        "shell_qkv_fwd_scheduler_state",
        "shell_o_fwd_scheduler_state",
        "shell_fa4_checkpoint_extra0",
        "shell_fa4_checkpoint_extra1",
        *fabric_names,
        *initial_gate_up_aux_names,
        *o_fwd_checkpoint_names,
    ]
    forward_lines = jit_signature("decoder_forward", forward_args)
    forward_lines.extend(
        (
            "    bidx, _, _ = cute.arch.block_idx()\n",
            "    tidx, _, _ = cute.arch.thread_idx()\n",
            "    for gen in cutlass.range(0, generation_count, 1, unroll=1):\n",
            *route_prelude,
        )
    )
    # Slot 0's last document is not the last cu_seqlens entry, so FA4's forward can load up to
    # one V tile (the forward member's FORWARD_TILE_N rows) past it, from the start of slot 1.
    # Those columns are masked, but 0 * V is NaN where V is not finite, so before layer 0,
    # while slot 1 still holds V from before this step, role 0 zeroes that tile.  Layer 1's
    # QKV forward then writes every row of slot 1's V before layer 1's attention uses it.  The
    # grid barrier after the route words below makes the zeroes visible.
    v_guard_rows = FORWARD_TILE_N
    v_row_elements = model.KV_HEADS * model.HEAD_DIM
    v_guard_element_count = v_guard_rows * v_row_elements
    v_guard_role_threads = model.PROGRAM_THREADS // 3
    if v_guard_rows <= 0 or v_guard_role_threads != 128:
        raise AssertionError("FA4 V-halo geometry drifted")
    forward_lines.extend(
        (
            "        if gen == Int32(0):\n",
            "            if const_expr(role == 0):\n",
            "                v_guard_grid_x, _, _ = cute.arch.grid_dim()\n",
            f"                v_guard_start = bidx * Int32({v_guard_role_threads}) + tidx\n",
            f"                v_guard_stride = v_guard_grid_x * Int32({v_guard_role_threads})\n",
            f"                for v_guard_route in cutlass.range_constexpr(1, {PHYSICAL_LAYER_SLOTS}):\n",
            "                    v_route_guard = anchors.row_table_view(\n",
            "                        row_table, row_table.iterator.memspace,\n",
            "                        v_guard_route * Int32(decoder_layer.ROW_TABLE_WIDTH)\n",
            "                        + Int32(decoder_layer.ROW_TABLE_INDEX['v']),\n",
            "                        BFloat16, cute.make_layout(\n",
            f"                            {v_guard_element_count}\n",
            "                        ),\n",
            "                    )\n",
            "                    for v_guard_element in cutlass.range(\n",
            f"                        v_guard_start, Int32({v_guard_element_count}),\n",
            "                        v_guard_stride, unroll=1,\n",
            "                    ):\n",
            "                        v_route_guard[v_guard_element] = (\n",
            "                            Float32(0.0).to(BFloat16)\n",
            "                        )\n",
        )
    )
    # The route words: the attention schedulers' control word and generation_slot's words 0 (the
    # layer slot, for the GEMMs' A, B and D), 1 (the layer, for the activation chain) and 2 (the
    # layer's residual-checkpoint record, top layer first).
    forward_lines.extend(
        (
            "        if const_expr(role == 0):\n",
            "            if bidx == 0:\n",
            "                if tidx == 0:\n",
            "                    generation_control[C_ROUTE] = route\n",
            "                    generation_slot[0] = route\n",
            "                    generation_slot[1] = gen\n",
            "                    generation_slot[2] = "
            "generation_count - Int32(1) - gen\n",
            "        self.grid_phase_barrier(phase_counter)\n",
        )
    )
    # The whole grid publishes layer 0's weights; each later layer's are prefetched during the
    # previous layer's down_fwd.
    forward_lines.extend(
        (
            "        if gen == Int32(0):\n",
            "            self.publish_layer_weights(\n",
            "                gen, gen, phase_counter,\n",
            "                fabric_weight_arena, fabric_weight_source_table,\n",
            "                fabric_control, fabric_step, fabric_status,\n",
            "            )\n",
            "        self.wait_layer_weights_ready(\n",
            "            gen, gen, phase_counter, fabric_control, fabric_step,\n",
            "            fabric_status,\n",
            "        )\n",
        )
    )
    for phase_name, phase_kind in selected_forward_phases:
        if phase_name == "swiglu_fwd":
            continue
        # The top RESIDUAL_MID_CHECKPOINT_LAYERS layers' attention-output GEMM also saves
        # residual_mid.
        if phase_name == "o_fwd":
            forward_lines.extend(
                (
                    f"        if gen + Int32({RESIDUAL_MID_CHECKPOINT_LAYERS}) >= generation_count:\n",
                    "            self.run_attention_output_checkpoint_gemm(\n",
                    "                role, phase_counter, shell_o_fwd_scheduler_state,\n",
                )
            )
            forward_lines.extend(argument_lines(
                o_fwd_checkpoint_names, "                "
            ))
            forward_lines.extend(("            )\n", "        else:\n"))
            emit_layer_phase(
                forward_lines,
                phase_name,
                phase_kind,
                "            ",
                backward=False,
                initial_forward=True,
            )
        else:
            emit_layer_phase(
                forward_lines,
                phase_name,
                phase_kind,
                "        ",
                backward=False,
                initial_forward=True,
            )
        # Save the checkpointed layers' attention output and LSE.
        if fa4_checkpoint_top9 and phase_name == "fa4_fwd_main":
            forward_lines.extend(
                (
                    "        if gen >= Int32(ATTENTION_CHECKPOINT_FIRST_LAYER):\n",
                    "            self.copy_attention_checkpoint(\n",
                    "                gen, route, f_3, f_4, row_table,\n",
                    "                fabric_grad_destination_table,\n",
                    "                shell_fa4_checkpoint_extra0,\n",
                    "                shell_fa4_checkpoint_extra1, False,\n",
                    "            )\n",
                )
            )
    forward_lines.append("        self.grid_phase_barrier(phase_counter)\n")
    forward_lines.extend(
        (
            "        self.release_layer_weights(\n",
            "            gen, gen, phase_counter, fabric_weight_arena,\n",
            "            fabric_step,\n",
            "        )\n",
        )
    )

    # final_norm_head_and_loss: the final RMSNorm, the LM head and loss (lm_head.HeadPhase),
    # then the final RMSNorm backward and its weight gradient.  The head GEMMs' argument tuples
    # are built and die inside it; it takes no decoder GEMM operand.
    head_args = [
        "role: cutlass.Constexpr[int]",
        "generation_count",
        "phase_counter",
        "row_table",
        "shell_scratch",
        *shell_rms_names,
        *hf_names,
        *hx_names,
        *hw_names,
        *shell_names,
        "shell_head_fwd_scheduler_state",
        "shell_head_dx_scheduler_state",
        "shell_head_dw_scheduler_state",
    ]
    head_lines = jit_signature("final_norm_head_and_loss", head_args)
    head_lines.append("    head_fwd_args = (\n")
    head_lines.extend(argument_lines(hf_names, "        "))
    head_lines.append("    )\n    head_dx_args = (\n")
    head_lines.extend(argument_lines(hx_names, "        "))
    head_lines.append("    )\n    head_dw_args = (\n")
    head_lines.extend(argument_lines(hw_names, "        "))
    head_lines.extend(
        (
            "    )\n",
            "    shell_dlogits_flat = lm_head.aligned_flat_view(\n",
            "        shell_dlogits, model.SEQUENCE * model.VOCAB\n",
            "    )\n",
            "    shell_head_dhidden_flat = cute.make_tensor(\n",
            "        shell_head_dhidden.iterator, cute.make_layout(model.SEQUENCE * model.HIDDEN)\n",
            "    )\n",
            "    shell_final_dnorm_flat = cute.make_tensor(\n",
            "        shell_final_dnorm.iterator, cute.make_layout(model.SEQUENCE * model.HIDDEN)\n",
            "    )\n",
            "    last_gen = generation_count - Int32(1)\n",
            "    self.run_final_rmsnorm(\n",
            "        False, role, last_gen, phase_counter, row_table,\n",
            "        shell_weight, shell_out, shell_rstd, shell_final_dnorm,\n",
            "        shell_final_partial,\n",
        )
    )
    head_lines.extend(argument_lines(shell_rms_names, "        "))
    head_lines.extend(
        (
            "    )\n",
            "    self.head_phase.run(\n",
            "        head_fwd_args, head_dx_args, head_dw_args,\n",
            "        shell_dlogits_flat, shell_labels,\n",
            "        shell_global_valid_tokens, shell_per_token_loss,\n",
            "        shell_loss, shell_task_records, shell_active_chunks,\n",
            "        shell_generation_slot, shell_scratch,\n",
            "        shell_head_fwd_scheduler_state,\n",
            "        shell_head_dx_scheduler_state,\n",
            "        shell_head_dw_scheduler_state,\n",
            "        phase_counter, role,\n",
            "    )\n",
            "    self.grid_phase_barrier(phase_counter)\n",
            "    fp32_to_bf16(shell_head_dhidden_flat, shell_final_dnorm_flat)\n",
            "    self.grid_phase_barrier(phase_counter)\n",
            "    self.run_final_rmsnorm(\n",
            "        True, role, last_gen, phase_counter, row_table,\n",
            "        shell_weight, shell_out, shell_rstd, shell_final_dnorm,\n",
            "        shell_final_partial,\n",
        )
    )
    head_lines.extend(argument_lines(shell_rms_names, "        "))
    head_lines.extend(
        (
            "    )\n",
            "    self.reduce_final_norm_gradient(\n",
            "        phase_counter, shell_final_partial, shell_final_grad\n",
            "    )\n",
        )
    )

    # decoder_backward: every layer, top layer first.  A reverse visit recomputes the layer's
    # forward, less what the checkpoints and retained outputs supply, then runs its backward and
    # reduce-scatters the previous visit's gradients.  It takes the forward operands for the
    # recompute but no head operand.
    backward_args = [
        "role: cutlass.Constexpr[int]",
        "generation_count",
        "phase_counter",
        "generation_control",
        "generation_slot",
        "row_table",
        *rms_names,
        *b_names,
        *ws_names,
        *f_names,
        *[name for index in forward_family_indices for name in g_names[index]],
        *[name for index in backward_family_indices for name in g_names[index]],
        "shell_gate_up_dx_scheduler_state",
        "shell_gate_up_dw_scheduler_state",
        "shell_gate_up_fwd_scheduler_state",
        "shell_down_dx_scheduler_state",
        "shell_down_dw_scheduler_state",
        "shell_qkv_dx_scheduler_state",
        "shell_qkv_dw_scheduler_state",
        "shell_qkv_fwd_scheduler_state",
        "shell_o_dx_scheduler_state",
        "shell_o_dw_scheduler_state",
        "shell_o_fwd_scheduler_state",
        "shell_fa4_checkpoint_extra0",
        "shell_fa4_checkpoint_extra1",
        "shell_residual_mid_checkpoint",
        *fabric_names,
    ]
    backward_lines = jit_signature("decoder_backward", backward_args)
    backward_lines.extend(
        (
            "    bidx, _, _ = cute.arch.block_idx()\n",
            "    tidx, _, _ = cute.arch.thread_idx()\n",
            "    for rev in cutlass.range(0, generation_count, 1, unroll=1):\n",
            "        gen = generation_count - Int32(1) - rev\n",
            *route_prelude,
            "        if const_expr(role == 0):\n",
            "            if bidx == 0:\n",
            "                if tidx == 0:\n",
            "                    generation_control[C_ROUTE] = route\n",
            "                    generation_slot[0] = route\n",
            "                    generation_slot[1] = gen\n",
        )
    )
    # Point the layer's residual_mid row entry at its checkpoint record on the first
    # RESIDUAL_MID_CHECKPOINT_LAYERS reverse visits and at its slot otherwise.
    backward_lines.extend(
        (
            f"        if rev < Int32({RESIDUAL_MID_CHECKPOINT_LAYERS}):\n",
            "            self.set_residual_mid_row(\n",
            "                gen, route, rev, row_table,\n",
            "                shell_residual_mid_checkpoint, True,\n",
            "            )\n",
            "        else:\n",
            "            self.set_residual_mid_row(\n",
            "                gen, route, rev, row_table,\n",
            "                shell_residual_mid_checkpoint, False,\n",
            "            )\n",
        )
    )
    backward_lines.append("        self.grid_phase_barrier(phase_counter)\n")
    # The first reverse visit publishes the top layer's weights again under its backward service
    # number; each later visit's weights are prefetched during the previous visit's qkv dX.
    backward_lines.extend(
        (
            "        service = Int32(model.DEPTH) + rev\n",
            "        if rev == Int32(0):\n",
            "            self.publish_layer_weights(\n",
            "                gen, service, phase_counter,\n",
            "                fabric_weight_arena, fabric_weight_source_table,\n",
            "                fabric_control, fabric_step, fabric_status,\n",
            "            )\n",
            "        self.wait_layer_weights_ready(\n",
            "            gen, service, phase_counter, fabric_control, fabric_step,\n",
            "            fabric_status,\n",
            "        )\n",
        )
    )
    # The recompute leaves out down_fwd: the forward keeps its output, the layer output, in the
    # activation chain, and no backward phase reads a recomputed copy.
    dead_recompute_terminal_phases = ("down_fwd",)
    selected_recompute_phases = tuple(
        phase
        for phase in selected_forward_phases
        if phase[0] not in dead_recompute_terminal_phases
    )
    for phase_name, phase_kind in selected_recompute_phases:
        if phase_name == "swiglu_fwd":
            continue
        if phase_name == "gate_up_fwd":
            backward_lines.extend(
                (
                    f"        if rev < Int32({GATE_UP_RETAINED_TOP_LAYERS}):\n",
                    # The forward kept these layers' gate/up outputs; keep the GEMM's grid barrier.
                    "            self.grid_phase_barrier(phase_counter)\n",
                    "        else:\n",
                )
            )
            emit_layer_phase(
                backward_lines,
                phase_name,
                phase_kind,
                "            ",
                backward=False,
            )
        elif phase_name == "o_fwd":
            backward_lines.extend(
                (
                    f"        if rev < Int32({RESIDUAL_MID_CHECKPOINT_LAYERS}):\n",
                    # residual_mid comes from the checkpoint; keep the GEMM's grid barrier.
                    "            self.grid_phase_barrier(phase_counter)\n",
                    "        else:\n",
                )
            )
            emit_layer_phase(
                backward_lines,
                phase_name,
                phase_kind,
                "            ",
                backward=False,
            )
        elif fa4_checkpoint_top9 and phase_name == "fa4_fwd_main":
            # Checkpointed layers restore their attention output and LSE instead.
            backward_lines.extend(
                (
                    "        if gen >= Int32(ATTENTION_CHECKPOINT_FIRST_LAYER):\n",
                    "            self.copy_attention_checkpoint(\n",
                    "                gen, route, f_3, f_4, row_table,\n",
                    "                fabric_grad_destination_table,\n",
                    "                shell_fa4_checkpoint_extra0,\n",
                    "                shell_fa4_checkpoint_extra1, True,\n",
                    "            )\n",
                    "            self.grid_phase_barrier(phase_counter)\n",
                    "        else:\n",
                )
            )
            emit_layer_phase(
                backward_lines,
                phase_name,
                phase_kind,
                "            ",
                backward=False,
            )
        else:
            emit_layer_phase(
                backward_lines,
                phase_name,
                phase_kind,
                "        ",
                backward=False,
            )
    backward_lines.append("        self.grid_phase_barrier(phase_counter)\n")
    for phase_name, phase_kind in selected_backward_phases:
        emit_layer_phase(
            backward_lines,
            phase_name,
            phase_kind,
            "        ",
            backward=True,
        )
    backward_lines.append("        self.grid_phase_barrier(phase_counter)\n")
    # Release the layer's weights, reduce-scatter the gradients of the layer above (published by
    # the previous visit), then publish this layer's.
    backward_lines.extend(
        (
            "        self.release_layer_weights(\n",
            "            gen, service, phase_counter, fabric_weight_arena,\n",
            "            fabric_step,\n",
            "        )\n",
            "        if rev > Int32(0):\n",
            "            pending_rev = rev - Int32(1)\n",
            "            pending_gen = gen + Int32(1)\n",
            "            self.reduce_scatter_layer_gradients(\n",
            "                pending_gen, pending_rev, phase_counter,\n",
            "                fabric_grad_arena, fabric_grad_destination_table,\n",
            "                fabric_control, fabric_step, fabric_status,\n",
            "            )\n",
            "        self.publish_layer_gradients(\n",
            "            gen, rev, phase_counter, fabric_grad_arena, fabric_step,\n",
            "        )\n",
        )
    )
    # Layer 0's gradients, which the last reverse visit published.
    backward_lines.extend(
        (
            "    if generation_count > Int32(0):\n",
            "        final_rev = generation_count - Int32(1)\n",
            "        self.reduce_scatter_layer_gradients(\n",
            "            Int32(0), final_rev, phase_counter, fabric_grad_arena,\n",
            "            fabric_grad_destination_table, fabric_control,\n",
            "            fabric_step, fabric_status,\n",
            "        )\n",
        )
    )

    # The kernel body: the page and a 16-float shared scratch, the step loop's prologue, the
    # decoder fabric's step, the valid-token all-reduce and the embedding gather, then the role
    # dispatch into the three regions; after it, the head and embedding gradient
    # reduce-scatters, the optimizer and the step loop's epilogue.
    kernel_lines = [
        kernel_header,
        "    allocator = cutlass.utils.SmemAllocator()\n",
        "    ProgramSmemAllocator.page = allocator.allocate(\n",
        "        PROGRAM_SMEM_PAGE_BYTES, byte_alignment=1024\n",
        "    )\n",
        "    scratch_allocator = ProgramSmemAllocator()\n",
        "    shell_scratch = scratch_allocator.allocate_tensor(\n",
        "        Float32, cute.make_layout(16), byte_alignment=16\n",
        "    )\n",
    ]
    kernel_lines.extend(resident_step.step_prologue())
    kernel_lines.extend(
        (
            "    self.advance_decoder_fabric_step(phase_counter, fabric_step)\n",
        )
    )
    kernel_lines.append(
        resident_step.slot_view("nstep_local_valid", "shell_local_valid_tokens", "1", 4)
    )
    kernel_lines.extend(
        (
            "    self.all_reduce_valid_tokens(\n",
            "        phase_counter, shell_valid_arena, shell_fabric_control,\n",
            "        nstep_local_valid, shell_global_valid_tokens,\n",
            "        shell_fabric_status,\n",
            "        optimizer_integer_control,\n",
            "    )\n",
        )
    )
    kernel_lines.append(
        resident_step.slot_view("nstep_input_ids_forward_route", "shell_input_ids", "model.SEQUENCE", 4)
    )
    kernel_lines.extend(_embedding_route_slot_views("forward"))
    kernel_lines.extend(
        (
            "    self.gather_embedding_rows(\n",
            "        phase_counter, row_table, shell_weight_arena,\n",
            "        shell_fabric_control, nstep_input_ids_forward_route,\n",
            "        optimizer_bf16_parameter,\n",
            "        optimizer_integer_control, shell_embedding_route_arena,\n",
            "        shell_embedding_route_peer_bases,\n",
            "        nstep_route_unique_forward,\n",
            "        nstep_route_inverse_forward,\n",
            "        nstep_route_offsets_forward,\n",
            "        nstep_route_counts_forward,\n",
            "        shell_embedding_route_status,\n",
            "    )\n",
        )
    )
    kernel_lines.append(
        "    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())\n"
    )

    forward_call_args = [
        "role",
        "generation_count",
        "phase_counter",
        "generation_control",
        "generation_slot",
        "row_table",
        *rms_names,
        *f_names,
        *[name for index in forward_family_indices for name in g_names[index]],
        "shell_gate_up_fwd_scheduler_state",
        "shell_down_fwd_scheduler_state",
        "shell_qkv_fwd_scheduler_state",
        "shell_o_fwd_scheduler_state",
        "shell_fa4_checkpoint_extra0",
        "shell_fa4_checkpoint_extra1",
        *fabric_names,
        *initial_gate_up_aux_names,
        *o_fwd_checkpoint_names,
    ]
    head_call_args = [
        "role",
        "generation_count",
        "phase_counter",
        "row_table",
        "shell_scratch",
        *shell_rms_names,
        *hf_names,
        *hx_names,
        *hw_names,
        *shell_names,
        "shell_head_fwd_scheduler_state",
        "shell_head_dx_scheduler_state",
        "shell_head_dw_scheduler_state",
    ]
    backward_call_args = [
        "role",
        "generation_count",
        "phase_counter",
        "generation_control",
        "generation_slot",
        "row_table",
        *rms_names,
        *b_names,
        *ws_names,
        *f_names,
        *[name for index in forward_family_indices for name in g_names[index]],
        *[name for index in backward_family_indices for name in g_names[index]],
        "shell_gate_up_dx_scheduler_state",
        "shell_gate_up_dw_scheduler_state",
        "shell_gate_up_fwd_scheduler_state",
        "shell_down_dx_scheduler_state",
        "shell_down_dw_scheduler_state",
        "shell_qkv_dx_scheduler_state",
        "shell_qkv_dw_scheduler_state",
        "shell_qkv_fwd_scheduler_state",
        "shell_o_dx_scheduler_state",
        "shell_o_dw_scheduler_state",
        "shell_o_fwd_scheduler_state",
        "shell_fa4_checkpoint_extra0",
        "shell_fa4_checkpoint_extra1",
        "shell_residual_mid_checkpoint",
        *fabric_names,
    ]

    # The role dispatch sets the register split: warps 0-3 are role 0 with 56 registers, warps
    # 4-7 and 8-11 roles 1 and 2 with 224.  Role 0 gives up registers before named barrier 13
    # and roles 1 and 2 claim theirs after it, so the claim finds them free; the head returns
    # to this split on the same barrier (lm_head.DECODER_REGISTER_SPLIT_BARRIER).  Each role
    # calls the regions with its role as a constant.
    for role in (0, 1, 2):
        kernel_lines.append(
            "    if warp_idx < 4:\n"
            if role == 0
            else ("    elif warp_idx < 8:\n" if role == 1 else "    else:\n")
        )
        kernel_lines.append(f"        role = {role}\n")
        if role == 0:
            kernel_lines.extend(
                (
                    "        cute.arch.setmaxregister_decrease(56)\n",
                    "        cute.arch.barrier(barrier_id=13, number_of_threads=model.PROGRAM_THREADS)\n",
                )
            )
        else:
            kernel_lines.extend(
                (
                    "        cute.arch.barrier(barrier_id=13, number_of_threads=model.PROGRAM_THREADS)\n",
                    "        cute.arch.setmaxregister_increase(224)\n",
                )
            )
        kernel_lines.append("        self.decoder_forward(\n")
        kernel_lines.extend(argument_lines(forward_call_args, "            "))
        kernel_lines.append("        )\n")
        kernel_lines.append(
            resident_step.slot_view(
                f"nstep_input_ids_role{role}", "shell_input_ids", "model.SEQUENCE", 8
            )
        )
        kernel_lines.extend(
            (
                "        self.all_gather_head_weight(\n",
                "            1, phase_counter, row_table, shell_weight_arena,\n",
                f"            shell_fabric_control, nstep_input_ids_role{role},\n",
                "            shell_head_weight_raw, optimizer_bf16_parameter,\n",
                "            shell_fabric_status,\n",
                "            optimizer_integer_control,\n",
                "        )\n",
            )
        )
        # The head reads this step's labels from the token slot.
        kernel_lines.append(
            resident_step.slot_view(f"nstep_labels_role{role}", "shell_labels", "model.SEQUENCE", 8)
        )
        role_head_call_args = [
            f"nstep_labels_role{role}" if name == "shell_labels" else name
            for name in head_call_args
        ]
        kernel_lines.append("        self.final_norm_head_and_loss(\n")
        kernel_lines.extend(argument_lines(role_head_call_args, "            "))
        kernel_lines.append("        )\n")
        kernel_lines.append("        self.decoder_backward(\n")
        kernel_lines.extend(argument_lines(backward_call_args, "            "))
        kernel_lines.append("        )\n")

    kernel_lines.extend(
        (
            "    self.reduce_scatter_head_gradient(\n",
            "        phase_counter, shell_gradient_arena, shell_fabric_control,\n",
            "        optimizer_gradient, shell_fabric_status,\n",
            "        optimizer_integer_control,\n",
            "    )\n",
        )
    )
    kernel_lines.extend(_embedding_route_slot_views("backward"))
    kernel_lines.extend(
        (
            "    self.reduce_scatter_embedding_gradient(\n",
            "        phase_counter, row_table, shell_head_dweight,\n",
            "        optimizer_gradient,\n",
            "        shell_gradient_arena,\n",
            "        shell_fabric_control, optimizer_integer_control,\n",
            "        shell_embedding_route_arena,\n",
            "        shell_embedding_route_peer_bases,\n",
            "        nstep_route_unique_backward,\n",
            "        nstep_route_inverse_backward,\n",
            "        nstep_route_offsets_backward,\n",
            "        nstep_route_counts_backward,\n",
            "        shell_embedding_route_status,\n",
            "        shell_scratch,\n",
            "    )\n",
        )
    )

    kernel_lines.extend(resident_step.pre_optimizer())
    kernel_lines.append("    self.run_optimizer_step(\n")
    kernel_lines.extend(argument_lines(optimizer_call_names, "        "))
    kernel_lines.append("        model.PROGRAM_THREADS,\n")
    kernel_lines.append("    )\n")
    kernel_lines.extend(resident_step.step_epilogue(PER_STEP_SCHEDULER_STATES))

    source = "".join((*forward_lines, *head_lines, *backward_lines, *kernel_lines))

    filename = f"{__file__}:generated_training_kernel"
    linecache.cache[filename] = (
        len(source), None, source.splitlines(keepends=True), filename
    )
    namespace: dict[str, object] = {}
    exec(compile(source, filename, "exec"), globals(), namespace)  # noqa: S102
    return (
        namespace["training_step_kernel"],
        namespace["decoder_forward"],
        namespace["final_norm_head_and_loss"],
        namespace["decoder_backward"],
        source,
    )


# The attention control word holding the current layer slot, which the generated kernel
# writes and the attention schedulers read.
C_ROUTE = attention.CONTROL_ROUTE
@functools.cache
def training_kernel(families: tuple[decoder_layer.GemmFamily, ...]) -> tuple[type, str]:
    """The program class with the generated kernel and regions bound, and the generated source.

    Cached, so each process generates the source once per family tuple.
    """

    kernel, forward_region, head_region, backward_region, source = _generate_training_kernel(families)
    # The class name becomes part of the kernel's symbol name.
    klass = type(
        "ResidentTrainingKernel",
        (TrainingProgram,),
        {
            "training_step_kernel": kernel,
            "decoder_forward": forward_region,
            "final_norm_head_and_loss": head_region,
            "decoder_backward": backward_region,
        },
    )
    return klass, source


def _dynamic_tensor(tensor: torch.Tensor):
    """A fully dynamic CuTe view of ``tensor``, assumed 16-byte aligned when its address is."""

    return to_cute_tensor(
        tensor,
        assumed_align=16 if tensor.data_ptr() % 16 == 0 else 4,
        fully_dynamic=True,
    )


def _shell_operand_pairs(shell: ShellTensors) -> list[tuple[object, torch.Tensor]]:
    """The launch ABI's `shell` group, as (compile-time CuTe view, runtime tensor) pairs.

    In order: the final RMSNorm's weight, output and rstd; the head GEMMs' operands (forward
    per head chunk, dX, dW); the loss and chunk-route tensors; the final RMSNorm backward's
    tensors; and the checkpoint stores.
    """

    hidden_chunks = shell.final_normalized_hidden.view(
        model.HEAD_CHUNKS, model.HEAD_CHUNK_ROWS, model.HIDDEN
    ).permute(1, 2, 0)
    weight_broadcast = shell.head_weight.unsqueeze(-1).expand(
        model.VOCAB, model.HIDDEN, model.HEAD_CHUNKS
    )
    # The chunk axis strides through dlogits_slab, so the head forward writes each chunk's
    # logits into that chunk's rows; the loss then overwrites them in place with dlogits, which
    # head dX and dW read.
    logits_broadcast = shell.dlogits_slab.view(
        model.HEAD_CHUNKS, model.HEAD_CHUNK_ROWS, model.VOCAB
    ).permute(1, 2, 0)
    raw = (
        shell.final_norm_weight,
        shell.final_normalized_hidden,
        shell.final_rstd,
        hidden_chunks,
        weight_broadcast,
        logits_broadcast,
        shell.dlogits_slab.unsqueeze(-1),
        shell.head_weight.mT.unsqueeze(-1),
        shell.head_dhidden_fp32.unsqueeze(-1),
        shell.dlogits_slab.mT.unsqueeze(-1),
        shell.final_normalized_hidden.mT.unsqueeze(-1),
        shell.head_dweight.unsqueeze(-1),
        shell.labels,
        shell.global_valid_tokens,
        shell.per_token_loss,
        shell.loss,
        shell.task_records,
        shell.active_chunks,
        shell.generation_slot,
        shell.final_dnorm_bf16,
        shell.final_norm_partial,
        shell.final_norm_grad,
        shell.fa4_checkpoint_extra0,
        shell.fa4_checkpoint_extra1,
        shell.residual_mid_checkpoint,
    )
    return [(_dynamic_tensor(tensor), tensor) for tensor in raw]


def allocate_step_loop_operands(
    device: torch.device, steps: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    """The step loop's control words and per-step records, for compiling the program.

    The compile does not read their values, and the runtime passes its own tensors;
    ``steps`` sizes the records the launch ABI describes.
    """

    control = [0] * RP.CONTROL_WORDS
    control[RP.CONTROL_STEPS] = steps
    control[RP.CONTROL_RING_MASK] = 3
    control[RP.CONTROL_RECORD_WIDTH] = RP.RECORD_WIDTH
    control[RP.CONTROL_REFILL_SLOT_MASK] = 1
    control[RP.CONTROL_REFILL_GENERATION] = 1
    control[RP.CONTROL_REFILL_TIMEOUT_NS] = 60_000_000_000
    return (
        torch.tensor(control, dtype=torch.int64, device=device),
        torch.zeros(steps * RP.RECORD_WIDTH, dtype=torch.float32, device=device),
    )


@dataclass(frozen=True)
class CompiledTrainingProgram:
    """The compiled function and the runtime argument groups it was compiled with.

    kernel/build.py writes the groups into the launch ABI.
    """

    function: object
    runtime_prefix: tuple
    runtime_suffix: tuple
    runtime_shell: tuple
    runtime_fabric: tuple
    runtime_optimizer: tuple
    runtime_full_shell: tuple
    runtime_nstep: tuple

def runtime_prefix_suffix(tensors: object) -> tuple[tuple, tuple]:
    """The runtime's `prefix` and `suffix` groups, in ``compile_training_program``'s order.

    The prefix holds the attention operands, three GEMM operands per family (A as (M, K, G),
    B as (N, K, G) and D as (M, N, G), with down_dw's as the transposed views of B, A and D),
    the two activation-chain views, the row table and the generation slot; the suffix holds
    the grid barrier's word and the attention gradients.
    """

    t = tensors.base
    family_operands: list = []
    for family in tensors.families:
        if family.name == "down_dw":
            family_operands.extend(
                tensors.slabs[name].permute(2, 1, 0)
                for name in (family.b, family.a, family.d)
            )
        else:
            a = tensors.slabs[family.a]
            family_operands.extend(
                (
                    a.permute(2, 1, 0) if family.a_t else a.permute(1, 2, 0),
                    tensors.slabs[family.b].permute(2, 1, 0),
                    tensors.slabs[family.d].permute(1, 2, 0),
                )
            )
    prefix = (
        t.q, t.k, t.v, t.out, t.lse, model.HEAD_DIM**-0.5, t.cu_seqlens, t.control,
        t.dout, t.lse_log2, t.dpsum, t.dq_accum, t.dk_accum, t.dv_accum,
        *family_operands,
        tensors.activation_chain[:-1].permute(1, 2, 0),
        tensors.activation_chain[1:].permute(1, 2, 0),
        tensors.row_table,
        t.generation_slot,
    )
    return prefix, (t.phase_counter, t.dq, t.dk, t.dv)


def compile_training_program(
    tensors: TrainingTensors,
    *,
    decoder_fabric: object,
    optimizer_tail: object,
    full_shell: object,
    nstep: tuple[torch.Tensor, torch.Tensor],
) -> CompiledTrainingProgram:
    """Compile the program for ``tensors`` and return the function and its argument groups.

    ``tensors`` must have two layer slots and model.DEPTH layers; the launch passes the depth
    as the generation count.  ``decoder_fabric``, ``optimizer_tail`` and ``full_shell``
    supply the `fabric`, `optimizer` and `full_shell` groups, and ``nstep`` the step loop's
    two tensors.  After compiling, every GEMM member's shared storage is checked against the
    page.
    """

    attention.check_generation_token_ranges(tensors.base.plan)
    if len(nstep) != len(STEP_LOOP_ARGUMENT_NAMES):
        raise ValueError("the resident program needs its nstep control and record tensors")
    t = tensors.base
    families = tensors.families
    capacity = tensors.capacity
    logical_depth = tensors.logical_depth
    if capacity != PHYSICAL_LAYER_SLOTS:
        raise ValueError("recompute slot count must equal physical tensor capacity")
    if logical_depth != model.DEPTH:
        raise ValueError(
            f"decoder fabric requires logical depth {model.DEPTH}, got {logical_depth}"
        )
    softmax_scale = model.HEAD_DIM**-0.5

    cu = attention.dynamic_cute_tensor(t.cu_seqlens)
    control = attention.dynamic_cute_tensor(t.control)
    counter = attention.static_cute_tensor(t.phase_counter, 4)
    slot = attention.static_cute_tensor(t.generation_slot, 16)
    attention_args = (
        attention.static_cute_tensor(t.q),
        attention.static_cute_tensor(t.k),
        attention.static_cute_tensor(t.v),
        attention.ragged_cute_tensor(t.out),
        attention.static_cute_tensor(t.lse),
        softmax_scale,
        cu,
        control,
        attention.static_cute_tensor(t.dout),
        attention.static_cute_tensor(t.lse_log2),
        attention.static_cute_tensor(t.dpsum),
        attention.static_cute_tensor(t.dq_accum),
        attention.static_cute_tensor(t.dk_accum),
        attention.static_cute_tensor(t.dv_accum),
    )
    family_args: list = []
    for family in families:
        if family.name == "down_dw":
            family_args.append(decoder_layer.stacked_gemm_operand(tensors.slabs[family.b], True))
            family_args.append(decoder_layer.stacked_gemm_operand(tensors.slabs[family.a], True))
            family_args.append(decoder_layer.stacked_gemm_operand(tensors.slabs[family.d], True))
        else:
            family_args.append(decoder_layer.stacked_gemm_operand(tensors.slabs[family.a], family.a_t))
            family_args.append(decoder_layer.stacked_gemm_operand(tensors.slabs[family.b], True))
            family_args.append(decoder_layer.stacked_gemm_operand(tensors.slabs[family.d], False))
    # The two activation-chain operands (O forward's residual input and down forward's second
    # output), indexed by the layer (generation_slot word 1); the family operands above sit on
    # the layer slots.
    family_args.append(decoder_layer.stacked_gemm_operand(tensors.activation_chain[:-1], False))
    family_args.append(decoder_layer.stacked_gemm_operand(tensors.activation_chain[1:], False))
    family_args.append(from_dlpack(tensors.row_table, assumed_align=16, enable_tvm_ffi=True))
    attention_gradients = (
        attention.static_cute_tensor(t.dq),
        attention.static_cute_tensor(t.dk),
        attention.static_cute_tensor(t.dv),
    )
    shell_pairs = _shell_operand_pairs(tensors.shell)
    compile_shell = tuple(pair[0] for pair in shell_pairs)
    runtime_shell = tuple(pair[1] for pair in shell_pairs)
    runtime_fabric = tuple(decoder_fabric.runtime_tensors())
    if len(runtime_fabric) != len(DECODER_FABRIC_ARGUMENT_NAMES):
        raise ValueError(
            f"decoder fabric returned {len(runtime_fabric)} runtime tensors"
        )
    compile_fabric = tuple(_dynamic_tensor(tensor) for tensor in runtime_fabric)
    runtime_optimizer = tuple(optimizer_tail.runtime_tensors())
    if len(runtime_optimizer) != len(OPTIMIZER_ARGUMENT_NAMES):
        raise ValueError(
            f"optimizer tail returned {len(runtime_optimizer)} runtime tensors"
        )
    compile_optimizer = tuple(
        _dynamic_tensor(tensor) for tensor in runtime_optimizer
    )
    runtime_full_shell = tuple(full_shell.runtime_tensors())
    if len(runtime_full_shell) != len(
        SHELL_FABRIC_AND_SCHEDULER_ARGUMENT_NAMES
    ):
        raise ValueError(
            f"full shell returned {len(runtime_full_shell)} runtime tensors"
        )
    compile_full_shell = tuple(
        _dynamic_tensor(tensor) for tensor in runtime_full_shell
    )
    compile_args = (
        *attention_args,
        *family_args,
        slot,
        Int32(logical_depth),
        counter,
        *attention_gradients,
        *compile_shell,
        *compile_fabric,
        *compile_optimizer,
        *compile_full_shell,
        *(_dynamic_tensor(tensor) for tensor in nstep),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )

    klass, _source = training_kernel(families)
    program = klass(capacity, workspace_rows=t.workspace_rows, families=families)
    function = cute.compile(
        program, *compile_args, options="--enable-tvm-ffi"
    )

    page_limit = PROGRAM_SMEM_PAGE_BYTES
    for name, size in program.member_smem_bytes().items():
        if size > page_limit:
            raise AssertionError(f"{name} shared storage {size} exceeds the page")
    for name, member in (
        ("head_forward", program.head_phase.head_forward),
        ("head_dx", program.head_phase.head_dx),
        ("head_dw", program.head_phase.head_dw),
    ):
        size = int(member.shared_storage.size_in_bytes())
        if size > page_limit:
            raise AssertionError(f"{name} shared storage {size} exceeds the page")

    prefix, suffix = runtime_prefix_suffix(tensors)
    return CompiledTrainingProgram(
        function,
        prefix,
        suffix,
        runtime_shell,
        runtime_fabric,
        runtime_optimizer,
        runtime_full_shell,
        tuple(nstep),
    )
