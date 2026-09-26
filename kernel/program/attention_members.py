"""FlashAttention 4's SM90 forward and backward kernels as members of the program.

`AttentionForwardMember` and `AttentionBackwardMember` subclass FA4's `FlashAttentionForwardSm90`
and `FlashAttentionBackwardSm90` with the program's settings (BF16, head dimension 128, causal,
384 threads). Their `kernel` methods record the kernel arguments instead of launching: the
program traces FA4's `__call__` once per member on the host (see `fa4_forward_call` and
`fa4_backward_call`), and its kernel hands the recorded arguments to the member bodies in
`fa4_forward_kernel` and `fa4_backward_kernel`. `run_attention_forward_member` runs one forward
pass inside the program.

The constants fix the members' tiles and the length of the recorded argument tuples, which
`attention` and the generated kernel source index by position. The members use FA4's classes
without copying their code: they call FA4's constructors and override `kernel` and the
backward `__call__`.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, const_expr
from flash_attn.cute.flash_bwd_postprocess import FlashAttentionBackwardPostprocess  # noqa: TID253
from flash_attn.cute.flash_bwd_preprocess import FlashAttentionBackwardPreprocess  # noqa: TID253
from flash_attn.cute.flash_bwd_sm90 import FlashAttentionBackwardSm90  # noqa: TID253
from flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90  # noqa: TID253
from flash_attn.cute.utils import AuxData  # noqa: TID253

from fa4_forward_kernel import forward_role, forward_setup, persistent_forward_mma
from grid_barrier import grid_barrier
from model import HEAD_DIM, KV_HEADS, PROGRAM_THREADS, QUERY_HEADS

# The forward member's tiles (pack_gqa folds a KV head's query heads into tile_m) and stages.
FORWARD_TILE_M = 128
FORWARD_TILE_N = 128
FORWARD_STAGES = 2
# The backward member's tiles, and those of its preprocess and its dQ and dK/dV postprocess.
BACKWARD_TILE_M = 64
BACKWARD_TILE_N = 64
PREPROCESS_TILE_M = 64
DQ_POSTPROCESS_TILE_M = 64
DKV_POSTPROCESS_TILE_M = 128

# Length of the argument tuples the members record: FA4's forward kernel parameters (36), and
# FA4's backward kernel parameters (44) followed by the dQ postprocess (7), dK/dV postprocess (7)
# and preprocess (2) operands.
FORWARD_RECORDED_ARG_COUNT = 36
BACKWARD_RECORDED_ARG_COUNT = 60

# The workspace and tensor parameters that `training_step_kernel` declares and passes to
# `run_attention_backward`, in order.
ATTENTION_BACKWARD_TENSOR_NAMES = (
    "ws_lse_log2",
    "ws_dpsum",
    "ws_dq_accum",
    "ws_dk_accum",
    "ws_dv_accum",
    "prog_out",
    "prog_dout",
    "prog_lse",
    "prog_dq",
    "prog_dk",
    "prog_dv",
)


class _SkippedLaunch:
    """What the members' `kernel` returns: the program runs the bodies itself, so `launch` does
    nothing."""

    def launch(self, *args, **kwargs):
        del args, kwargs
        return None


class AttentionForwardMember(FlashAttentionForwardSm90):
    """FA4's forward; its `kernel` records the kernel arguments instead of launching.

    `mma` is `persistent_forward_mma`, so one CTA can run many tiles.
    """

    mma = persistent_forward_mma

    def __init__(self):
        super().__init__(
            BFloat16,
            HEAD_DIM,
            HEAD_DIM,
            QUERY_HEADS // KV_HEADS,
            is_causal=True,
            is_local=False,
            pack_gqa=True,
            tile_m=FORWARD_TILE_M,
            tile_n=FORWARD_TILE_N,
            num_stages=FORWARD_STAGES,
            num_threads=PROGRAM_THREADS,
            Q_in_regs=False,
            has_aux_tensors=False,
            q_subtile_factor=1,
            intra_wg_overlap=False,
            mma_pv_is_rs=False,
        )
        self.recorded_args = None

    def kernel(self, *forward_args):
        self.recorded_args = forward_args
        return _SkippedLaunch()


class AttentionBackwardMember(FlashAttentionBackwardSm90):
    """FA4's backward, preprocess and postprocess; calling it records the kernel arguments."""

    def __init__(self):
        super().__init__(
            BFloat16,
            HEAD_DIM,
            HEAD_DIM,
            QUERY_HEADS // KV_HEADS,
            True,  # causal
            is_local=False,
            deterministic=False,
            tile_m=BACKWARD_TILE_M,
            tile_n=BACKWARD_TILE_N,
            Q_stage=2,
            dO_stage=2,
            PdS_stage=2,
            SdP_swapAB=False,
            dKV_swapAB=False,
            dQ_swapAB=False,
            AtomLayoutMSdP=1,
            AtomLayoutNdKV=1,
            AtomLayoutMdQ=1,
            num_threads=PROGRAM_THREADS,
            V_in_regs=False,
            has_aux_tensors=False,
            q_subtile_factor=2,
            dQ_single_wg=False,
        )
        self.postprocess_dq = FlashAttentionBackwardPostprocess(
            BFloat16,
            HEAD_DIM,
            90,
            DQ_POSTPROCESS_TILE_M,
            256,
            1,
            False,
        )
        self.postprocess_dkv = FlashAttentionBackwardPostprocess(
            BFloat16,
            HEAD_DIM,
            90,
            DKV_POSTPROCESS_TILE_M,
            256,
            2,
            False,
        )
        self.preprocess = FlashAttentionBackwardPreprocess(
            BFloat16,
            HEAD_DIM,
            HEAD_DIM,
            tile_m=PREPROCESS_TILE_M,
            num_threads=256,
            use_padded_offsets=True,
            nheads_major=False,
            pack_gqa=False,
            qhead_per_kvhead=1,
            nheads_kv=1,
        )
        self._postprocess_dq_args = None
        self._postprocess_dkv_args = None
        self._preprocess_args = None
        self.recorded_args = None
        # The program binds `_call_setup` to `fa4_backward_call.backward_call_setup`, FA4's
        # `__call__` on the program's backward scheduler.

    @cute.jit
    def prepare_postprocess_arguments(self):
        """Sets up the postprocess and preprocess objects and collects their arguments, which
        `kernel` appends to the recorded ones. The program's own preprocess does not use the
        preprocess pair."""

        self.postprocess_dq.tiled_mma = self.postprocess_dq._get_tiled_mma()
        self.postprocess_dq._setup_attributes()
        self._postprocess_dq_args = (
            self.postprocess_dq.tiled_mma,
            self.postprocess_dq.dQ_swapAB,
            self.postprocess_dq.sdQaccum_layout,
            self.postprocess_dq.sdQ_layout,
            self.postprocess_dq.g2s_tiled_copy_dQaccum,
            self.postprocess_dq.s2r_tiled_copy_dQaccum,
            self.postprocess_dq.gmem_tiled_copy_dQ,
        )
        self.postprocess_dkv.tiled_mma = self.postprocess_dkv._get_tiled_mma()
        self.postprocess_dkv._setup_attributes()
        self._postprocess_dkv_args = (
            self.postprocess_dkv.tiled_mma,
            self.postprocess_dkv.dQ_swapAB,
            self.postprocess_dkv.sdQaccum_layout,
            self.postprocess_dkv.sdQ_layout,
            self.postprocess_dkv.g2s_tiled_copy_dQaccum,
            self.postprocess_dkv.s2r_tiled_copy_dQaccum,
            self.postprocess_dkv.gmem_tiled_copy_dQ,
        )
        self.preprocess._setup_attributes()
        self._preprocess_args = (
            self.preprocess.gmem_tiled_copy_O,
            self.preprocess.gmem_tiled_copy_dQaccum,
        )

    def kernel(self, *fa4_args):
        # FA4's backward kernel parameters (44) + dq post (7) + dkv post (7) + preprocess (2).
        self.recorded_args = (
            *fa4_args,
            *self._postprocess_dq_args,
            *self._postprocess_dkv_args,
            *self._preprocess_args,
        )
        return _SkippedLaunch()

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mSeqUsedQ: cute.Tensor,
        mSeqUsedK: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        """FA4's backward `__call__` interface: runs `_call_setup`, which records the kernel
        arguments through `kernel`.

        """

        self.prepare_postprocess_arguments()
        self._call_setup(
            self,
            mQ,
            mK,
            mV,
            mdO,
            mLSE,
            mdPsum,
            mdQaccum,
            mdK,
            mdV,
            softmax_scale,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            None,
            None,
            None,
            None,
            None,
            AuxData(),
            None,
            stream,
        )


@cute.jit
def run_attention_forward_member(
    self,
    role: cutlass.Constexpr[int],
    phase_counter: cute.Tensor,
    *forward_args,
):
    """One forward pass of the member: each role runs its FA4 body, then the grid meets.

    The member runs in the program's register split (56 registers for role 0, 224 for roles 1
    and 2), which the program sets before the layer loop; it issues no `setmaxnreg` of its own.
    """

    state = forward_setup(self.forward, *forward_args)
    forward_role(self.forward, *forward_args, *state, role)

    # FA4's forward leaves one half-arrival outstanding on named barrier 2, warpgroup 1's
    # warp-scheduler barrier: `mma_init` arrives on it once more than the tile loop waits.
    # Role 1 completes it here, or the next forward pass would deadlock. The arrive is
    # selected by the constexpr role rather than by a runtime warp-index test.
    if const_expr(role == 1):
        cute.arch.barrier_arrive(barrier_id=2, number_of_threads=256)
    cute.arch.sync_threads()

    grid_barrier(phase_counter)


@cute.jit
def assume_pointer_aligned(tensor: cute.Tensor, alignment: cutlass.Constexpr[int]):
    """`tensor` with its pointer declared `alignment`-byte aligned: the torch allocation behind
    it guarantees that, but the tensor's type does not carry it."""

    pointer = cute.make_ptr(
        dtype=tensor.element_type,
        value=tensor.iterator.toint(),
        mem_space=tensor.iterator.memspace,
        assumed_align=alignment,
    )
    return cute.make_tensor(pointer, tensor.layout)
