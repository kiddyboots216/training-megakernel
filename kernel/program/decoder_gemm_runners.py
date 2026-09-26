"""The runners of the decoder's GEMM phases.

`TrainingProgram` binds these as methods, and the generated step calls one per GEMM phase in
each role's code, with the member's 17 prepared arguments as `gemm_args`.  A runner re-anchors
the page and anchors the scheduler parameters (element 16) so that LLVM cannot hoist them to
kernel entry, binds those parameters to the family's queue state
(`tile_schedulers.dynamic_scheduler_params`), relabels the warp indices for the role
(`gemm_role_rotation`) and traces `quack_gemm_bodies.gemm_body` with an
`EpochDynamicTileScheduler`.  All but the qkv dX runner then end the
phase at the grid barrier (`finish_decoder_gemm`).  A family's queue state is reset once per
step and every layer's phase of the family reuses it; the Epoch schedulers keep it valid
across those phases.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from anchors import anchor_scheduler_params
from gemm_role_rotation import pop_role_rotation, push_role_rotation
from quack_gemm_bodies import gemm_body
from tile_schedulers import (
    EpochDynamicTileScheduler,
    EpochDynamicTileSchedulerWithoutCta0,
    dynamic_scheduler_params,
)


@cute.jit
def run_decoder_gemm(
    self,
    fam: cutlass.Constexpr[int],
    role: cutlass.Constexpr[int],
    phase_counter: cute.Tensor,
    scheduler_state: cute.Tensor,
    *gemm_args,
):
    """Run family `fam`'s member for `role` on `EpochDynamicTileScheduler`."""

    self.reanchor_smem_page()
    member = self._members[fam]
    sched = anchor_scheduler_params(gemm_args[16])
    sched = dynamic_scheduler_params(sched, scheduler_state)
    push_role_rotation(role)
    gemm_body(
        member,
        *gemm_args[:16],
        sched,
        EpochDynamicTileScheduler,
    )
    pop_role_rotation()
    finish_decoder_gemm(self, phase_counter)


@cute.jit
def run_qkv_dx_gemm_without_cta0(
    self,
    fam: cutlass.Constexpr[int],
    role: cutlass.Constexpr[int],
    phase_counter: cute.Tensor,
    scheduler_state: cute.Tensor,
    *gemm_args,
):
    """Run the qkv dX member on CTAs 1..131 while CTA 0 reduces the q/k norm gradients.

    It returns after a CTA barrier, without the grid barrier; `phase_counter` is unused.
    """

    assert const_expr(self.families[fam].name == "qkv_dx")
    self.reanchor_smem_page()
    member = self._members[fam]
    sched = anchor_scheduler_params(gemm_args[16])
    sched = dynamic_scheduler_params(sched, scheduler_state)
    push_role_rotation(role)
    gemm_body(
        member,
        *gemm_args[:16],
        sched,
        EpochDynamicTileSchedulerWithoutCta0,
    )
    pop_role_rotation()
    # The generated code follows the CTA 0 / qkv dX branch with one grid barrier for both
    # sides, so every CTA arrives at the barrier once.
    cute.arch.sync_threads()


@cute.jit
def run_initial_gate_up_gemm(
    self,
    fam: cutlass.Constexpr[int],
    role: cutlass.Constexpr[int],
    phase_counter: cute.Tensor,
    scheduler_state: cute.Tensor,
    *gemm_args,
):
    """Run the activation-only gate/up member (`self.initial_gate_up_aux_member`).

    The initial forward runs it in the layers whose gate/up preactivation the backward
    recomputes; it stores only the SwiGLU output.  It takes the gate/up forward family's queue
    (`fam` must be that family), whose tile count it shares.
    """

    assert const_expr(self.families[fam].name == "gate_up_fwd")
    self.reanchor_smem_page()
    member = self.initial_gate_up_aux_member
    sched = anchor_scheduler_params(gemm_args[16])
    sched = dynamic_scheduler_params(sched, scheduler_state)
    push_role_rotation(role)
    gemm_body(
        member,
        *gemm_args[:16],
        sched,
        EpochDynamicTileScheduler,
    )
    pop_role_rotation()
    finish_decoder_gemm(self, phase_counter)


@cute.jit
def finish_decoder_gemm(
    self,
    phase_counter: cute.Tensor,
):
    """End a decoder GEMM phase: a CTA barrier, then the grid barrier.

    The members run without ping-pong, so no MMA or epilogue drain precedes the barriers.
    """

    cute.arch.sync_threads()
    self.grid_phase_barrier(phase_counter)
