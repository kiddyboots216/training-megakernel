"""Relabel warp and thread indices while a Quack GEMM member is traced.

Quack's SM90 GEMM kernel numbers its warps with the two MMA warpgroups at warps 0-7 and the
A/B-load producer at warps 8-11, and it and the CUTLASS pipeline helpers it calls pick roles
from `cute.arch.warp_idx()` and `cute.arch.thread_idx()`: Quack's producer and consumer
branches, `warp_group_idx = tidx // 128`, `MbarrierArray.mbarrier_init` (warp 0 initializes the
barriers) and `PipelineTmaAsync.init_empty_barrier_arrive_signal` (the thread index picks the
signaling threads). The program runs the producer on role 0 (physical warps 0-3, 56 registers)
and the MMA warpgroups on roles 1 and 2 (warps 4-11, 224 registers). `push_role_rotation(role)`
replaces the three `cute.arch` functions for the duration of one role's trace, so these indices
come out in Quack's numbering; `pop_role_rotation()` restores them.

Each replacement is written so that LLVM can bound its value to the role's window: 4 warps or
128 threads. Quack's role predicates then fold to constants and each role's region holds only
its own branch. If a role's region kept another role's branch, ptxas would drop the register
split, and the build would warn that the kernel has no `USETMAXREG` instruction.
"""

from __future__ import annotations

import cutlass.cute as cute
from cutlass import Int32


_cute_warp_idx = cute.arch.warp_idx
_cute_thread_idx = cute.arch.thread_idx
_cute_make_warp_uniform = cute.arch.make_warp_uniform

# The first physical warp of each role.
_ROLE_FIRST_WARP = {0: 0, 1: 4, 2: 8}

# The first warp and thread each role maps to in Quack's numbering: role 0 is Quack's
# producer warpgroup, roles 1 and 2 are its two MMA warpgroups.
_ROLE_FIRST_QUACK_WARP = {0: 8, 1: 0, 2: 4}
_ROLE_FIRST_QUACK_THREAD = {0: 256, 1: 0, 2: 128}

_active_role: list[int | None] = [None]
_rotation_depth = [0]


def _rotated_warp_idx(*, loc=None, ip=None):
    """The warp index in Quack's numbering: the role's first Quack warp plus `warp & 3`.

    On the four warps that run the role, this equals the plain rotation `(warp + 8) % 12`, and
    LLVM can see that the result lies in a 4-warp window, so Quack's tests against
    `ab_load_warp_id` (8) and against 4 fold to constants. The mask must be the last
    operation: an opaque step after it, such as `make_warp_uniform` or an anchor, hides the range.
    """

    role = _active_role[0]
    logical_base = _ROLE_FIRST_QUACK_WARP[role]
    physical_base = _ROLE_FIRST_WARP[role]
    warp = _cute_warp_idx(loc=loc, ip=ip)
    return Int32(logical_base) + ((warp - Int32(physical_base)) & Int32(3))


def _rotated_thread_idx(*, loc=None, ip=None):
    """The thread index in Quack's numbering: the role's first Quack thread plus `tidx & 127`.

    The result lies in a 128-thread window, so Quack's `warp_group_idx = tidx // 128` and the
    tests on it fold to constants. Role tests derived from the thread index need this as much
    as those derived from the warp index.
    """

    role = _active_role[0]
    logical_base = _ROLE_FIRST_QUACK_THREAD[role]
    tidx, tidy, tidz = _cute_thread_idx(loc=loc, ip=ip)
    return (Int32(logical_base) + (tidx & Int32(127)), tidy, tidz)


def _warp_uniform_identity(value, *, loc=None, ip=None):
    """`make_warp_uniform` as the identity, for the duration of the Quack trace.

    `make_warp_uniform` only marks a value as the same across the warp, but LLVM cannot see
    through it, so the range of the relabeled index would be lost. Quack and CUTLASS apply it
    to the indices they derive roles from. The relabeled indices are warp-uniform by
    construction, so the hint is not needed.
    """

    return Int32(value)


def push_role_rotation(role: int) -> None:
    """Relabel the warp and thread indices for `role` until `pop_role_rotation()`."""

    assert role in (0, 1, 2), role
    _rotation_depth[0] += 1
    _active_role[0] = role
    cute.arch.warp_idx = _rotated_warp_idx
    cute.arch.thread_idx = _rotated_thread_idx
    cute.arch.make_warp_uniform = _warp_uniform_identity


def pop_role_rotation() -> None:
    _rotation_depth[0] -= 1
    assert _rotation_depth[0] == 0, "unbalanced role rotation"
    _active_role[0] = None
    cute.arch.warp_idx = _cute_warp_idx
    cute.arch.thread_idx = _cute_thread_idx
    cute.arch.make_warp_uniform = _cute_make_warp_uniform
