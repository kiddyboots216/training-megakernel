"""The grid-wide barrier the program's CTAs pass between phases.

``grid_barrier`` works on one Int32 word in global memory, the way CUDA's cooperative-groups
``grid.sync()`` does. After a CTA-wide sync, thread 0 of each CTA fences and adds to the word
with release semantics: CTA 0 adds ``2**31 - (PROGRAM_CTAS - 1)`` and every other CTA adds 1.
The arrivals of one barrier add exactly ``2**31``, so the last arrival flips bit 31 and leaves
the other bits where they were, and each CTA spins on acquire loads until bit 31 differs from
the value its own add returned. A second CTA-wide sync then releases the rest of the CTA.

The word returns to its starting value every two barriers, so it never needs a reset. A CTA
that leaves one barrier and arrives at the next cannot flip the bit again before every CTA has
left the first, because the next flip needs every CTA's arrival. The host zeroes the word before
the first launch. The spin relies on the cooperative launch: all 132 CTAs are resident at once.
"""

from __future__ import annotations

import cutlass.cute as cute
from cutlass import Int32
from model import PROGRAM_CTAS


@cute.jit
def grid_barrier(barrier_word: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    cute.arch.sync_threads()
    if tidx == 0:
        increment = Int32(1)
        if bidx == 0:
            increment = Int32(2**31 - (PROGRAM_CTAS - 1))
        cute.arch.fence_acq_rel_gpu()
        pointer = barrier_word.iterator
        before = cute.arch.atomic_add(pointer, increment, sem="release", scope="gpu")
        current = cute.arch.atomic_add(pointer, Int32(0), sem="acquire", scope="gpu")
        while ((before ^ current) & Int32(-2147483648)) == Int32(0):
            current = cute.arch.atomic_add(pointer, Int32(0), sem="acquire", scope="gpu")
    cute.arch.sync_threads()


def grid_phase_barrier(self, barrier_word: cute.Tensor):
    """`grid_barrier` with a method's signature, bound as `TrainingProgram.grid_phase_barrier`."""

    del self
    grid_barrier(barrier_word)
