"""Inline-PTX memory operations for the program's communication, as CuTe DSL user ops.

The multicast operations use PTX ``multimem`` (NVLink SHARP) on a symmetric-memory multicast
address, which reaches every rank's copy of an arena: ``multicast_min_acquire_u32``,
``multicast_sum_f32`` and ``multicast_sum_2xf32`` read all eight copies and return their minimum
or sum, and ``multicast_store_b64`` and ``copy_to_multicast_b128`` write all of them. The other
operations take one ordinary address, local or a peer's: plain loads and stores, the system-scope
acquire load and release store of the epoch words that publish data to other ranks and to the
host, ``fence_sys``, the relaxed store and compare-and-swap that record a failed wait, and the
``%globaltimer`` read for wait deadlines. ``step_exit_branch`` and ``step_exit_label`` form the
step loop's exit.

Every memory operation has one result register; stores and the fence return a dummy zero that
callers discard (``_ = store_f32(...)``). The data loads (``load_b64``, ``load_u64``, ``load_f32``
and the multicast sums) have no side effects, so the compiler may reorder, overlap or merge them.
Callers place them after the wait that observed the data's publication, an acquire wait loop or a
grid barrier that follows one, and rely on that control dependence for their ordering.
"""

from __future__ import annotations

from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm as mlir_llvm
from cutlass.cutlass_dsl import T, dsl_user_op


def _inline_asm(result_type, operands, template, constraints, *, side_effects=True):
    """Inline PTX with one result register, in the AT&T dialect.

    ``side_effects=False`` is only for the data loads (see the module docstring).
    """

    return mlir_llvm.inline_asm(
        result_type,
        operands,
        template,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def global_timer_ns(*, loc=None, ip=None) -> Int64:
    """The ``%globaltimer`` clock in nanoseconds, for wait deadlines and wait times."""

    return Int64(_inline_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l"))


@dsl_user_op
def multicast_min_acquire_u32(address, *, loc=None, ip=None) -> Int32:
    """Minimum of a u32 epoch word over every rank's copy, read through its multicast address.

    The minimum reaches ``e`` once every rank has published epoch ``e`` or later, and the acquire
    orders this thread's later loads after those publications.
    """

    return Int32(
        _inline_asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "multimem.ld_reduce.acquire.sys.global.min.u32 $0, [$1];",
            "=r,l",
        )
    )


@dsl_user_op
def multicast_sum_f32(address, *, loc=None, ip=None) -> Float32:
    """Sum of one FP32 element over every rank's copy, read through its multicast address.

    A relaxed load with no side effects, so independent sums can overlap. Each rank release-stores
    its READY epoch after its data, so a single acquire wait on those epochs, before these loads,
    orders all of them.
    """

    return Float32(
        _inline_asm(
            T.f32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "multimem.ld_reduce.relaxed.sys.global.add.f32 $0, [$1];",
            "=f,l",
            side_effects=False,
        )
    )


@dsl_user_op
def multicast_sum_2xf32(address, *, loc=None, ip=None) -> Int64:
    """Sums of two consecutive FP32 elements over every rank's copy, packed in one 64-bit value.

    The ``v2.f32`` reduction returns two registers; packing them keeps the single result of the
    other operations, and callers store the pair with ``store_b64``. No side effects, as for
    ``multicast_sum_f32``.
    """

    return Int64(
        _inline_asm(
            T.i64(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "{ .reg .f32 plo, phi; "
            "multimem.ld_reduce.relaxed.sys.global.add.v2.f32 {plo, phi}, [$1]; "
            "mov.b64 $0, {plo, phi}; }",
            "=l,l",
            side_effects=False,
        )
    )


@dsl_user_op
def load_b64(address, *, loc=None, ip=None) -> Int64:
    """Plain 64-bit load of two FP32 or four BF16 values; no side effects."""

    return Int64(
        _inline_asm(
            T.i64(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.b64 $0, [$1];",
            "=l,l",
            side_effects=False,
        )
    )


@dsl_user_op
def store_b64(address, value, *, loc=None, ip=None) -> Int32:
    """Plain 64-bit store of two FP32 or four BF16 values."""

    return Int32(
        _inline_asm(
            T.i32(),
            [
                Int64(address).ir_value(loc=loc, ip=ip),
                Int64(value).ir_value(loc=loc, ip=ip),
            ],
            "st.global.b64 [$1], $2; mov.b32 $0, 0;",
            "=r,l,l",
        )
    )


@dsl_user_op
def multicast_store_b64(address, value, *, loc=None, ip=None) -> Int32:
    """Store one 64-bit word into every rank's copy through a multicast address.

    The head weight all-gather writes each rank's rows of a vocabulary panel with it.
    """

    return Int32(
        _inline_asm(
            T.i32(),
            [
                Int64(address).ir_value(loc=loc, ip=ip),
                Int64(value).ir_value(loc=loc, ip=ip),
            ],
            "multimem.st.relaxed.sys.global.b64 [$1], $2; mov.b32 $0, 0;",
            "=r,l,l",
        )
    )


@dsl_user_op
def copy_b128(source_address, destination_address, *, loc=None, ip=None) -> Int32:
    """Copy one 16-byte word between two ordinary global addresses, both 16-byte aligned."""

    return Int32(
        _inline_asm(
            T.i32(),
            [
                Int64(source_address).ir_value(loc=loc, ip=ip),
                Int64(destination_address).ir_value(loc=loc, ip=ip),
            ],
            "{ .reg .b32 v0, v1, v2, v3; "
            "ld.global.v4.b32 {v0, v1, v2, v3}, [$1]; "
            "st.global.v4.b32 [$2], {v0, v1, v2, v3}; "
            "mov.b32 $0, 0; }",
            "=r,l,l",
        )
    )


@dsl_user_op
def copy_to_multicast_b128(source_address, destination_address, *, loc=None, ip=None) -> Int32:
    """Copy one 16-byte word from an ordinary address into every rank's copy of an arena.

    The decoder weight all-gather moves each rank's shard of a weight panel with it. The words are
    packed BF16 weights; ``ld.v4.f32`` and ``multimem.st.v4.f32`` move their bits without
    conversion, and ptxas accepts the vector ``multimem.st`` for ``.f32`` but not for ``.b32``.
    Both addresses must be 16-byte aligned, and every rank's shard of a weight panel is a whole
    number of 16-byte words.
    """

    return Int32(
        _inline_asm(
            T.i32(),
            [
                Int64(source_address).ir_value(loc=loc, ip=ip),
                Int64(destination_address).ir_value(loc=loc, ip=ip),
            ],
            "{ .reg .f32 v0, v1, v2, v3; "
            "ld.global.v4.f32 {v0, v1, v2, v3}, [$1]; "
            "multimem.st.relaxed.sys.global.v4.f32 "
            "[$2], {v0, v1, v2, v3}; mov.b32 $0, 0; }",
            "=r,l,l",
        )
    )


@dsl_user_op
def load_acquire_sys_u32(address, *, loc=None, ip=None) -> Int32:
    """System-scope acquire load of a u32 epoch or flag word, at a local or a peer's address."""

    return Int32(
        _inline_asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.acquire.sys.global.u32 $0, [$1];",
            "=r,l",
        )
    )


@dsl_user_op
def load_f32(address, *, loc=None, ip=None) -> Float32:
    """Plain FP32 load with no side effects, so independent loads in a loop can overlap."""

    return Float32(
        _inline_asm(
            T.f32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            side_effects=False,
        )
    )


@dsl_user_op
def store_f32(address, value, *, loc=None, ip=None) -> Int32:
    return Int32(
        _inline_asm(
            T.i32(),
            [
                Int64(address).ir_value(loc=loc, ip=ip),
                Float32(value).ir_value(loc=loc, ip=ip),
            ],
            "st.global.f32 [$1], $2; mov.b32 $0, 0;",
            "=r,l,f",
        )
    )


@dsl_user_op
def store_release_sys_u32(address, value, *, loc=None, ip=None) -> Int32:
    """System-scope release store of a u32 epoch or mailbox word.

    A reader that acquires the word sees every write ordered before the store.
    """

    return Int32(
        _inline_asm(
            T.i32(),
            [
                Int64(address).ir_value(loc=loc, ip=ip),
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "st.release.sys.global.u32 [$1], $2; mov.b32 $0, 0;",
            "=r,l,r",
        )
    )


@dsl_user_op
def store_relaxed_gpu_u32(address, value, *, loc=None, ip=None) -> Int32:
    """Relaxed gpu-scope store of a u32.

    The failure recorders fill in a status record with it after claiming the record with
    ``atomic_cas_gpu_u32``.
    """

    return Int32(
        _inline_asm(
            T.i32(),
            [
                Int64(address).ir_value(loc=loc, ip=ip),
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "st.relaxed.gpu.global.u32 [$1], $2; mov.b32 $0, 0;",
            "=r,l,r",
        )
    )


@dsl_user_op
def load_u64(address, *, loc=None, ip=None) -> Int64:
    """Plain 64-bit integer load, the same load as ``load_b64`` apart from the PTX type.

    No side effects, like ``load_b64``. Besides host-written tables it reads route row IDs that
    peers write during the step, so callers load those only after the wait for their publication.
    """

    return Int64(
        _inline_asm(
            T.i64(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.u64 $0, [$1];",
            "=l,l",
            side_effects=False,
        )
    )


@dsl_user_op
def store_u64(address, value, *, loc=None, ip=None) -> Int32:
    """Plain 64-bit integer store, the same store as ``store_b64`` apart from the PTX type."""

    return Int32(
        _inline_asm(
            T.i32(),
            [
                Int64(address).ir_value(loc=loc, ip=ip),
                Int64(value).ir_value(loc=loc, ip=ip),
            ],
            "st.global.u64 [$1], $2; mov.b32 $0, 0;",
            "=r,l,l",
        )
    )


@dsl_user_op
def fence_sys(*, loc=None, ip=None) -> Int32:
    """System-scope fence (``membar.sys``).

    Publishers issue it before the release store of an epoch, to order the global and peer stores
    of the data before that system-scope release.
    """

    return Int32(
        _inline_asm(
            T.i32(),
            [],
            "membar.sys; mov.b32 $0, 0;",
            "=r",
        )
    )


@dsl_user_op
def atomic_cas_gpu_u32(address, compare, value, *, loc=None, ip=None) -> Int32:
    """gpu-scope acquire-release compare-and-swap of a u32; returns the old value.

    The failure recorders use it so that only the first failed wait fills in the status record.
    """

    return Int32(
        _inline_asm(
            T.i32(),
            [
                Int64(address).ir_value(loc=loc, ip=ip),
                Int32(compare).ir_value(loc=loc, ip=ip),
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.global.acq_rel.gpu.cas.b32 $0, [$1], $2, $3;",
            "=r,l,r,r",
        )
    )


# The step loop's exit (resident_step.py). When another step follows, the branch skips the exit
# sentinel store and jumps to this label at the end of the kernel. ptxas assembles the branch as a
# predicated EXIT, which kernel/tools/patch_step_backedge.py rewrites after assembly into a branch
# to the kernel's first instruction; that rewritten EXIT is the step loop.
EXIT_LABEL_NAME = "RESIDENT_STEP_EXIT"


@dsl_user_op
def step_exit_branch(take_branch, *, loc=None, ip=None) -> None:
    """Branch to the exit label when ``take_branch`` is nonzero."""

    mlir_llvm.inline_asm(
        None,
        [Int32(take_branch).ir_value(loc=loc, ip=ip)],
        (
            "{ .reg .pred p; "
            "setp.ne.s32 p, $0, 0; "
            f"@p bra {EXIT_LABEL_NAME}; }}"
        ),
        "r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def step_exit_label(*, loc=None, ip=None) -> None:
    """Place the exit label; the step loop emits it right after the exit sentinel store."""

    mlir_llvm.inline_asm(
        None,
        [],
        f"{EXIT_LABEL_NAME}:",
        "",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
