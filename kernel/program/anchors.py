"""Anchors: identity moves that keep address computations where the addresses are used.

An anchor passes a value through an inline-asm ``mov`` with side effects: ``anchor_address`` for
64-bit addresses, ``anchor_index`` for 32-bit values (indices and shared-memory addresses). LLVM
cannot see through the move or hoist it, so an address derived from an anchored value is computed
where it is used instead of at kernel entry, where it would stay live across other members'
224-register phases. The anchored value is opaque, so any alignment or range LLVM knew for it is
lost; the helpers re-declare the alignment their inputs guarantee.

The helpers anchor a global tensor's base (``anchored_tensor``), a Quack tile scheduler's
parameters (``anchor_scheduler_params``), the page (``reanchor_smem_page``, bound on the program
class and called at the start of member phases) and the tensor behind a row-table entry
(``row_table_view``).
"""

from __future__ import annotations

import dataclasses

import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm as mlir_llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from program_smem import ProgramSmemAllocator


@dsl_user_op
def anchor_address(value, *, loc=None, ip=None) -> Int64:
    return Int64(
        mlir_llvm.inline_asm(
            T.i64(),
            [Int64(value).ir_value(loc=loc, ip=ip)],
            "mov.b64 $0, $1;",
            "=l,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def anchor_index(value, *, loc=None, ip=None) -> Int32:
    return Int32(
        mlir_llvm.inline_asm(
            T.i32(),
            [Int32(value).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
        )
    )


def anchored_tensor(tensor: cute.Tensor):
    """``tensor`` with its base address anchored; the base must be 16-byte aligned."""

    return cute.make_tensor(
        cute.make_ptr(
            tensor.element_type,
            anchor_address(Int64(tensor.iterator.toint())),
            mem_space=tensor.iterator.memspace,
            assumed_align=16,
        ),
        tensor.layout,
    )


def _is_global_tensor(value) -> bool:
    """Whether ``value`` is a tensor in global (or generic) memory with an integer address."""

    if not isinstance(value, cute.Tensor):
        return False
    iterator = getattr(value, "iterator", None)
    if iterator is None or not hasattr(iterator, "toint"):
        return False
    return getattr(iterator, "memspace", None) in (
        cute.AddressSpace.gmem,
        cute.AddressSpace.generic,
    )


def anchor_scheduler_params(params):
    """Quack tile-scheduler ``Params`` with their grid extents and group count anchored.

    ``problem_shape_ncluster_mnl`` and ``num_groups_regular`` pass through ``anchor_index``, and a
    global ``batch_idx_permute`` table is re-pointed through an anchored address. The GEMM runners
    call this where each member runs, so the scheduler's values are computed there.
    """

    params = dataclasses.replace(
        params,
        problem_shape_ncluster_mnl=tuple(
            anchor_index(Int32(extent)) for extent in params.problem_shape_ncluster_mnl
        ),
        num_groups_regular=anchor_index(Int32(params.num_groups_regular)),
    )
    table = getattr(params, "batch_idx_permute", None)
    if table is None or not _is_global_tensor(table):
        return params
    return dataclasses.replace(params, batch_idx_permute=anchored_tensor(table))


def anchored_smem_page(page):
    """The page pointer, anchored; the program allocates the page 1024-byte aligned."""

    return cute.make_ptr(
        page.value_type,
        anchor_index(page.toint()),
        mem_space=page.memspace,
        assumed_align=1024,
    )


def reanchor_smem_page(self) -> None:
    """Point ``ProgramSmemAllocator.page`` at a freshly anchored copy of the kernel's page.

    The first call records the page the kernel allocated in ``self.smem_page_origin``; every call
    anchors that origin again, so each phase computes the page address itself.
    """

    if self.smem_page_origin is None:
        self.smem_page_origin = ProgramSmemAllocator.page
    ProgramSmemAllocator.page = anchored_smem_page(self.smem_page_origin)


@cute.jit
def row_table_view(row_table, memspace, index, dtype, layout):
    """The tensor at the anchored address ``row_table[index]``, with ``dtype`` and ``layout``.

    The pointer is declared 16-byte aligned; ``training_tensors`` checks every row-table address.
    """

    from cutlass import Int64

    address = anchor_address(Int64(row_table[index]))
    return cute.make_tensor(
        cute.make_ptr(dtype, address, mem_space=memspace, assumed_align=16), layout
    )
