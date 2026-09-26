"""The allocator every member uses to take its shared memory from the program's page.

The kernel allocates one dynamic shared-memory page at entry and stores its pointer in
``ProgramSmemAllocator.page``. The FA4 and Quack bodies construct a ``ProgramSmemAllocator``
where their upstream code constructs CUTLASS's ``SmemAllocator``, and the program's own bodies
use it too. It has the same ``allocate`` and ``allocate_tensor`` signatures, but it hands out
pieces of that one page, so every member shares it.
"""

from __future__ import annotations

import cutlass.cute as cute
from cutlass.utils import SmemPartition


class ProgramSmemAllocator:
    """Hands out the page: body structs at byte 0, tensor scratch above the barrier prefix.

    ``page`` and ``capacity_bytes`` are class attributes: the kernel sets ``page`` at entry,
    ``anchors.reanchor_smem_page`` replaces it with an anchored copy before member phases, and
    the RMSNorm phases narrow both to one shard's half (``quack_rmsnorm_bodies``). Each instance
    has its own scratch cursor.

    The two allocation paths are not interchangeable. ``allocate`` binds a body's shared-storage
    struct at page byte 0, the base the FA4 and Quack storage layouts require; it reserves
    nothing and does not move the cursor. ``allocate_tensor`` hands out scratch from the cursor,
    and ``_reserve`` never lets scratch start below ``STRUCT_BARRIER_PREFIX_BYTES``, because the
    first bytes of a struct bound at byte 0 hold the body's TMA and consumer mbarriers. Scratch
    written over them corrupts barrier state with writes that are all in bounds, so no bounds
    check or fence would catch it.
    """

    page = None
    # The bound _reserve checks tensor scratch against.
    capacity_bytes = 230400

    # Scratch starts at or above this offset, so the mbarriers at the start of a struct bound at
    # byte 0 must fit below it: 128 bytes hold 16 mbarriers of 8 bytes. Raise it if a body's
    # struct ever places more barriers at the page base.
    STRUCT_BARRIER_PREFIX_BYTES = 128

    def __init__(self):
        self.offset = type(self).STRUCT_BARRIER_PREFIX_BYTES
        self._struct_reserved_offset = None
        self._struct_reserved_bytes = 0
        self._struct_reserved_used = 0

    def _reserve(self, size_bytes, alignment):
        """Reserve ``size_bytes`` at the cursor's next ``alignment`` boundary; return the offset."""

        # Scratch never starts inside the barrier prefix, whatever the cursor holds.
        prefix = type(self).STRUCT_BARRIER_PREFIX_BYTES
        start = max(self.offset, prefix)
        offset = ((start + alignment - 1) // alignment) * alignment
        end = offset + size_bytes
        assert offset >= prefix, "program tensor scratch overlaps struct barrier storage"
        assert end <= self.capacity_bytes, "program-local shared-memory page overflow"
        self.offset = end
        return offset

    def allocate(self, storage_type, byte_alignment=1):
        """Bind the shared-storage struct ``storage_type`` at page byte 0 and return it."""

        del byte_alignment
        assert self.page is not None
        # Quack's GEMM body allocates its persistent scheduler's scratch separately, with
        # partition=SmemPartition.RESERVED. The program's GEMM storage structs keep a sched_data
        # field sized for it; remember that field so allocate_tensor serves RESERVED requests
        # from it, inside the struct.
        annotations = getattr(storage_type, "_annotations", {})
        offsets = getattr(storage_type, "_offsets", {})
        sched_data = annotations.get("sched_data")
        if sched_data is not None and "sched_data" in offsets:
            self._struct_reserved_offset = int(offsets["sched_data"])
            self._struct_reserved_bytes = int(sched_data.size_in_bytes)
            self._struct_reserved_used = 0
        return storage_type(self.page)

    def allocate_tensor(
        self,
        element_type,
        layout,
        byte_alignment=1,
        swizzle=None,
        *,
        partition=SmemPartition.USER,
        loc=None,
        ip=None,
    ):
        """A tensor from the cursor, or from the bound struct's sched_data field if RESERVED."""

        assert self.page is not None
        size_bytes = cute.size_in_bytes(element_type, layout)
        if partition == SmemPartition.RESERVED:
            assert self._struct_reserved_offset is not None, (
                "RESERVED shared-memory request has no sched_data ABI field"
            )
            start = self._struct_reserved_offset + self._struct_reserved_used
            offset = ((start + byte_alignment - 1) // byte_alignment) * byte_alignment
            end = offset + size_bytes
            assert end <= self._struct_reserved_offset + self._struct_reserved_bytes, (
                "RESERVED scheduler scratch exceeds the selected sched_data ABI field"
            )
            self._struct_reserved_used = end - self._struct_reserved_offset
        else:
            offset = self._reserve(size_bytes, byte_alignment)
        pointer = cute.recast_ptr(
            self.page + offset, swizzle, dtype=element_type, loc=loc, ip=ip
        )
        return cute.make_tensor(pointer, layout, loc=loc, ip=ip)
