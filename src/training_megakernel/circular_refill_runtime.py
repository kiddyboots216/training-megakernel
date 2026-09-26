"""Host runtime for two-slot HBM token refill during one CUfunction.

The producer is a CPU thread.  It prepares each token window of the bundle's shape,
waits until the device releases the destination slot, copies the seven payloads
on a dedicated non-default CUDA stream, waits for that stream's completion,
joins a WORLD8 host/Gloo gate, and only then publishes ``ready``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import mmap
import os
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from . import contract
from . import resident_protocol as P
from .geometry import Geometry
from .host_memory import register_mapped, unregister_mapped
from .nstep import DEFAULT_RING_SLOTS, power_of_two, safe_run_id
from .shards import TokenShardRing
from .workload import TokenWindow

TOKEN_SLOTS = 2
TOKEN_SLOT_MASK = TOKEN_SLOTS - 1
MAPPING_BYTES = 4096
_ATOMIC_ACQUIRE = 2
_ATOMIC_RELEASE = 3


def _slot_extents(sequence: int) -> tuple[int, ...]:
    """Elements of one slot of each of the seven token-ring tensors."""

    return (sequence, sequence, 1, sequence, sequence, contract.WORLD_SIZE, contract.WORLD_SIZE)


class _AtomicLibrary:
    def __init__(self) -> None:
        path = ctypes.util.find_library("atomic") or "libatomic.so.1"
        library = ctypes.CDLL(path)
        self.load4 = getattr(library, "__atomic_load_4")
        self.load4.argtypes = (ctypes.c_void_p, ctypes.c_int)
        self.load4.restype = ctypes.c_uint32
        self.store4 = getattr(library, "__atomic_store_4")
        self.store4.argtypes = (ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int)
        self.store4.restype = None
        self.library = library


_ATOMIC = _AtomicLibrary()


class MappedRefillControl:
    """One rank's file-backed, mapped ready/free generation page."""

    def __init__(self, *, run_id: str, directory: Path = Path("/dev/shm")) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"mk_refill_{safe_run_id(run_id)}_{os.getpid()}"
        self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            os.ftruncate(self._fd, MAPPING_BYTES)
            self._mapping = mmap.mmap(
                self._fd,
                MAPPING_BYTES,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        except BaseException:
            os.close(self._fd)
            self.path.unlink(missing_ok=True)
            raise
        self.address = ctypes.addressof(ctypes.c_char.from_buffer(self._mapping))
        self.device_address = 0
        self._closed = False

    @staticmethod
    def ready_offset(slot: int) -> int:
        return P.REFILL_READY_BASE + slot * P.REFILL_SLOT_STRIDE

    @staticmethod
    def free_offset(slot: int) -> int:
        return P.REFILL_FREE_BASE + slot * P.REFILL_SLOT_STRIDE

    def load_acquire(self, offset: int) -> int:
        return int(_ATOMIC.load4(self.address + offset, _ATOMIC_ACQUIRE))

    def store_release(self, offset: int, value: int) -> None:
        _ATOMIC.store4(self.address + offset, value, _ATOMIC_RELEASE)

    def register_cuda(self, *, device_index: int) -> int:
        self.device_address = register_mapped(self.address, MAPPING_BYTES, device_index)
        return self.device_address

    def publish_ready(self, *, slot: int, generation: int) -> None:
        self.store_release(self.ready_offset(slot), generation)

    def wait_free(
        self,
        *,
        slot: int,
        generation: int,
        timeout_seconds: float,
        progress: Callable[[], object] | None = None,
    ) -> None:
        """Wait for the device to free ``slot``: at most ``timeout_seconds`` without a
        change in ``progress()``, which lets an in-launch checkpoint hold the slot for
        as long as it keeps advancing."""

        deadline = time.monotonic() + timeout_seconds
        offset = self.free_offset(slot)
        last_progress = None if progress is None else progress()
        while True:
            observed = self.load_acquire(offset)
            if observed == generation:
                return
            if observed > generation:
                self.request_abort()
                raise RuntimeError(
                    f"future free generation {observed} for slot={slot}; "
                    f"expected={generation}"
                )
            status = self.load_acquire(P.REFILL_STATUS)
            if status:
                raise RuntimeError(f"device refill wait failed with status={status}")
            if progress is not None:
                observed_progress = progress()
                if observed_progress != last_progress:
                    last_progress = observed_progress
                    deadline = time.monotonic() + timeout_seconds
            if time.monotonic() >= deadline:
                self.request_abort()
                raise TimeoutError(
                    f"device did not free slot={slot}, generation={generation}"
                )
            time.sleep(0.0005)

    def request_abort(self) -> None:
        self.store_release(P.REFILL_ABORT, 1)

    def close(self) -> None:
        if self._closed:
            return
        if self.device_address:
            unregister_mapped(self.address)
            self.device_address = 0
        self._mapping.close()
        os.close(self._fd)
        self.path.unlink(missing_ok=True)
        self._closed = True


@dataclass
class CircularNstepRuntimeState:
    control: torch.Tensor
    records: torch.Tensor
    resident_steps: int

    @classmethod
    def allocate(
        cls,
        *,
        resident_steps: int,
        device: torch.device,
        refill_device_address: int,
        base_generation: int,
        timeout_ns: int,
        mailbox_address: int = 0,
        telemetry_ring_slots: int = DEFAULT_RING_SLOTS,
    ) -> CircularNstepRuntimeState:
        if not power_of_two(telemetry_ring_slots):
            raise ValueError("telemetry ring slots must be a power of two")
        if base_generation < 1 or base_generation + resident_steps > 0x7FFF_FFFF:
            raise ValueError("generation range does not fit positive Int32")
        # nstep_control, in resident_protocol's CONTROL_* order.
        words = [
            resident_steps,
            mailbox_address,
            telemetry_ring_slots - 1,
            int(mailbox_address != 0),
            P.RECORD_WIDTH,
            0,
            0,
            refill_device_address,
            TOKEN_SLOT_MASK,
            base_generation,
            timeout_ns,
        ]
        return cls(
            control=torch.tensor(words, dtype=torch.int64, device=device),
            records=torch.full(
                (resident_steps * P.RECORD_WIDTH,),
                float("nan"),
                dtype=torch.float32,
                device=device,
            ),
            resident_steps=resident_steps,
        )

    def runtime_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.control, self.records

    def reset(self) -> None:
        self.control[P.CONTROL_STEP] = 0
        self.control[P.CONTROL_CHECKPOINT] = P.CHECKPOINT_NONE
        self.records.fill_(float("nan"))


class _PinnedCpuWindowStaging:
    """Reuse one pinned payload tuple for every producer-side refill.

    Packing still constructs an ordinary CPU ``TokenShardRing`` for the current
    row.  Only the page-locked copy destinations are retained: after the first
    row, their storage is overwritten in place once the preceding asynchronous
    H2D copies have completed.  The producer owns one instance, so this does not
    add another queued workload or weaken the two-slot backpressure protocol.
    """

    def __init__(self, *, geometry: Geometry) -> None:
        self.geometry = geometry
        self._buffers: tuple[torch.Tensor, ...] | None = None

    def pack(self, workload: TokenWindow) -> tuple[torch.Tensor, ...]:
        sources = TokenShardRing.from_workloads(
            (workload,),
            device=torch.device("cpu"),
            geometry=self.geometry,
        ).runtime_tensors()
        if self._buffers is None:
            self._buffers = tuple(tensor.pin_memory() for tensor in sources)
        else:
            for destination, source in zip(self._buffers, sources, strict=True):
                destination.copy_(source)
        return self._buffers


def _slot_views(ring: TokenShardRing, slot: int) -> tuple[torch.Tensor, ...]:
    return tuple(
        backing.narrow(0, slot * extent, extent)
        for backing, extent in zip(
            ring.runtime_tensors(), _slot_extents(ring.sequence), strict=True
        )
    )


class _WorkloadCursor:
    """Take exactly ``count`` consecutive windows, with one packing, from a one-pass source."""

    def __init__(self, workloads: Iterator[TokenWindow], *, count: int) -> None:
        self.count = count
        self._iterator = workloads
        self._packing: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        self._next_stream_index: int | None = None

    def take(self) -> TokenWindow:
        row = next(self._iterator)
        packing = (row.segment_extents, row.rotary_segment_extents)
        if self._packing is None:
            self._packing = packing
        elif packing != self._packing:
            raise ValueError(f"window packing changed: {packing!r} != {self._packing!r}")
        if self._next_stream_index is not None and row.stream_index != self._next_stream_index:
            raise ValueError(
                f"window stream {row.stream_index} does not follow {self._next_stream_index - 1}"
            )
        self._next_stream_index = row.stream_index + 1
        return row

    def take_initial_slots(self) -> tuple[TokenWindow, TokenWindow]:
        first = self.take()
        if self.count == 1:
            # The compiled ABI always exposes two equally sized HBM slots.  A
            # one-step invocation consumes only slot zero, so initialize the
            # physically required second slot with a deterministic duplicate
            # without advancing the logical workload cursor.
            return first, first
        return first, self.take()


GenerationGate = Callable[[int], None]


class CircularTokenRefillSession:
    """Own the two-slot device ring, producer thread, and mapped page.

    ``geometry`` is the bundle's: it sets the window size and the embedding-route
    capacity.  The step loop's control words also carry the telemetry mailbox,
    ``mailbox_address`` with ``telemetry_ring_slots`` slots.
    """

    def __init__(
        self,
        *,
        workloads: Iterator[TokenWindow],
        workload_count: int,
        device: torch.device,
        run_id: str,
        geometry: Geometry,
        base_generation: int,
        timeout_ns: int,
        generation_gate: GenerationGate,
        mailbox_address: int,
        telemetry_ring_slots: int,
        progress: Callable[[], object] | None = None,
    ) -> None:
        if not 1 <= workload_count <= geometry.resident_step_limit:
            raise ValueError(
                f"steps must be in [1, {geometry.resident_step_limit}], got {workload_count}"
            )
        source = _WorkloadCursor(workloads, count=workload_count)
        initial_rows = source.take_initial_slots()
        self.workload_count = workload_count
        self._source = source
        self.device = device
        self.device_index = device.index
        self.page = MappedRefillControl(run_id=run_id)
        try:
            device_pointer = self.page.register_cuda(device_index=self.device_index)
            self.ring = TokenShardRing.from_workloads(
                initial_rows,
                device=device,
                geometry=geometry,
            )
            self.nstep = CircularNstepRuntimeState.allocate(
                resident_steps=self.workload_count,
                device=device,
                refill_device_address=device_pointer,
                base_generation=base_generation,
                timeout_ns=timeout_ns,
                mailbox_address=mailbox_address,
                telemetry_ring_slots=telemetry_ring_slots,
            )
        except BaseException:
            self.page.close()
            raise
        self.base_generation = base_generation
        self.timeout_seconds = timeout_ns / 1_000_000_000
        self.generation_gate = generation_gate
        self.progress = progress
        self.copy_stream = torch.cuda.Stream(device=device)
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def bind(self, state: object) -> None:
        state.nstep = self.nstep
        state.token_shards = self.ring

    def _generation(self, iteration: int) -> int:
        return self.base_generation + iteration

    def preload(self) -> None:
        """Publish only constructor-filled slots that this invocation consumes."""

        torch.cuda.synchronize(self.device)
        for iteration in range(min(TOKEN_SLOTS, self.workload_count)):
            generation = self._generation(iteration)
            self.generation_gate(generation)
            self.page.publish_ready(slot=iteration, generation=generation)

    def _run(self) -> None:
        try:
            torch.cuda.set_device(self.device_index)
            staging = _PinnedCpuWindowStaging(geometry=self.ring.geometry)
            for iteration in range(TOKEN_SLOTS, self.workload_count):
                workload = self._source.take()
                packed = staging.pack(workload)
                slot = iteration & TOKEN_SLOT_MASK
                prior_generation = self._generation(iteration - TOKEN_SLOTS)
                self.page.wait_free(
                    slot=slot,
                    generation=prior_generation,
                    timeout_seconds=self.timeout_seconds,
                    progress=self.progress,
                )
                with torch.cuda.stream(self.copy_stream):
                    for destination, source in zip(
                        _slot_views(self.ring, slot), packed, strict=True
                    ):
                        destination.copy_(source, non_blocking=True)
                    copied = torch.cuda.Event(enable_timing=False, blocking=False)
                    copied.record(self.copy_stream)
                copied.synchronize()
                generation = self._generation(iteration)
                self.generation_gate(generation)
                self.page.publish_ready(slot=slot, generation=generation)
                # The copy event makes the sole pinned staging tuple reusable.
                # Drop the row before asking the source for its successor; the
                # staging tuple itself stays bounded and lives only in this
                # producer thread.
                del workload
        except BaseException as error:
            self._error = error
            self.page.request_abort()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("refill producer already started")
        self.preload()
        self._thread = threading.Thread(
            target=self._run,
            name="training-megakernel-token-refill",
            daemon=True,
        )
        self._thread.start()

    def finish(self) -> None:
        """Call after synchronizing the application stream."""

        if self._thread is None:
            raise RuntimeError("refill producer was not started")
        self._thread.join(timeout=self.timeout_seconds)
        if self._thread.is_alive():
            self.page.request_abort()
            raise TimeoutError("refill producer did not reach terminal state")
        if self._error is not None:
            raise RuntimeError("refill producer failed") from self._error

    def abort(self) -> None:
        self.page.request_abort()

    def close(self) -> bool:
        """Abort and join the producer before unregistering its mapped page."""

        if self._thread is not None and self._thread.is_alive():
            self.page.request_abort()
            self._thread.join(timeout=min(self.timeout_seconds, 5.0))
        if self._thread is not None and self._thread.is_alive():
            # Keep the page registered so the live thread/device can never
            # touch an unmapped address during teardown.
            return False
        self.page.close()
        return True


def gloo_generation_gate(group: object) -> GenerationGate:
    """Return a per-generation WORLD8 host rendezvous callback."""

    import torch.distributed as dist

    def gate(generation: int) -> None:
        observed: list[int | None] = [None] * contract.WORLD_SIZE
        dist.all_gather_object(observed, generation, group=group)
        if observed != [generation] * contract.WORLD_SIZE:
            raise RuntimeError(
                f"WORLD8 refill generations diverged: expected={generation}, "
                f"observed={observed}"
            )

    return gate


__all__ = (
    "TOKEN_SLOTS",
    "CircularNstepRuntimeState",
    "CircularTokenRefillSession",
    "MappedRefillControl",
    "gloo_generation_gate",
)
