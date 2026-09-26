"""Host half of the bounded in-launch checkpoint protocol.

The device owns snapshot production and the ready generation.  This module
owns durable rank-shard publication, WORLD8 agreement, the final ack, and a
bounded-memory restore path.  The stored tensors' shapes follow the bundle's
depth: callers pass ``compact_stored_surfaces(O(D))``.
"""

from __future__ import annotations

import concurrent.futures
import ctypes
import ctypes.util
import json
import mmap
import os
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from training_megakernel import contract
from training_megakernel import resident_protocol as P
from training_megakernel.checkpoint_protocol import (
    Surface,
    chunks_for_surfaces,
    compact_derived_surfaces,
    compact_stored_surfaces,
    rank_file_bytes,
)
from training_megakernel.runtime import CHECKPOINT_SCHEMA

_ACQUIRE = 2
_RELEASE = 3
_HEADER_OFFSET = {
    "host_request_step": P.CHECKPOINT_HOST_REQUEST_STEP,
    "host_ack_generation": P.CHECKPOINT_HOST_ACK_GENERATION,
    "host_abort": P.CHECKPOINT_HOST_ABORT,
    "device_ready_generation": P.CHECKPOINT_DEVICE_READY_GENERATION,
    "device_step": P.CHECKPOINT_DEVICE_STEP,
    "device_surface_index": P.CHECKPOINT_DEVICE_SURFACE,
    "device_chunk_index": P.CHECKPOINT_DEVICE_CHUNK,
    "device_chunk_bytes": P.CHECKPOINT_DEVICE_CHUNK_BYTES,
    "device_status": P.CHECKPOINT_DEVICE_STATUS,
    "device_done_generation": P.CHECKPOINT_DEVICE_DONE_GENERATION,
}
DEFAULT_CHECKPOINT_PREFLUSH_BYTES = 2 * 1024**3


def _libatomic() -> ctypes.CDLL:
    path = ctypes.util.find_library("atomic")
    if path is None:
        raise RuntimeError("libatomic is required for mapped checkpoint control words")
    library = ctypes.CDLL(path)
    load = getattr(library, "__atomic_load_4")
    load.argtypes = (ctypes.c_void_p, ctypes.c_int)
    load.restype = ctypes.c_uint32
    store = getattr(library, "__atomic_store_4")
    store.argtypes = (ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int)
    store.restype = None
    return library


_ATOMIC = _libatomic()


class AtomicCheckpointHeader:
    """Acquire/release access to the single-writer mapped header words."""

    def __init__(self, buffer: mmap.mmap) -> None:
        self._buffer = buffer
        self._base = ctypes.addressof(ctypes.c_char.from_buffer(buffer))

    def load(self, name: str) -> int:
        return int(
            getattr(_ATOMIC, "__atomic_load_4")(
                self._base + _HEADER_OFFSET[name], _ACQUIRE
            )
        )

    def load_latest_completed_step(self) -> int:
        """Acquire the device's existing post-optimizer telemetry step."""

        return int(
            getattr(_ATOMIC, "__atomic_load_4")(
                self._base + P.MAILBOX_LATEST_STEP, _ACQUIRE
            )
        )

    def store(self, name: str, value: int) -> None:
        if not 0 <= value < 2**32:
            raise ValueError(f"checkpoint word is outside uint32: {name}={value}")
        getattr(_ATOMIC, "__atomic_store_4")(
            self._base + _HEADER_OFFSET[name], value, _RELEASE
        )


class AtomicCheckpointWord:
    """Acquire/release access to one uint32 outside the live header."""

    def __init__(self, buffer: mmap.mmap, offset: int) -> None:
        self._buffer = buffer
        self._address = ctypes.addressof(ctypes.c_char.from_buffer(buffer)) + offset

    def load(self) -> int:
        return int(getattr(_ATOMIC, "__atomic_load_4")(self._address, _ACQUIRE))

    def store(self, value: int) -> None:
        if not 0 <= value < 2**32:
            raise ValueError(f"checkpoint word is outside uint32: {value}")
        getattr(_ATOMIC, "__atomic_store_4")(self._address, value, _RELEASE)


def _rank_name(rank: int) -> str:
    return f"rank-{rank:05d}.raw"


def _surface_records(surfaces: tuple[Surface, ...]) -> list[dict[str, Any]]:
    records = [asdict(row) for row in surfaces]
    for record in records:
        record["shape"] = list(record["shape"])
    return records


def _write_all_at(descriptor: int, data: memoryview, offset: int) -> None:
    written = 0
    while written < len(data):
        count = os.pwrite(descriptor, data[written:], offset + written)
        if count <= 0:
            raise OSError("checkpoint pwrite made no progress")
        written += count


class _BoundedFdatasyncPreflusher:
    """Keep checkpoint writeback within two intervals of the writes.

    At most one ``fdatasync`` may be outstanding; when the next interval is due
    the writer waits for it, so dirty data stays bounded and writeback overlaps
    the device's chunks instead of piling up for the final fdatasync.  A
    completed background failure is raised before another slot-reuse ack, and
    ``finish`` joins the last request before the publisher's final fdatasync.
    """

    def __init__(
        self,
        descriptor: int,
        executor: concurrent.futures.ThreadPoolExecutor,
        *,
        interval_bytes: int,
    ) -> None:
        self.descriptor = descriptor
        self.executor = executor
        self.interval_bytes = interval_bytes
        self.next_boundary = interval_bytes
        self.future: concurrent.futures.Future[None] | None = None

    def _reap_if_done(self) -> None:
        future = self.future
        if future is None or not future.done():
            return
        self.future = None
        future.result()

    def schedule_if_due(self, committed_payload_bytes: int) -> bool:
        self._reap_if_done()
        if self.interval_bytes == 0:
            return False
        if committed_payload_bytes < self.next_boundary:
            return False
        self.finish()
        while self.next_boundary <= committed_payload_bytes:
            self.next_boundary += self.interval_bytes
        self.future = self.executor.submit(os.fdatasync, self.descriptor)
        return True

    def raise_if_failed_before_ack(self) -> None:
        self._reap_if_done()

    def finish(self) -> None:
        future = self.future
        self.future = None
        if future is not None:
            future.result()


def _read_all_at(descriptor: int, target: memoryview, offset: int) -> None:
    read = 0
    while read < len(target):
        count = os.preadv(descriptor, [target[read:]], offset + read)
        if count <= 0:
            raise EOFError("raw checkpoint shard is truncated")
        read += count


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        payload = value.encode()
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            if count <= 0:
                raise OSError("atomic checkpoint metadata write made no progress")
            offset += count
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    os.replace(temporary, path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class RawRankPublisher:
    """Drain one device-produced checkpoint while withholding its final ack.

    The device fills two alternating payload slots while the resident grid stays
    live.  ``host_claim_generation`` is published only after the slot's write
    has been submitted.  ``host_ack_generation`` is published only after that
    write has returned, so the device can fill the alternate slot but cannot
    overwrite a slot the write still reads.
    """

    def __init__(
        self,
        mailbox: Any,
        directory: str | Path,
        *,
        rank: int,
        world_size: int,
        step: int,
        data_cursor: Mapping[str, Any],
        surfaces: tuple[Surface, ...],
        preflush_interval_bytes: int = DEFAULT_CHECKPOINT_PREFLUSH_BYTES,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.mailbox = mailbox
        self.header = mailbox.checkpoint_header
        self.claim_generation = mailbox.claim_generation
        self.directory = Path(directory)
        self.rank = rank
        self.world_size = world_size
        self.step = step
        self.data_cursor = dict(data_cursor)
        self.surfaces = surfaces
        self.chunks = chunks_for_surfaces(surfaces)
        self.preflush_interval_bytes = preflush_interval_bytes
        self.timeout_seconds = timeout_seconds
        self.generation_start = self.header.load("host_ack_generation") + 1
        self.generation_end = self.generation_start + len(self.chunks) - 1

    def request(self) -> None:
        header = self.header
        if self.claim_generation.load() != self.generation_start - 1:
            raise RuntimeError(
                "double-buffered checkpoint starts with claim/ack disagreement"
            )
        if header.load("device_status") != 0:
            raise RuntimeError("device checkpoint status is already failed")
        if header.load("device_ready_generation") >= self.generation_start:
            raise RuntimeError("device published a checkpoint before the host request")
        header.store("host_abort", 0)
        header.store("host_request_step", self.step)

    def _wait_ready(self, generation: int) -> None:
        # A periodic request is intentionally visible long before the device
        # reaches its optimizer boundary.  Do not charge that useful training
        # interval against the payload-drain timeout.  The resident body
        # release-publishes telemetry word zero before reading the checkpoint
        # request at the same boundary, so it is the arrival signal for the
        # first generation.  Every later generation retains the original
        # bounded wait beginning immediately after the prior chunk is acked.
        deadline = (
            None
            if generation == self.generation_start
            else time.monotonic() + self.timeout_seconds
        )
        header = self.header
        while True:
            status = header.load("device_status")
            observed = header.load("device_ready_generation")
            if status != 0:
                raise RuntimeError(f"device checkpoint producer failed with status {status}")
            if header.load("host_abort") != 0:
                raise RuntimeError("host aborted checkpoint publication")
            if observed == generation:
                return
            if observed > generation:
                raise RuntimeError("device checkpoint generation skipped ahead")
            now = time.monotonic()
            if (
                deadline is None
                and header.load_latest_completed_step() >= self.step
            ):
                deadline = now + self.timeout_seconds
            if deadline is not None and now >= deadline:
                raise TimeoutError(f"checkpoint generation {generation} timed out")
            time.sleep(0.0001)

    def drain(self) -> None:
        """Write this rank's shard and make it durable, acking every chunk but the last."""

        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "COMPLETE").exists():
            raise FileExistsError(f"checkpoint is already complete: {self.directory}")
        destination = self.directory / _rank_name(self.rank)
        temporary = self.directory / f".{destination.name}.tmp.{os.getpid()}"
        descriptor = os.open(
            temporary, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
        )
        committed_payload_bytes = 0
        final_generation = self.generation_end
        header = self.header
        try:
            os.ftruncate(descriptor, rank_file_bytes(self.surfaces))
            with (
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix=f"training-megakernel-checkpoint-rank-{self.rank}",
                ) as executor,
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix=(
                        f"training-megakernel-checkpoint-preflush-rank-{self.rank}"
                    ),
                ) as preflush_executor,
            ):
                preflusher = _BoundedFdatasyncPreflusher(
                    descriptor,
                    preflush_executor,
                    interval_bytes=self.preflush_interval_bytes,
                )
                for index, chunk in enumerate(self.chunks):
                    generation = self.generation_start + index
                    self._wait_ready(generation)
                    expected_metadata = (
                        self.step,
                        chunk.surface_index,
                        chunk.chunk_index,
                        chunk.nbytes,
                    )
                    observed_metadata = (
                        header.load("device_step"),
                        header.load("device_surface_index"),
                        header.load("device_chunk_index"),
                        header.load("device_chunk_bytes"),
                    )
                    if observed_metadata != expected_metadata:
                        raise RuntimeError(
                            "device checkpoint metadata differs from the host plan: "
                            f"expected={expected_metadata} observed={observed_metadata}"
                        )
                    slot = (generation - 1) & 1
                    payload = self.mailbox.payload(chunk.nbytes, slot=slot)
                    try:
                        write_future = executor.submit(
                            _write_all_at,
                            descriptor,
                            payload,
                            chunk.file_offset,
                        )
                        # The submitted write now owns this immutable slot.
                        self.claim_generation.store(generation)
                        concurrent.futures.wait((write_future,))
                        # A failure withholds ack and therefore forbids slot reuse.
                        write_future.result()
                    finally:
                        payload.release()
                    committed_payload_bytes += chunk.nbytes
                    preflusher.raise_if_failed_before_ack()
                    if generation != final_generation:
                        preflusher.schedule_if_due(committed_payload_bytes)
                        preflusher.raise_if_failed_before_ack()
                        header.store("host_ack_generation", generation)
                preflusher.finish()
            if header.load("device_done_generation") != final_generation:
                raise RuntimeError("device did not mark the final checkpoint generation")
            os.fdatasync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.replace(temporary, destination)
            _fsync_directory(self.directory)
        except BaseException:
            header.store("host_abort", 1)
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
            raise

    def release(self) -> None:
        """Acknowledge the final generation, which lets the resident grid continue."""

        self.header.store("host_request_step", 0)
        self.header.store("host_ack_generation", self.generation_end)

    def abort(self) -> None:
        self.header.store("host_abort", 1)


def write_manifest(
    directory: Path,
    *,
    step: int,
    data_cursor: Mapping[str, Any],
    surfaces: tuple[Surface, ...],
    world_size: int,
) -> dict[str, Any]:
    """Durably write the manifest, then the COMPLETE marker that commits it."""

    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "world_size": world_size,
        "completed_steps": step,
        "data_cursor": dict(data_cursor),
        "rank_file_bytes": rank_file_bytes(surfaces),
        "stored_surfaces": _surface_records(surfaces),
    }
    text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_text(directory / "manifest.json", text)
    _atomic_text(directory / "COMPLETE", CHECKPOINT_SCHEMA + "\n")
    _fsync_directory(directory)
    _fsync_directory(directory.parent)
    return manifest


def drain_coordinate_and_commit(
    publisher: RawRankPublisher,
    *,
    control_group: object,
) -> dict[str, Any]:
    """Collectively publish, then release every parked resident grid.

    Every rank makes its shard durable and reports over Gloo; rank 0 commits
    the manifest only if all succeeded and broadcasts the outcome, and only
    then does any rank acknowledge the final generation.
    """

    local_error = None
    try:
        publisher.drain()
    except BaseException as error:  # every rank must still enter Gloo agreement
        local_error = f"{type(error).__name__}: {error}"
        publisher.abort()
    errors: list[str | None] = [None] * publisher.world_size
    dist.all_gather_object(errors, local_error, group=control_group)

    outcome: list[dict[str, Any] | None] = [None]
    if publisher.rank == 0:
        try:
            failed = [
                f"rank {rank}: {error}" for rank, error in enumerate(errors) if error
            ]
            if failed:
                raise RuntimeError("; ".join(failed))
            manifest = write_manifest(
                publisher.directory,
                step=publisher.step,
                data_cursor=publisher.data_cursor,
                surfaces=publisher.surfaces,
                world_size=publisher.world_size,
            )
            outcome[0] = {"manifest": manifest, "error": None}
        except BaseException as error:
            outcome[0] = {
                "manifest": None,
                "error": f"{type(error).__name__}: {error}",
            }
    dist.broadcast_object_list(outcome, src=0, group=control_group)
    result = outcome[0]
    if result is None or result["error"] is not None:
        publisher.abort()
        reason = None if result is None else result["error"]
        raise RuntimeError(f"WORLD8 checkpoint publication failed: {reason}")
    publisher.release()
    return result["manifest"]


def load_checkpoint_manifest(
    directory: Path,
    *,
    step: int,
    surfaces: tuple[Surface, ...],
) -> dict[str, Any]:
    """Read a committed manifest and require this bundle's stored tensors."""

    try:
        manifest = json.loads((directory / "manifest.json").read_text())
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read the checkpoint manifest in {directory}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != CHECKPOINT_SCHEMA
        or manifest.get("world_size") != contract.WORLD_SIZE
        or manifest.get("completed_steps") != step
        or not isinstance(manifest.get("data_cursor"), dict)
    ):
        raise RuntimeError(f"checkpoint manifest is not a WORLD8 step-{step} manifest")
    if manifest.get("stored_surfaces") != _surface_records(surfaces):
        raise RuntimeError(
            "checkpoint's stored tensors differ from this bundle's, whose optimizer "
            f"owns {surfaces[0].shape[0]} elements per GPU (a checkpoint of another "
            "depth?)"
        )
    return manifest


def load_raw_checkpoint(
    state: Any,
    directory: Path,
    *,
    rank: int,
    completed_steps: int,
    timeout_ns: int = 60_000_000_000,
) -> None:
    """Reconstruct restartable state, then stream this rank's shard in place.

    The checkpoint's stored tensors must have the shapes of ``state``, which was
    allocated for the bundle that resumes it.
    """

    optimizer = state.optimizer
    surfaces = compact_stored_surfaces(state.geometry.optimizer_elements)
    path = directory / _rank_name(rank)
    if path.stat().st_size != rank_file_bytes(surfaces):
        raise RuntimeError(f"raw checkpoint shard is truncated: {path}")
    state.prepare(timeout_ns=timeout_ns)
    chunk_bytes = P.CHECKPOINT_CHUNK_BYTES
    use_pinned = optimizer.parameter.device.type == "cuda"
    host_buffer = torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=use_pinned)
    host_view = memoryview(host_buffer.numpy())
    descriptor = os.open(path, os.O_RDONLY)
    try:
        with torch.no_grad():
            for chunk in chunks_for_surfaces(surfaces, chunk_bytes):
                _read_all_at(descriptor, host_view[: chunk.nbytes], chunk.file_offset)
                target = getattr(optimizer, chunk.surface_name).view(-1)
                element_bytes = target.element_size()
                begin = chunk.source_byte_offset // element_bytes
                count = chunk.nbytes // element_bytes
                source = host_buffer[: chunk.nbytes].view(target.dtype)
                target[begin : begin + count].copy_(source, non_blocking=False)
    finally:
        os.close(descriptor)
    with torch.no_grad():
        for surface in compact_derived_surfaces(state.geometry.optimizer_elements):
            # Tensor.copy_ performs the FP32 -> BF16 RNE conversion while
            # preserving the runtime-owned tensor object and storage.
            getattr(optimizer, surface.name).copy_(
                getattr(optimizer, surface.source), non_blocking=False
            )
        optimizer.completed_steps.fill_(completed_steps)
        optimizer.gradient.zero_()
        optimizer.final_gradient.zero_()
        optimizer.outputs.zero_()
        optimizer.status.zero_()
        state.model.attention.phase_counter.zero_()
        state.nstep.reset()
    if optimizer.parameter.device.type == "cuda":
        torch.cuda.synchronize(optimizer.parameter.device)
