"""Runtime adapter for in-launch resident checkpoints.

This module extends the existing telemetry mapping on every rank with two
chunk-sized checkpoint payload slots, registers the entire mapping as CUDA
mapped host memory, and runs the durable rank publisher on a background thread
while the application CUfunction is parked at each scheduled checkpoint.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch.distributed as dist

from training_megakernel import resident_protocol as P
from training_megakernel.checkpoint_protocol import Surface
from training_megakernel.nstep import NstepMailbox

from .checkpoint_history import step_directory
from .checkpoint_storage import (
    AtomicCheckpointHeader,
    AtomicCheckpointWord,
    RawRankPublisher,
    drain_coordinate_and_commit,
)


class CheckpointRankMailbox(NstepMailbox):
    """Per-rank telemetry mailbox plus two reusable checkpoint payload slots.

    ``NstepMailbox`` methods continue to expose telemetry and the rank-zero
    logger; ``RawRankPublisher`` drains the payload slots.
    """

    def __init__(self, *, run_id: str, rank: int, ring_slots: int) -> None:
        self.payload_offset = P.checkpoint_payload_offset(ring_slots)
        super().__init__(
            run_id=f"{run_id}-rank{rank}",
            ring_slots=ring_slots,
            size=self.payload_offset + P.CHECKPOINT_PAYLOAD_SLOTS * P.CHECKPOINT_CHUNK_BYTES,
        )
        self.checkpoint_header = AtomicCheckpointHeader(self._mapping)
        self.claim_generation = AtomicCheckpointWord(
            self._mapping, P.checkpoint_claim_offset(ring_slots)
        )

    def payload(self, nbytes: int, *, slot: int) -> memoryview:
        begin = self.payload_offset + slot * P.CHECKPOINT_CHUNK_BYTES
        return memoryview(self._mapping)[begin : begin + nbytes]


def create_checkpoint_control_group(*, timeout_seconds: float = 180.0) -> object:
    """Create the dedicated Gloo group used only by checkpoint publishers.

    Circular refill performs ordered collectives from a different host thread.
    Reusing its Gloo group would make the two collective sequences race.  All
    ranks must therefore call this once, on the main thread, before launch.
    """

    return dist.new_group(
        ranks=list(range(dist.get_world_size())),
        backend="gloo",
        timeout=timedelta(seconds=timeout_seconds),
    )


def checkpoint_cursor(
    invocation_cursor: Mapping[str, Any],
    *,
    step: int,
    source_cursor_resolver: Callable[[int], Mapping[str, Any]],
) -> dict[str, Any]:
    """Advance the invocation's start cursor to the checkpoint at global ``step``."""

    next_stream = int(invocation_cursor["next_stream"]) + (
        step - int(invocation_cursor["completed_steps"])
    )
    return {
        **invocation_cursor,
        "next_stream": next_stream,
        "completed_steps": step,
        "dclm_source_cursor": dict(source_cursor_resolver(next_stream)),
    }


class CheckpointService:
    """Publish the scheduled checkpoints of one resident CUfunction invocation.

    ``start()`` runs the host checkpoint path on one bounded background thread,
    leaving the caller free to enter and remain inside the application launch.
    Checkpoint ``s`` goes to ``ROOT/step-NNNNNNNN``.  The next request is not
    made visible until the prior WORLD manifest is durable and its final device
    generation has been acknowledged.

    ``control_group`` must be the dedicated Gloo group returned by
    ``create_checkpoint_control_group``, never the circular-refill group.
    """

    def __init__(
        self,
        *,
        mailbox: CheckpointRankMailbox,
        root: Path,
        rank: int,
        world_size: int,
        steps: tuple[int, ...],
        invocation_cursor: Mapping[str, Any],
        surfaces: tuple[Surface, ...],
        source_cursor_resolver: Callable[[int], Mapping[str, Any]],
        control_group: object,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.mailbox = mailbox
        self.root = root
        self.rank = rank
        self.world_size = world_size
        self.steps = steps
        self.invocation_cursor = dict(invocation_cursor)
        self.source_cursor_resolver = source_cursor_resolver
        self.surfaces = surfaces
        self.timeout_seconds = timeout_seconds
        self.control_group = control_group
        self._publisher: RawRankPublisher | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._done = threading.Event()
        self._abort_requested = threading.Event()
        self._control_lock = threading.Lock()
        self._manifests: list[dict[str, Any]] = []
        self._error: BaseException | None = None

    def _abort(self) -> None:
        # Serialize with request publication so request() can never clear an
        # abort that raced between the loop check and the mapped-header store.
        with self._control_lock:
            self._abort_requested.set()
            publisher = self._publisher
            if publisher is not None:
                publisher.abort()
            else:
                self.mailbox.checkpoint_header.store("host_abort", 1)

    def _run(self) -> None:
        try:
            for index, step in enumerate(self.steps):
                if self._abort_requested.is_set():
                    raise RuntimeError("checkpoint service was aborted")
                publisher = RawRankPublisher(
                    self.mailbox,
                    step_directory(self.root, step),
                    rank=self.rank,
                    world_size=self.world_size,
                    step=step,
                    data_cursor=checkpoint_cursor(
                        self.invocation_cursor,
                        step=step,
                        source_cursor_resolver=self.source_cursor_resolver,
                    ),
                    surfaces=self.surfaces,
                    timeout_seconds=self.timeout_seconds,
                )
                with self._control_lock:
                    if self._abort_requested.is_set():
                        raise RuntimeError("checkpoint service was aborted")
                    self._publisher = publisher
                    publisher.request()
                if index == 0:
                    # start() returns only after the first request is visible.
                    self._ready.set()
                self._manifests.append(
                    drain_coordinate_and_commit(
                        publisher,
                        control_group=self.control_group,
                    )
                )
        except BaseException as error:
            self._error = error
            self._abort()
        finally:
            self._ready.set()
            self._done.set()

    def start(self) -> CheckpointService:
        self._thread = threading.Thread(
            target=self._run,
            name=f"training-megakernel-checkpoint-rank-{self.rank}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=min(self.timeout_seconds, 10.0)):
            self._abort()
            raise TimeoutError("checkpoint service did not publish its request")
        if self._error is not None:
            raise RuntimeError("checkpoint service failed during startup") from self._error
        return self

    def wait(self, timeout_seconds: float) -> tuple[dict[str, Any], ...]:
        assert self._thread is not None
        if not self._done.wait(timeout=timeout_seconds):
            self._abort()
            self._thread.join(timeout=min(self.timeout_seconds, 5.0))
            raise TimeoutError("checkpoint service did not reach terminal state")
        self._thread.join(timeout=0)
        if self._error is not None:
            raise RuntimeError("checkpoint service failed") from self._error
        return tuple(self._manifests)

    def abort(self) -> None:
        self._abort()

    def close(self) -> bool:
        """Fail-stop the controller and wait a bounded time for its worker."""

        thread = self._thread
        if thread is not None and thread.is_alive():
            self._abort()
            thread.join(timeout=min(self.timeout_seconds, 5.0))
        return thread is None or not thread.is_alive()
