"""CUDA-free mailbox transport for resident N-step execution."""

from __future__ import annotations

import ctypes
import mmap
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from . import resident_protocol as P
from .host_memory import register_mapped, unregister_mapped

HEADER_BYTES = P.MAILBOX_HEADER_BYTES
SLOT_BYTES = P.MAILBOX_SLOT_BYTES
DEFAULT_RING_SLOTS = 4_096
HEADER_LOGGER_STATUS_OFFSET = 12
LOGGER_READY = 1


def power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def safe_run_id(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not result:
        raise ValueError("run id has no filesystem-safe characters")
    return result


class NstepMailbox:
    """Named POSIX mapping of ``size`` bytes, registered into one GPU's address space.

    The first ``HEADER_BYTES + ring_slots * SLOT_BYTES`` bytes are the telemetry
    header and ring the device publishes each step into.
    """

    def __init__(self, *, run_id: str, ring_slots: int, size: int) -> None:
        if not power_of_two(ring_slots):
            raise ValueError("ring slots must be a power of two")
        self.run_id = safe_run_id(run_id)
        self.path = self.path_for(self.run_id)
        self.ring_slots = ring_slots
        self.size = size
        self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            # A new file's pages read as zero.
            os.ftruncate(self._fd, self.size)
            self._mapping = mmap.mmap(
                self._fd,
                self.size,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        except BaseException:
            os.close(self._fd)
            self.path.unlink(missing_ok=True)
            raise
        self.address = ctypes.addressof(ctypes.c_char.from_buffer(self._mapping))
        self.device_address = 0
        self._logger: subprocess.Popen[str] | None = None
        self._closed = False

    @staticmethod
    def path_for(run_id: str) -> Path:
        """Return the deterministic POSIX mapping path for one runtime rank."""

        return Path("/dev/shm") / f"mk_nstep_{safe_run_id(run_id)}"

    def register_cuda(self, *, device_index: int) -> int:
        self.device_address = register_mapped(self.address, self.size, device_index)
        return self.device_address

    def start_logger(
        self,
        *,
        resident_steps: int,
        start_completed_step: int,
        output_dir: Path,
        poll_ms: int,
        wandb_mode: str,
        wandb_project: str | None,
        loss_shm_paths: Sequence[Path],
    ) -> None:
        """Start the CUDA-free logger; it prints each step and writes ``telemetry.jsonl``."""

        if self._logger is not None:
            raise RuntimeError("logger is already running")
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "training_megakernel.logger",
            "--shm-path",
            str(self.path),
            "--ring-slots",
            str(self.ring_slots),
            "--steps",
            str(resident_steps),
            "--start-step",
            str(start_completed_step),
            "--poll-ms",
            str(poll_ms),
            "--jsonl",
            str(output_dir / "telemetry.jsonl"),
            "--summary",
            str(output_dir / "logger-summary.json"),
            "--wandb-mode",
            wandb_mode,
            "--run-id",
            self.run_id,
        ]
        for path in loss_shm_paths:
            command.extend(("--loss-shm-path", str(path)))
        if wandb_project:
            command.extend(("--wandb-project", wandb_project))
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["PYTHONUNBUFFERED"] = "1"
        self._logger = subprocess.Popen(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            text=True,
        )

    def stop_logger(self, timeout_seconds: float = 10.0) -> None:
        """Boundedly stop a logger during fail-stop or mailbox teardown."""

        if self._logger is None or self._logger.poll() is not None:
            return
        self._logger.terminate()
        try:
            self._logger.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            self._logger.kill()
            self._logger.wait(timeout=timeout_seconds)

    def mark_final_and_wait(self, final_step: int, timeout_seconds: float = 120.0) -> None:
        if final_step < 1:
            raise ValueError("final step must be positive")
        self._mapping[4:8] = int(final_step).to_bytes(4, "little", signed=False)
        self._mapping.flush(0, HEADER_BYTES)
        deadline = time.monotonic() + timeout_seconds
        while int.from_bytes(self._mapping[8:12], "little", signed=False) < final_step:
            if self._logger is not None and self._logger.poll() is not None:
                raise RuntimeError(
                    f"telemetry logger exited early with status {self._logger.returncode}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("telemetry logger did not acknowledge the final step")
            time.sleep(0.02)
        if self._logger is not None:
            status = self._logger.wait(timeout=30)
            if status != 0:
                raise RuntimeError(f"telemetry logger failed with status {status}")

    def wait_logger_ready(self, timeout_seconds: float = 30.0) -> None:
        if self._logger is None:
            raise RuntimeError("logger has not been started")
        deadline = time.monotonic() + timeout_seconds
        while (
            int.from_bytes(
                self._mapping[
                    HEADER_LOGGER_STATUS_OFFSET : HEADER_LOGGER_STATUS_OFFSET + 4
                ],
                "little",
                signed=False,
            )
            != LOGGER_READY
        ):
            if self._logger.poll() is not None:
                raise RuntimeError(
                    f"telemetry logger exited during startup with status "
                    f"{self._logger.returncode}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("telemetry logger did not become ready")
            time.sleep(0.02)

    def close(self) -> None:
        if self._closed:
            return
        self.stop_logger()
        if self.device_address:
            unregister_mapped(self.address)
            self.device_address = 0
        try:
            self._mapping.close()
        finally:
            os.close(self._fd)
            self.path.unlink(missing_ok=True)
            self._closed = True
