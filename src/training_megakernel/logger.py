"""CUDA-free POSIX-shm reader and W&B writer for resident execution."""

from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import signal
import struct
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import resident_protocol as P

HEADER_BYTES = P.MAILBOX_HEADER_BYTES
SLOT_BYTES = P.MAILBOX_SLOT_BYTES
SLOT = struct.Struct("<IIfffIi")
LOGGER_STATUS_OFFSET = 12
LOGGER_READY = 1
WORLD_SIZE = 8
LOSS_REDUCTION = (
    "sum_of_world8_rank_local_cross_entropy_contributions_"
    "already_normalized_by_world8_global_valid_tokens"
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _u32(mapping: mmap.mmap, offset: int) -> int:
    return int.from_bytes(mapping[offset : offset + 4], "little", signed=False)


def _open_mapping(
    path: Path,
    expected_bytes: int,
    *,
    timeout_seconds: float = 25.0,
) -> mmap.mmap:
    """Open one peer mapping, allowing ranks to create their files concurrently."""

    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            descriptor = os.open(path, os.O_RDWR)
            break
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"telemetry peer mapping did not appear: {path}"
                ) from None
            time.sleep(0.01)
    try:
        return mmap.mmap(
            descriptor,
            expected_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
    finally:
        os.close(descriptor)


def _stable_step_record(
    mapping: mmap.mmap,
    *,
    step: int,
    ring_slots: int,
) -> dict[str, float | int] | None:
    """Acquire one complete device-published slot or report that it is not ready."""

    offset = HEADER_BYTES + (step & (ring_slots - 1)) * SLOT_BYTES
    before = _u32(mapping, offset)
    latest = _u32(mapping, 0)
    if before != step:
        if latest >= step + ring_slots:
            raise RuntimeError(
                f"telemetry step {step} was overwritten before aggregation; "
                f"latest={latest}"
            )
        return None
    seq, observed_step, loss, norm, clip, completed, optimizer_status = SLOT.unpack_from(
        mapping, offset
    )
    after = _u32(mapping, offset)
    if before != after:
        return None
    if seq != step or observed_step != step or completed != step:
        raise RuntimeError(
            "stable telemetry slot identity mismatch: "
            f"requested={step}, seq={seq}, step={observed_step}, "
            f"completed={completed}"
        )
    return {
        "step": observed_step,
        "loss": loss,
        "global_norm": norm,
        "clip": clip,
        "optimizer_completed_steps": completed,
        "optimizer_status_completed_epoch": optimizer_status,
    }


def _all_rank_step_records(
    mappings: list[mmap.mmap],
    *,
    step: int,
    ring_slots: int,
) -> list[dict[str, float | int]] | None:
    rows: list[dict[str, float | int]] = []
    for mapping in mappings:
        row = _stable_step_record(mapping, step=step, ring_slots=ring_slots)
        if row is None:
            return None
        rows.append(row)
    return rows


def _telemetry_row(
    records: list[dict[str, float | int]],
    *,
    host_receive_time_ns: int,
) -> dict[str, Any]:
    """Build a row without mislabeling one rank's contribution as global loss."""

    rank0 = records[0]
    losses = [float(record["loss"]) for record in records]
    row: dict[str, Any] = {
        "schema": "training_megakernel_telemetry_row_v1",
        "step": int(rank0["step"]),
        "loss_local_rank0": losses[0],
        "loss_local_contributions": losses,
        "loss_world_size": len(records),
        "loss_exact_all_rank_sum": len(records) == WORLD_SIZE,
        "global_norm": float(rank0["global_norm"]),
        "clip": float(rank0["clip"]),
        "optimizer_completed_steps": int(rank0["optimizer_completed_steps"]),
        "optimizer_status_completed_epoch": int(
            rank0["optimizer_status_completed_epoch"]
        ),
        "host_receive_time_ns": host_receive_time_ns,
    }
    if len(records) == WORLD_SIZE:
        loss = math.fsum(losses)
        row.update(
            {
                "loss": loss,
                "loss_world8_total_mean": loss,
                "loss_reduction": LOSS_REDUCTION,
            }
        )
    return row


def _wandb_metrics(row: dict[str, Any]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "train/loss_local_rank0": float(row["loss_local_rank0"]),
        "train/global_norm": float(row["global_norm"]),
        "train/clip": float(row["clip"]),
        "train/optimizer_completed_steps": int(row["optimizer_completed_steps"]),
        "telemetry/host_receive_time_ns": int(row["host_receive_time_ns"]),
    }
    if row.get("loss_exact_all_rank_sum") is True:
        metrics["train/loss"] = float(row["loss_world8_total_mean"])
    return metrics


def run(
    args: argparse.Namespace,
    *,
    stop_signal: Callable[[], int | None] | None = None,
) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("logger requires CUDA_VISIBLE_DEVICES='' ")
    # One-step resident images are useful isolation diagnostics.  The release
    # claim still requires two steps, but the logger itself only needs a
    # positive, image-matched step count.
    if (
        args.steps < 1
        or args.start_step < 0
        or args.start_step + args.steps >= 2**32
        or args.poll_ms < 1
    ):
        raise ValueError("invalid resident-step logger bounds")
    final_expected_step = args.start_step + args.steps
    expected_bytes = HEADER_BYTES + args.ring_slots * SLOT_BYTES
    loss_paths = list(args.loss_shm_path or (args.shm_path,))
    if loss_paths[0] != args.shm_path:
        raise ValueError("the first loss mailbox must be the rank-zero control mailbox")
    if len(set(loss_paths)) != len(loss_paths):
        raise ValueError("loss mailbox paths must be distinct")
    mapping = _open_mapping(args.shm_path, expected_bytes)
    loss_mappings = [mapping]
    try:
        loss_mappings.extend(
            _open_mapping(path, expected_bytes) for path in loss_paths[1:]
        )
    except BaseException:
        for peer_mapping in reversed(loss_mappings):
            peer_mapping.close()
        raise

    try:
        wandb_run = None
        if args.wandb_mode != "disabled":
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.run_id,
                id=args.run_id,
                mode=args.wandb_mode,
                dir=args.jsonl.parent,
                resume="allow",
                config={
                    "execution_form": "one cooperative application CUfunction per rank",
                    "resident_optimizer_steps": args.steps,
                    "invocation_start_completed_step": args.start_step,
                    "invocation_final_completed_step": final_expected_step,
                    "telemetry_transport": "mapped POSIX shm system-scope stores",
                    "loss_telemetry": (
                        "exact WORLD8 total mean"
                        if len(loss_mappings) == WORLD_SIZE
                        else "rank-local contribution only"
                    ),
                    "loss_world_size": len(loss_mappings),
                    "logger_cuda_visible_devices": "",
                },
            )

        accepted: list[dict[str, Any]] = []
        next_step = args.start_step + 1
        started_ns = time.time_ns()
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        terminal_reason = "complete"

        with args.jsonl.open("w", buffering=1) as stream:
            mapping[LOGGER_STATUS_OFFSET : LOGGER_STATUS_OFFSET + 4] = (
                LOGGER_READY.to_bytes(4, "little", signed=False)
            )
            mapping.flush(0, HEADER_BYTES)

            while True:
                observed_signal = stop_signal() if stop_signal is not None else None
                if observed_signal is not None:
                    terminal_reason = f"signal_{observed_signal}"
                    break
                latest = _u32(mapping, 0)
                while next_step <= min(latest, final_expected_step):
                    records = _all_rank_step_records(
                        loss_mappings,
                        step=next_step,
                        ring_slots=args.ring_slots,
                    )
                    if records is None:
                        break
                    row = _telemetry_row(
                        records,
                        host_receive_time_ns=time.time_ns(),
                    )
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    accepted.append(row)
                    print(
                        f"step {row['step']} loss {row.get('loss', row['loss_local_rank0']):.6f} "
                        f"grad_norm {row['global_norm']:.6f}",
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            _wandb_metrics(row),
                            step=int(row["step"]),
                            commit=True,
                        )
                    next_step += 1
                final_step = _u32(mapping, 4)
                if final_step > final_expected_step:
                    raise AssertionError(
                        "logger observed final step "
                        f"{final_step} beyond requested {final_expected_step}"
                    )
                if (
                    final_step == final_expected_step
                    and latest >= final_step
                    and next_step > final_step
                ):
                    break
                time.sleep(args.poll_ms / 1_000.0)

        final_step = _u32(mapping, 4)
        passed = (
            terminal_reason == "complete"
            and final_step == final_expected_step
            and len(accepted) == args.steps
            and [row["step"] for row in accepted]
            == list(range(args.start_step + 1, final_expected_step + 1))
            and all(row["optimizer_completed_steps"] == row["step"] for row in accepted)
        )
        if wandb_run is not None:
            wandb_run.summary["telemetry/all_steps_logged"] = passed
            wandb_run.summary["telemetry/logged_steps"] = len(accepted)
            wandb_run.summary["telemetry/loss_exact_all_rank_sum"] = (
                len(loss_mappings) == WORLD_SIZE
            )
            wandb_run.finish(exit_code=0 if passed else 1)
        mapping[8:12] = int(final_step).to_bytes(4, "little", signed=False)
        mapping.flush(0, HEADER_BYTES)
        summary = {
            "schema": "training_megakernel_telemetry_summary_v1",
            "pass": passed,
            "steps_requested": args.steps,
            "start_completed_step": args.start_step,
            "final_completed_step": final_expected_step,
            "steps_logged": len(accepted),
            "step_ids": [row["step"] for row in accepted],
            "optimizer_completed_steps": [row["optimizer_completed_steps"] for row in accepted],
            "loss_world_size": len(loss_mappings),
            "loss_exact_all_rank_sum": len(loss_mappings) == WORLD_SIZE,
            "loss_reduction": LOSS_REDUCTION if len(loss_mappings) == WORLD_SIZE else None,
            "loss_shm_paths": [str(path) for path in loss_paths],
            "mailbox_latest_observed": _u32(mapping, 0),
            "mailbox_final_step_observed": final_step,
            "started_ns": started_ns,
            "finished_ns": time.time_ns(),
            "wandb_mode": args.wandb_mode,
            "wandb_run_id": getattr(wandb_run, "id", None),
            "wandb_run_url": getattr(wandb_run, "url", None),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jsonl": str(args.jsonl),
            "terminal_reason": terminal_reason,
        }
        _write_json(args.summary, summary)
        if not passed:
            raise AssertionError(json.dumps(summary, sort_keys=True))
        return summary
    finally:
        for peer_mapping in reversed(loss_mappings):
            peer_mapping.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--shm-path", type=Path, required=True)
    result.add_argument(
        "--loss-shm-path",
        type=Path,
        action="append",
        help=(
            "ordered rank mailbox used for exact loss aggregation; repeat for "
            "ranks 0..7, or omit to retain local-only telemetry"
        ),
    )
    result.add_argument("--ring-slots", type=int, required=True)
    result.add_argument("--steps", type=int, required=True)
    result.add_argument("--start-step", type=int, default=0)
    result.add_argument("--poll-ms", type=int, default=20)
    result.add_argument("--jsonl", type=Path, required=True)
    result.add_argument("--summary", type=Path, required=True)
    result.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled"
    )
    result.add_argument("--wandb-project")
    result.add_argument("--run-id", required=True)
    return result


def main() -> None:
    observed_signal: int | None = None

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal observed_signal
        observed_signal = signum

    prior_term = signal.signal(signal.SIGTERM, request_stop)
    prior_interrupt = signal.signal(signal.SIGINT, request_stop)
    try:
        run(parser().parse_args(), stop_signal=lambda: observed_signal)
    finally:
        signal.signal(signal.SIGTERM, prior_term)
        signal.signal(signal.SIGINT, prior_interrupt)


if __name__ == "__main__":
    main()
