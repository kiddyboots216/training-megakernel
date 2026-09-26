"""Periodic checkpoints under one checkpoint root.

Each checkpoint is a ``step-NNNNNNNN`` directory.  Its ``COMPLETE`` marker is
written last, after the manifest, and is the commit record.
"""

from __future__ import annotations

from pathlib import Path


def step_directory(root: Path, step: int) -> Path:
    return root / f"step-{step:08d}"


def latest_checkpoint(root: Path) -> tuple[int, Path]:
    """The highest-step committed checkpoint under ``root``."""

    complete = []
    for entry in root.glob("step-*"):
        digits = entry.name.removeprefix("step-")
        if (
            digits.isascii()
            and digits.isdigit()
            and entry == step_directory(root, int(digits))
            and (entry / "COMPLETE").is_file()
        ):
            complete.append((int(digits), entry))
    if not complete:
        raise FileNotFoundError(f"no complete step-NNNNNNNN checkpoint under {root}")
    return max(complete)
