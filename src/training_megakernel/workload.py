"""One GPU's packed token window for one step."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenWindow:
    """One rank-local packed token window."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    local_valid_tokens: int
    segment_extents: tuple[int, ...]
    rotary_segment_extents: tuple[int, ...]
    stream_index: int


__all__ = ("TokenWindow",)
