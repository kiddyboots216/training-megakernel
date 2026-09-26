"""The build's stand-ins for the program's symmetric-memory arenas.

At run time each arena is torch symmetric memory, mapped on every rank with a multicast address
that reaches all copies; the host runtime allocates them (``training_megakernel.distributed``).
The build compiles on one GPU with no peers, so ``allocate_arena`` returns a zeroed device
tensor whose own address stands in for the multicast address. The compile reads only shapes,
dtypes and pointer alignment from these tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SymmetricArena:
    """One rank's byte arena and the multicast address of all copies."""

    tensor: torch.Tensor
    multicast_base: int


def allocate_arena(nbytes: int, device: torch.device) -> SymmetricArena:
    """A zeroed byte arena and its stand-in multicast address."""

    tensor = torch.zeros(nbytes, dtype=torch.uint8, device=device)
    return SymmetricArena(tensor=tensor, multicast_base=tensor.data_ptr())
