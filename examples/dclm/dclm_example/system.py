"""Small GPU and process-group helpers used by the DCLM example."""

from __future__ import annotations

import ctypes
import gc

import torch
import torch.distributed as dist

from training_megakernel.host_memory import cudart


def clear_gpu_cache(device: torch.device) -> None:
    """Release inactive Torch allocations."""

    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def set_cuda_stack_limit(requested: int) -> None:
    """Set the CUDA per-thread stack limit a bundle requires."""

    status = int(cudart().cudaDeviceSetLimit(0, ctypes.c_size_t(requested)))
    if status:
        raise RuntimeError(f"cudaDeviceSetLimit failed with status {status}")


def initialize_world8(*, device: torch.device) -> tuple[int, object]:
    """Initialize NCCL, create the host control group, and prime collectives."""

    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    control_group = dist.new_group(backend="gloo")
    primer = torch.ones(1, device=device)
    dist.all_reduce(primer, group=dist.group.WORLD)
    torch.cuda.synchronize(device)
    observed = float(primer.item())
    expected = float(dist.get_world_size())
    if observed != expected:
        raise RuntimeError(f"NCCL primer returned {observed}, expected {expected}")
    del primer
    clear_gpu_cache(device)
    return rank, control_group


__all__ = (
    "clear_gpu_cache",
    "initialize_world8",
    "set_cuda_stack_limit",
)
