"""Pointer-independent launch ABI argument layout."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def _storage_key(tensor: torch.Tensor) -> tuple[str, int]:
    if tensor.device.type == "meta":
        return "meta", int(tensor.untyped_storage()._cdata)
    try:
        return tensor.device.type, int(tensor.untyped_storage().data_ptr())
    except (RuntimeError, AttributeError):
        return tensor.device.type, id(tensor)


def argument_rows(values: Iterable[tuple[str, object]]) -> list[dict[str, object]]:
    storage_ids: dict[tuple[str, int], int] = {}
    rows: list[dict[str, object]] = []

    def visit(name: str, value: object) -> None:
        if isinstance(value, torch.Tensor):
            storage_id = storage_ids.setdefault(_storage_key(value), len(storage_ids))
            rows.append(
                {
                    "name": name,
                    "kind": "tensor",
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "stride": list(value.stride()),
                    "storage_offset_bytes": int(value.storage_offset()) * value.element_size(),
                    "storage_group": storage_id,
                    "device_type": value.device.type,
                    "requires_grad": bool(value.requires_grad),
                    "assumed_alignment_bytes": 16,
                    "descriptor_class": "fully_dynamic_tvm_ffi_tensor",
                }
            )
        elif isinstance(value, (tuple, list)):
            rows.append({"name": name, "kind": type(value).__name__, "length": len(value)})
            for index, child in enumerate(value):
                visit(f"{name}.{index}", child)
        elif isinstance(value, (bool, int, float, str)) or value is None:
            rows.append({"name": name, "kind": "scalar", "type": type(value).__name__})
        else:
            rows.append(
                {
                    "name": name,
                    "kind": "opaque_runtime_value",
                    "type": f"{type(value).__module__}.{type(value).__qualname__}",
                }
            )

    for name, value in values:
        visit(name, value)
    return rows


def runtime_contract(
    *,
    prefix: tuple,
    suffix: tuple,
    shell: tuple,
    fabric: tuple,
    optimizer: tuple,
    full_shell: tuple,
    nstep: tuple = (),
) -> dict[str, object]:
    groups: list[tuple[str, object]] = [
        ("prefix", prefix),
        ("runtime_depth", 0),
        ("suffix", suffix),
        ("shell", shell),
        ("fabric", fabric),
        ("optimizer", optimizer),
        ("full_shell", full_shell),
    ]
    if nstep:
        groups.append(("nstep", nstep))
    rows = argument_rows(groups)
    return {
        "argument_rows": rows,
        "argument_row_count": len(rows),
        "pointer_values_excluded": True,
        "stream_values_excluded": True,
        "rank_and_generation_excluded": True,
    }

