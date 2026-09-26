"""Dependency and fixed WORLD8 hardware preflight without compiling the image."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
from typing import Any

from . import contract

# The packages the kernel code is written against, at their exact versions.
RUNTIME_PINS = {"nvidia-cutlass-dsl": "4.6.0"}
BUILD_PINS = {
    "flash-attn-4": "4.0.0b20.dev8+g890f238",
    "quack-kernels": "0.6.0",
}

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
GPU_NAME = re.compile(r"GPU([0-9]+)")


def _installed(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_nvlink_topology(output: str, *, device_count: int) -> dict[str, Any]:
    """Require the eight GPUs to form the NVLink fabric used by NVLS mappings."""

    rows: dict[int, list[str]] = {}
    for line in ANSI_ESCAPE.sub("", output).splitlines():
        fields = line.split()
        match = GPU_NAME.fullmatch(fields[0]) if fields else None
        if match is None:
            continue
        index = int(match.group(1))
        if index < device_count and len(fields) >= device_count + 1:
            rows[index] = fields[1 : device_count + 1]
    full_mesh = bool(
        device_count > 0
        and len(rows) == device_count
        and all(
            cells[peer] == "X" if peer == index else cells[peer].startswith("NV")
            for index, cells in rows.items()
            for peer in range(device_count)
        )
    )
    return {
        "rows": {str(index): cells for index, cells in sorted(rows.items())},
        "pass": full_mesh,
    }


def _inspect_world8(torch: Any) -> dict[str, Any]:
    cuda = torch.cuda
    available = bool(cuda.is_available())
    count = int(cuda.device_count()) if available else 0
    devices = []
    for index in range(count):
        properties = cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "sm_count": properties.multi_processor_count,
            }
        )
    shape_pass = count == contract.WORLD_SIZE and all(
        row["compute_capability"] == [9, 0] and row["sm_count"] == contract.PROGRAM_CTAS
        for row in devices
    )
    try:
        output = subprocess.run(
            ["nvidia-smi", "topo", "-m"], text=True, capture_output=True, check=True
        ).stdout
        topology = inspect_nvlink_topology(output, device_count=count)
    except (OSError, subprocess.SubprocessError) as error:
        topology = {"pass": False, "error": f"{type(error).__name__}: {error}"}
    return {
        "device_count": count,
        "devices": devices,
        "nvlink_topology": topology,
        "pass": bool(shape_pass and topology["pass"]),
    }


def inspect_environment(*, require_world8: bool, require_build: bool) -> dict[str, Any]:
    """Return a machine-readable preflight."""

    pins = {**RUNTIME_PINS, **(BUILD_PINS if require_build else {})}
    packages = {}
    for name, version in pins.items():
        observed = _installed(name)
        packages[name] = {
            "observed": observed,
            "required": f"=={version}",
            "pass": observed == version,
        }
    try:
        import torch
    except ImportError:
        torch = None
    torch_cuda = None if torch is None else torch.version.cuda
    packages["torch"] = {
        "observed": _installed("torch"),
        "cuda": torch_cuda,
        "required": "CUDA 13",
        "pass": bool(torch_cuda and torch_cuda.startswith("13.")),
    }
    python = ".".join(map(str, sys.version_info[:2]))
    report: dict[str, Any] = {
        "python": {"observed": python, "required": "3.12", "pass": python == "3.12"},
        "packages": packages,
    }
    if require_world8:
        report["world8"] = (
            _inspect_world8(torch) if torch is not None else {"pass": False}
        )
    report["pass"] = all(
        row["pass"]
        for row in (
            report["python"],
            *packages.values(),
            report.get("world8", {"pass": True}),
        )
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--world8",
        action="store_true",
        help="also require exactly eight 132-SM compute-capability-9.0 GPUs on NVLink",
    )
    result.add_argument(
        "--build",
        action="store_true",
        help="also require FlashAttention and Quack compiler dependencies",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    report = inspect_environment(require_world8=args.world8, require_build=args.build)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
