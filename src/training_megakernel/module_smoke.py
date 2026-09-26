"""Validate, load, and resolve the packaged CUfunction without launching it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .circular_refill_runtime import TOKEN_SLOTS, CircularNstepRuntimeState
from .geometry import Geometry
from .runtime import FUNCTION_NAME, ApplicationState, load_image, read_bundle
from .shards import TokenShardRing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    bundle = parser.parse_args().bundle.resolve()
    _, launch = read_bundle(bundle)
    geometry = Geometry.from_launch_abi(launch)
    # The complete state graph on the meta device: load_image compares its every
    # argument's shape, stride and storage alias with the bundle's without a GPU.
    state = ApplicationState.meta(geometry)
    state.nstep = CircularNstepRuntimeState.allocate(
        resident_steps=launch["resident_compile_sample_steps"],
        device=torch.device("meta"),
        refill_device_address=0,
        base_generation=1,
        timeout_ns=1,
    )
    state.token_shards = TokenShardRing.meta(slots=TOKEN_SLOTS, geometry=geometry)
    image = load_image(bundle, state)
    print(
        json.dumps(
            {
                "bundle": str(bundle),
                "function_name": FUNCTION_NAME,
                "function_type": (
                    f"{type(image.function).__module__}.{type(image.function).__qualname__}"
                ),
                "logical_depth": geometry.depth,
                "sequence": geometry.sequence,
                "pass": True,
                "resident_step_limit": geometry.resident_step_limit,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
