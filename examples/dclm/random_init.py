#!/usr/bin/env python3
"""Write a randomly initialized Qwen3-8B-width snapshot with D decoder layers.

Pretrained Qwen3-8B weights exist only at 36 layers.  For any other depth this
writes a Hugging Face snapshot the DCLM example loads like any other
(``TMK_HF_MODEL_SNAPSHOT``): Qwen3-8B's config with ``num_hidden_layers = D``,
weights initialized by transformers from a fixed seed (normal with standard
deviation ``initializer_range`` for the matrices and embeddings, ones for the
norms), saved as BF16 safetensors.

Requires ``transformers`` (``python -m pip install transformers``), which the
training runtime itself does not use.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from dclm_example.model_loader import HFSafetensorSource

from training_megakernel.shape import MAX_DEPTH, MIN_DEPTH


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("output", type=Path, help="new snapshot directory")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="the Qwen3-8B snapshot directory, or its config.json",
    )
    depth = parser.add_mutually_exclusive_group(required=True)
    depth.add_argument("--depth", type=int, help="decoder layers")
    depth.add_argument(
        "--bundle", type=Path, help="release bundle whose depth to use"
    )
    parser.add_argument("--seed", type=int, default=0, help="torch seed (default 0)")
    args = parser.parse_args()
    if args.bundle is not None:
        from training_megakernel.runtime import bundle_geometry

        args.depth = bundle_geometry(args.bundle).depth
    if not MIN_DEPTH <= args.depth <= MAX_DEPTH:
        parser.error(f"--depth must be in [{MIN_DEPTH}, {MAX_DEPTH}]")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.config.is_dir():
        args.config = args.config / "config.json"
    return args


def main() -> None:
    args = _parse_args()
    source = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or source.get("model_type") != "qwen3":
        raise SystemExit(f"{args.config} is not a Qwen3 config")

    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    values = {
        name: value
        for name, value in source.items()
        # Derived from num_hidden_layers by transformers; drop the source's.
        if name != "layer_types"
    }
    values["num_hidden_layers"] = args.depth
    values["max_window_layers"] = args.depth
    config = Qwen3Config(**values)

    torch.manual_seed(args.seed)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Qwen3ForCausalLM(config)
    finally:
        torch.set_default_dtype(default_dtype)
    if {parameter.dtype for parameter in model.parameters()} != {torch.bfloat16}:
        raise RuntimeError("random initialization did not produce BF16 parameters")
    parameters = sum(parameter.numel() for parameter in model.parameters())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    )
    try:
        model.save_pretrained(staging)
        del model
        # The snapshot must load the way training loads it.
        loaded = HFSafetensorSource(staging, depth=args.depth)
        tensors = len(loaded.manifest)
        os.chmod(staging, 0o755)
        staging.rename(args.output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output": str(args.output),
                "depth": args.depth,
                "seed": args.seed,
                "parameters": parameters,
                "tensors": tensors,
                "files": sorted(path.name for path in args.output.iterdir()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
