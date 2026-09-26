"""Load a Qwen3-8B-width checkpoint into the WORLD8 owner layout.

The compiled CUfunction consumes runtime weight pointers, so model ingress is a
host concern.  This module intentionally contains no compiler or CuTe imports.
It checks that a snapshot has the model dimensions compiled into the image and
the bundle's depth D (``num_hidden_layers``), with its 3 + 11 * D tensors,
describes every source-to-owner slice of the 5 * D + 2 owner segments, and
optionally reads local safetensors files without making safetensors a package
import-time dependency.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from training_megakernel.contract import VOCAB, WORLD_SIZE
from training_megakernel.layout import (
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    KV_HEADS,
    KV_HIDDEN,
    QUERY_HEADS,
    QUERY_HIDDEN,
    owner_segments,
    segment_by_name,
)
from training_megakernel.shape import OWNER_SEGMENTS_PER_LAYER, optimizer_elements

MAPPING_SCHEMA = "training_megakernel_hf_owner_mapping_v1"
TENSORS_PER_LAYER = 11
ROPE_THETA = 1e6


def total_parameters(depth: int) -> int:
    """Parameters of the untied model: every GPU's owner shard plus the final norm."""

    return optimizer_elements(depth) * WORLD_SIZE + HIDDEN


def pinned_config_values(depth: int) -> dict[str, object]:
    """The config values compiled into an image of ``depth`` decoder layers."""

    return {
        "model_type": "qwen3",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": depth,
        "num_attention_heads": QUERY_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 40_960,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
    }


def _normalize_dtype_name(dtype: object) -> str:
    value = str(dtype).lower()
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "torch.bfloat16": "bfloat16",
        "f32": "float32",
        "float32": "float32",
        "torch.float32": "float32",
    }
    try:
        return aliases[value]
    except KeyError as error:
        raise ValueError(f"unsupported tensor dtype {dtype!r}") from error


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if not self.shape or any(not isinstance(extent, int) or extent <= 0 for extent in self.shape):
            raise ValueError(f"invalid tensor shape {self.shape!r}")
        object.__setattr__(self, "dtype", _normalize_dtype_name(self.dtype))

    @property
    def elements(self) -> int:
        return math.prod(self.shape)


def expected_hf_tensor_specs(depth: int) -> dict[str, TensorSpec]:
    """Return the exact untied, bias-free parameter manifest at ``depth`` layers."""

    result: dict[str, TensorSpec] = {
        "model.embed_tokens.weight": TensorSpec((VOCAB, HIDDEN)),
        "model.norm.weight": TensorSpec((HIDDEN,)),
        "lm_head.weight": TensorSpec((VOCAB, HIDDEN)),
    }
    for layer in range(depth):
        prefix = f"model.layers.{layer}"
        result.update(
            {
                f"{prefix}.input_layernorm.weight": TensorSpec((HIDDEN,)),
                f"{prefix}.self_attn.q_proj.weight": TensorSpec((QUERY_HIDDEN, HIDDEN)),
                f"{prefix}.self_attn.k_proj.weight": TensorSpec((KV_HIDDEN, HIDDEN)),
                f"{prefix}.self_attn.v_proj.weight": TensorSpec((KV_HIDDEN, HIDDEN)),
                f"{prefix}.self_attn.q_norm.weight": TensorSpec((HEAD_DIM,)),
                f"{prefix}.self_attn.k_norm.weight": TensorSpec((HEAD_DIM,)),
                f"{prefix}.self_attn.o_proj.weight": TensorSpec((HIDDEN, HIDDEN)),
                f"{prefix}.post_attention_layernorm.weight": TensorSpec((HIDDEN,)),
                f"{prefix}.mlp.gate_proj.weight": TensorSpec((INTERMEDIATE, HIDDEN)),
                f"{prefix}.mlp.up_proj.weight": TensorSpec((INTERMEDIATE, HIDDEN)),
                f"{prefix}.mlp.down_proj.weight": TensorSpec((HIDDEN, INTERMEDIATE)),
            }
        )
    if len(result) != 3 + TENSORS_PER_LAYER * depth or sum(
        spec.elements for spec in result.values()
    ) != total_parameters(depth):
        raise AssertionError("pinned Qwen3-8B parameter ledger drifted")
    return result


def _rope_theta(config: Mapping[str, object]) -> tuple[object, list[str]]:
    """RoPE base from either config form: ``rope_theta``, or transformers 5's
    ``rope_parameters``.  Only unscaled (default) RoPE is compiled in."""

    failures: list[str] = []
    if config.get("rope_scaling") is not None:
        failures.append("rope_scaling must be null or absent")
    parameters = config.get("rope_parameters")
    if "rope_theta" in config or parameters is None:
        return config.get("rope_theta"), failures
    if not isinstance(parameters, Mapping):
        return None, [*failures, "rope_parameters must be an object"]
    if parameters.get("rope_type", "default") != "default":
        failures.append(f"rope_type={parameters.get('rope_type')!r}, expected 'default'")
    return parameters.get("rope_theta"), failures


def validate_pinned_config(config: Mapping[str, object], depth: int) -> None:
    """Reject a config that is not the architecture compiled into the image."""

    failures: list[str] = []
    for name, expected in pinned_config_values(depth).items():
        if name not in config:
            failures.append(f"missing {name}")
        elif config[name] != expected:
            failures.append(f"{name}={config[name]!r}, expected {expected!r}")
    rope_theta, rope_failures = _rope_theta(config)
    failures.extend(rope_failures)
    if rope_theta is None:
        failures.append("missing rope_theta")
    elif rope_theta != ROPE_THETA:
        failures.append(f"rope_theta={rope_theta!r}, expected {ROPE_THETA!r}")
    if config.get("sliding_window") is not None:
        failures.append("sliding_window must be null or absent")
    if config.get("use_sliding_window", False) is not False:
        failures.append("use_sliding_window must be false or absent")
    configured_dtype = config.get("torch_dtype", config.get("dtype"))
    if configured_dtype is not None and _normalize_dtype_name(configured_dtype) != "bfloat16":
        failures.append(f"checkpoint dtype is {configured_dtype!r}, expected bfloat16")
    if failures:
        raise ValueError("checkpoint config does not match the compiled Qwen3-8B contract: " + "; ".join(failures))


def validate_hf_parameter_manifest(manifest: Mapping[str, TensorSpec], depth: int) -> None:
    """Require every and only the untied, bias-free model parameter at ``depth`` layers."""

    bias_names = sorted(name for name in manifest if name.endswith(".bias"))
    if bias_names:
        raise ValueError(f"bias tensors are not supported: {bias_names[:3]!r}")
    expected = expected_hf_tensor_specs(depth)
    missing = sorted(set(expected) - set(manifest))
    unexpected = sorted(set(manifest) - set(expected))
    if missing or unexpected:
        raise ValueError(
            "checkpoint tensor set differs from the pinned untied model: "
            f"missing={missing[:3]!r}, unexpected={unexpected[:3]!r}"
        )
    mismatches: list[str] = []
    for name, required in expected.items():
        observed = manifest[name]
        if observed != required:
            mismatches.append(f"{name}: observed={observed!r}, expected={required!r}")
    if mismatches:
        raise ValueError("checkpoint tensor metadata differs from the pinned model: " + "; ".join(mismatches[:3]))


@dataclass(frozen=True)
class SourceSlice:
    tensor_name: str
    source_begin: int
    source_end: int
    target_begin: int
    target_end: int

    @property
    def elements(self) -> int:
        return self.source_end - self.source_begin


@dataclass(frozen=True)
class OwnerSegmentPlan:
    segment_name: str
    rank: int
    target_begin: int
    target_end: int
    sources: tuple[SourceSlice, ...]

    @property
    def elements(self) -> int:
        return self.target_end - self.target_begin


@dataclass(frozen=True)
class _SourcePiece:
    tensor_name: str
    elements: int


def _decoder_source_pieces(layer: int, site: str) -> tuple[_SourcePiece, ...]:
    prefix = f"model.layers.{layer}"
    rows: dict[str, tuple[tuple[str, int], ...]] = {
        "down_dw": ((f"{prefix}.mlp.down_proj.weight", HIDDEN * INTERMEDIATE),),
        "gate_up_dw": (
            (f"{prefix}.mlp.gate_proj.weight", INTERMEDIATE * HIDDEN),
            (f"{prefix}.mlp.up_proj.weight", INTERMEDIATE * HIDDEN),
        ),
        "o_dw": ((f"{prefix}.self_attn.o_proj.weight", HIDDEN * HIDDEN),),
        "qkv_dw": (
            (f"{prefix}.self_attn.q_proj.weight", QUERY_HIDDEN * HIDDEN),
            (f"{prefix}.self_attn.k_proj.weight", KV_HIDDEN * HIDDEN),
            (f"{prefix}.self_attn.v_proj.weight", KV_HIDDEN * HIDDEN),
        ),
        "norm": (
            (f"{prefix}.input_layernorm.weight", HIDDEN),
            (f"{prefix}.self_attn.q_norm.weight", HEAD_DIM),
            (f"{prefix}.self_attn.k_norm.weight", HEAD_DIM),
            (f"{prefix}.post_attention_layernorm.weight", HIDDEN),
        ),
    }
    try:
        return tuple(_SourcePiece(*row) for row in rows[site])
    except KeyError as error:
        raise ValueError(f"unknown decoder owner site {site!r}") from error


def _source_pieces_for_segment(segment_name: str, depth: int) -> tuple[_SourcePiece, ...]:
    if segment_name == "embedding":
        return (_SourcePiece("model.embed_tokens.weight", VOCAB * HIDDEN),)
    if segment_name == "head":
        return (_SourcePiece("lm_head.weight", VOCAB * HIDDEN),)
    parts = segment_name.split(".")
    if len(parts) != 3 or parts[0] != "decoder":
        raise ValueError(f"unknown owner segment {segment_name!r}")
    layer = int(parts[1])
    if not 0 <= layer < depth:
        raise ValueError(f"decoder layer is outside the depth: {layer}")
    return _decoder_source_pieces(layer, parts[2])


def owner_segment_plan(segment_name: str, rank: int, depth: int) -> OwnerSegmentPlan:
    """Map one rank-contiguous fused owner shard onto its HF source tensors."""

    if not 0 <= rank < WORLD_SIZE:
        raise ValueError(f"rank {rank} is outside WORLD{WORLD_SIZE}")
    try:
        segment = segment_by_name(depth)[segment_name]
    except KeyError as error:
        raise ValueError(f"unknown owner segment {segment_name!r}") from error
    pieces = _source_pieces_for_segment(segment_name, depth)
    full_elements = sum(piece.elements for piece in pieces)
    if full_elements != segment.elements * WORLD_SIZE:
        raise AssertionError(f"source ledger does not fill owner segment {segment_name}")

    global_begin = rank * segment.elements
    global_end = global_begin + segment.elements
    piece_begin = 0
    sources: list[SourceSlice] = []
    for piece in pieces:
        piece_end = piece_begin + piece.elements
        intersection_begin = max(global_begin, piece_begin)
        intersection_end = min(global_end, piece_end)
        if intersection_begin < intersection_end:
            target_begin = segment.begin + intersection_begin - global_begin
            target_end = target_begin + intersection_end - intersection_begin
            sources.append(
                SourceSlice(
                    tensor_name=piece.tensor_name,
                    source_begin=intersection_begin - piece_begin,
                    source_end=intersection_end - piece_begin,
                    target_begin=target_begin,
                    target_end=target_end,
                )
            )
        piece_begin = piece_end
    if (
        not sources
        or sources[0].target_begin != segment.begin
        or sources[-1].target_end != segment.end
        or sum(row.elements for row in sources) != segment.elements
    ):
        raise AssertionError(f"source slices do not exactly cover owner segment {segment_name}")
    return OwnerSegmentPlan(
        segment_name=segment_name,
        rank=rank,
        target_begin=segment.begin,
        target_end=segment.end,
        sources=tuple(sources),
    )


def owner_parameter_plan(rank: int, depth: int) -> tuple[OwnerSegmentPlan, ...]:
    result = tuple(
        owner_segment_plan(segment.name, rank, depth) for segment in owner_segments(depth)
    )
    if len(result) != OWNER_SEGMENTS_PER_LAYER * depth + 2 or sum(
        row.elements for row in result
    ) != optimizer_elements(depth):
        raise AssertionError("rank owner plan does not cover the optimizer parameter")
    return result


def _import_safe_open() -> Any:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError(
            "loading an HF checkpoint requires the optional 'safetensors' package"
        ) from error
    return safe_open


class HFSafetensorSource:
    """Validated, local-only reader for a Qwen3-8B-width HF safetensors snapshot.

    ``depth`` is the bundle's decoder depth; the snapshot must have exactly that
    many layers.
    """

    def __init__(self, model_directory: str | Path, *, depth: int) -> None:
        root = Path(model_directory).resolve()
        config_path = root / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing checkpoint config: {config_path}")
        self.root = root
        self.depth = depth
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(self.config, dict):
            raise ValueError("checkpoint config.json must contain an object")
        validate_pinned_config(self.config, depth)
        self._tensor_files = self._discover_tensor_files()
        self.manifest = self._inspect_manifest()

    def _resolve_shard(self, relative_name: object) -> Path:
        if not isinstance(relative_name, str) or not relative_name:
            raise ValueError(f"invalid safetensors shard name {relative_name!r}")
        path = (self.root / relative_name).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise FileNotFoundError(f"invalid or missing safetensors shard: {relative_name!r}")
        return path

    def _discover_tensor_files(self) -> dict[str, Path]:
        index_path = self.root / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("safetensors index has no weight_map")
            return {str(name): self._resolve_shard(shard) for name, shard in weight_map.items()}

        single = self.root / "model.safetensors"
        if not single.is_file():
            raise FileNotFoundError("checkpoint has neither model.safetensors nor its index")
        safe_open = _import_safe_open()
        with safe_open(single, framework="pt", device="cpu") as handle:
            return {str(name): single for name in handle.keys()}

    def _inspect_manifest(self) -> dict[str, TensorSpec]:
        safe_open = _import_safe_open()
        grouped: dict[Path, list[str]] = {}
        for name, path in self._tensor_files.items():
            grouped.setdefault(path, []).append(name)
        result: dict[str, TensorSpec] = {}
        for path, names in grouped.items():
            with safe_open(path, framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                absent = sorted(set(names) - available)
                if absent:
                    raise ValueError(f"safetensors index names missing from {path.name}: {absent[:3]!r}")
                for name in names:
                    row = handle.get_slice(name)
                    result[name] = TensorSpec(tuple(row.get_shape()), str(row.get_dtype()))
        validate_hf_parameter_manifest(result, self.depth)
        return result

    def read_flat(self, tensor_name: str, begin: int, end: int) -> torch.Tensor:
        try:
            spec = self.manifest[tensor_name]
            path = self._tensor_files[tensor_name]
        except KeyError as error:
            raise ValueError(f"unknown source tensor {tensor_name!r}") from error
        if not 0 <= begin <= end <= spec.elements:
            raise ValueError(f"invalid flat slice {begin}:{end} for {tensor_name}")
        safe_open = _import_safe_open()
        with safe_open(path, framework="pt", device="cpu") as handle:
            row = handle.get_slice(tensor_name)
            if len(spec.shape) == 1:
                result = row[begin:end]
            elif len(spec.shape) == 2:
                columns = spec.shape[1]
                if begin % columns or end % columns:
                    raise ValueError(f"source slice for {tensor_name} is not row aligned")
                result = row[begin // columns : end // columns, :]
            else:
                raise ValueError(f"unsupported checkpoint tensor rank for {tensor_name}")
        return result.reshape(-1).contiguous()


def load_hf_rank_into_optimizer(
    optimizer: Any,
    source: HFSafetensorSource,
    *,
    rank: int,
) -> dict[str, object]:
    """Stream one BF16 owner shard into an allocated optimizer state.

    This is intended for step-zero initialization.  It resets gradients,
    moments, and progress fields but deliberately leaves runtime
    hyperparameters under the runner's control.  The source's depth is the
    bundle's; the optimizer must own O(depth) elements.
    """

    depth = source.depth
    bf16_parameter = optimizer.bf16_parameter
    final_bf16 = optimizer.final_bf16_parameter
    with torch.no_grad():
        for segment in owner_parameter_plan(rank, depth):
            for source_slice in segment.sources:
                values = source.read_flat(
                    source_slice.tensor_name,
                    source_slice.source_begin,
                    source_slice.source_end,
                )
                bf16_parameter[source_slice.target_begin : source_slice.target_end].copy_(
                    values.to(device=bf16_parameter.device)
                )
        optimizer.parameter.copy_(bf16_parameter)
        final_bf16.copy_(
            source.read_flat("model.norm.weight", 0, HIDDEN).to(device=final_bf16.device)
        )
        optimizer.final_parameter.copy_(final_bf16)

        for name in (
            "gradient",
            "exp_avg",
            "exp_avg_sq",
            "final_gradient",
            "final_exp_avg",
            "final_exp_avg_sq",
            "norm_partials",
            "outputs",
            "status",
            "completed_steps",
        ):
            getattr(optimizer, name).zero_()

    return {
        "schema": "training_megakernel_hf_initialization_v1",
        "mapping_schema": MAPPING_SCHEMA,
        "rank": rank,
        "depth": depth,
        "owner_elements": optimizer_elements(depth),
        "owner_segments": len(owner_segments(depth)),
        "final_norm_replicated": True,
        "optimizer_step": 0,
    }
