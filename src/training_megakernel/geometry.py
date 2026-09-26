"""The shape of one compiled bundle, as the host runtime allocates it.

A bundle's ``launch_abi.json`` records the depth D and tokens per GPU S it was built
for (``logical_depth`` and ``sequence``), the values that follow from them, the most
steps one launch may run, and the embedding-route layout.

Every published derived value must equal the formula in ``shape.py``, because the
host allocates by those formulas; the load-time comparison of the whole argument
layout (``runtime._runtime_abi_matches``) then checks each tensor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .route_layout import EmbeddingRouteLayout, embedding_route_layout
from .shape import CheckpointPlan, Shape


@dataclass(frozen=True)
class Geometry:
    """Depth, tokens per GPU and launch limits of one bundle.

    The formulas live in :class:`~training_megakernel.shape.Shape`; this object adds
    what only the bundle knows, and every allocation and validation in the host
    runtime takes its sizes from here.
    """

    shape: Shape
    resident_step_limit: int
    route_layout: EmbeddingRouteLayout

    @classmethod
    def from_launch_abi(cls, launch: Mapping[str, Any]) -> Geometry:
        """Read a bundle's shape from its ``launch_abi.json`` object."""

        try:
            shape = Shape(launch["logical_depth"], launch["sequence"])
        except ValueError as error:
            raise RuntimeError(f"bundle shape is not supported: {error}") from error
        return cls(
            shape=shape,
            resident_step_limit=launch["resident_step_limit"],
            route_layout=embedding_route_layout(
                launch["embedding_route_records_per_source_owner"],
                compact_owner_gradient=launch["embedding_route_compact_owner_gradient"],
            ),
        )

    @property
    def depth(self) -> int:
        return self.shape.depth

    @property
    def sequence(self) -> int:
        return self.shape.sequence

    @property
    def optimizer_elements(self) -> int:
        return self.shape.optimizer_elements

    @property
    def head_chunk_rows(self) -> int:
        return self.shape.head_chunk_rows

    @property
    def slot_rows(self) -> int:
        return self.shape.slot_rows

    @property
    def control_elements(self) -> int:
        return self.shape.control_elements

    @property
    def checkpoint_plan(self) -> CheckpointPlan:
        return self.shape.checkpoint_plan

    @property
    def route_records(self) -> int:
        """Embedding-route records per (source, owner) GPU pair."""

        return self.route_layout.records_per_source_owner

    def describe(self) -> str:
        return f"{self.depth} layers and {self.sequence} tokens per GPU"


__all__ = ("Geometry",)
