"""Dependency-light text backbone contracts for Sprint 3 baseline modeling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class TextBackboneOutput:
    """Stable encoded text representation consumed by baseline models."""

    token_embeddings: Any
    pooled_embedding: Any
    attention_mask: Any


@runtime_checkable
class TextBackbone(Protocol):
    """Minimal interface expected from text backbones."""

    @property
    def output_dim(self) -> int:
        """Embedding width of the pooled text representation."""

        ...

    def __call__(
        self,
        texts: Sequence[str],
        *,
        device: Any | None = None,
    ) -> TextBackboneOutput:
        """Encode raw English text strings into a stable tensor representation."""

        ...
