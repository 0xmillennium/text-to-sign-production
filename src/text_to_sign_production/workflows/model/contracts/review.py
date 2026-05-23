"""Presentation data rows for model workflow review surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelReviewArtifactRow:
    label: str
    kind: str
    path: Path
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ModelReviewStageRow:
    index: int
    stage_id: str
    kind: str
    status: str | None = None


@dataclass(frozen=True, slots=True)
class ModelReviewPublishRow:
    label: str
    kind: str
    source_path: Path
    target_path: Path


__all__ = [
    "ModelReviewArtifactRow",
    "ModelReviewPublishRow",
    "ModelReviewStageRow",
]
