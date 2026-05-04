"""Typed models for pose-level semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from text_to_sign_production.data.samples.types import (
    BfhPosePayload,
    FrameQualitySummary,
    SelectedPersonMetadata,
)
from text_to_sign_production.data.sources.types import SourceCandidate

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class FrameFileListing:
    """Deterministic collection of frame JSON files for one sample."""

    directory: Path
    files: tuple[Path, ...]
    missing: bool

    @property
    def frame_count(self) -> int:
        return len(self.files)


@dataclass(slots=True)
class ParsedPerson:
    """Result of parsing a single person in a frame."""

    coords: dict[str, FloatArray]
    confidences: dict[str, FloatArray]
    person_valid: bool
    face_missing: bool
    out_of_bounds_coordinate_count: int
    has_any_zeroed_canonical_joint: bool
    issue_codes: list[str]


@dataclass(slots=True)
class ParsedFrameResult:
    """Result of parsing an OpenPose frame JSON."""

    people: list[ParsedPerson]
    frame_valid: bool
    issue_codes: list[str]


@dataclass(frozen=True, slots=True)
class PoseBuildInput:
    """The complete input required to build pose tensors for a candidate."""

    candidate: SourceCandidate
    frames: FrameFileListing


@dataclass(frozen=True, slots=True)
class PoseBuildDiagnostics:
    """Diagnostics and facts generated during pose building."""

    parse_error: str | None = None
    # Add other diagnostic fields if necessary, or just rely on FrameQualitySummary


@dataclass(frozen=True, slots=True)
class PoseBuildOutput:
    """The canonical output of the pose building stage.

    This aligns perfectly with what the `samples` package needs to construct
    a `ProcessedSamplePayload`.
    """

    pose: BfhPosePayload
    frame_quality: FrameQualitySummary
    selected_person: SelectedPersonMetadata
    diagnostics: PoseBuildDiagnostics
