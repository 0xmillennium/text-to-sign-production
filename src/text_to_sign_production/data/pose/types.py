"""Typed models for pose-level semantics."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from text_to_sign_production.data._shared.types import ValidationIssue
from text_to_sign_production.data.samples.types import (
    BfhPosePayload,
    FrameQualitySummary,
    SelectedPersonMetadata,
)
from text_to_sign_production.data.sources.types import SourceCandidate

FloatArray = npt.NDArray[np.float32]


class PersonSelectionPolicy(enum.StrEnum):
    """Deterministic policy for selecting one OpenPose person track."""

    OPENPOSE_PRIMARY = "openpose_primary"
    HIGHEST_CANONICAL_SIGNAL = "highest_canonical_signal"


@dataclass(frozen=True, slots=True)
class PersonSelectionCandidateScore:
    """Inspectable signal score for one OpenPose person index."""

    index: int
    aggregate_canonical_signal: float
    aggregate_body_signal: float
    valid_frame_presence_count: int


@dataclass(frozen=True, slots=True)
class PersonSelectionResult:
    """Resolved person selection plus policy diagnostics."""

    target_index: int
    policy: PersonSelectionPolicy
    fallback_used: bool
    fallback_reason: str | None
    candidate_scores: tuple[PersonSelectionCandidateScore, ...]


@dataclass(frozen=True, slots=True)
class PersonTrackingResult:
    """Frame-wise person resolution anchored by the global selection result."""

    anchor_selection: PersonSelectionResult
    selected_person_indices: tuple[int, ...]
    continuity_break_count: int
    continuity_break_ratio: float
    reanchor_count: int
    reanchor_ratio: float
    tracked_target_missing_frame_count: int
    tracked_target_missing_frame_ratio: float


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
    person_selection_policy: PersonSelectionPolicy = PersonSelectionPolicy.HIGHEST_CANONICAL_SIGNAL


@dataclass(frozen=True, slots=True)
class PoseBuildDiagnostics:
    """Diagnostics and facts generated during pose building."""

    person_selection_policy: PersonSelectionPolicy
    person_selection_fallback_used: bool = False
    person_selection_fallback_reason: str | None = None
    person_selection_candidate_scores: tuple[PersonSelectionCandidateScore, ...] = ()
    tracked_person_indices: tuple[int, ...] = ()
    person_tracking_continuity_break_count: int = 0
    person_tracking_continuity_break_ratio: float = 0.0
    person_tracking_reanchor_count: int = 0
    person_tracking_reanchor_ratio: float = 0.0
    tracked_target_missing_frame_count: int = 0
    tracked_target_missing_frame_ratio: float = 0.0
    unrecoverable_error: str | None = None


@dataclass(frozen=True, slots=True)
class PoseBuildOutput:
    """The canonical output of the pose building stage.

    This aligns perfectly with what the `samples` package needs to construct
    a `ProcessedSamplePayload`.
    """

    pose: BfhPosePayload
    frame_quality: FrameQualitySummary
    selected_person: SelectedPersonMetadata
    people_per_frame: npt.NDArray[np.integer[Any]]
    frame_valid_mask: npt.NDArray[np.bool_]
    diagnostics: PoseBuildDiagnostics


PoseValidationIssue = ValidationIssue


@dataclass(frozen=True, slots=True)
class PoseIssueFrequencyRecord:
    """Frequency of pose frame/build issue codes."""

    issue_code: str
    count: int


@dataclass(frozen=True, slots=True)
class PosePersonSelectionFallbackCountRecord:
    """Selected-person fallback count for one policy."""

    policy: PersonSelectionPolicy
    fallback_used: bool
    count: int


@dataclass(frozen=True, slots=True)
class PoseChannelNonzeroDistributionRecord:
    """Distribution of nonzero frame counts for one pose channel."""

    channel: str
    sample_count: int
    missing_count: int
    minimum: float | None
    p50: float | None
    p95: float | None
    maximum: float | None


@dataclass(frozen=True, slots=True)
class PosePeoplePerFrameDistributionRecord:
    """Distribution of people-per-frame values across pose outputs."""

    output_count: int
    frame_count: int
    minimum: float | None
    p50: float | None
    p95: float | None
    maximum: float | None


@dataclass(frozen=True, slots=True)
class PoseBuildDiagnosticAggregateRecord:
    """Aggregate pose build diagnostic counts."""

    output_count: int
    unrecoverable_error_count: int
    fallback_used_count: int
