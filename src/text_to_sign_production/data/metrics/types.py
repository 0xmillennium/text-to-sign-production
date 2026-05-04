"""Typed models for sample-local metrics."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from text_to_sign_production.data._shared.identities import SampleSplit


class MetricFamily(enum.StrEnum):
    """Metric families owned by the metric computation layer."""

    OOB = "oob"
    COVERAGE = "coverage"
    HAND = "hand"
    FACE = "face"
    VALID = "valid"
    CONFIDENCE = "confidence"
    TEXT = "text"
    LENGTH = "length"


@dataclass(frozen=True, slots=True)
class MetricValidationIssue:
    """A specific issue found during metric bundle validation."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class OobMetrics:
    """Out-of-bounds metrics for a sample."""

    out_of_bounds_coordinate_count: int
    total_coordinate_slots: int
    out_of_bounds_ratio: float


@dataclass(frozen=True, slots=True)
class CoverageMetrics:
    """Landmark completeness metrics for canonical pose channels."""

    body_landmark_coverage_ratio: float
    left_hand_landmark_coverage_ratio: float
    right_hand_landmark_coverage_ratio: float
    any_hand_landmark_coverage_ratio: float
    face_landmark_coverage_ratio: float


@dataclass(frozen=True, slots=True)
class HandMetrics:
    """Temporal hand availability metrics for a sample."""

    left_hand_available_frame_count: int
    right_hand_available_frame_count: int
    any_hand_available_frame_count: int
    left_hand_available_frame_ratio: float
    right_hand_available_frame_ratio: float
    any_hand_available_frame_ratio: float
    max_any_hand_unavailable_run_count: int
    max_any_hand_unavailable_run_ratio: float


@dataclass(frozen=True, slots=True)
class FaceMetrics:
    """Temporal face availability metrics for a sample."""

    face_available_frame_count: int
    face_unavailable_frame_count: int
    face_available_frame_ratio: float
    face_unavailable_frame_ratio: float


@dataclass(frozen=True, slots=True)
class ValidMetrics:
    """Valid frame metrics for a sample."""

    valid_frame_count: int
    invalid_frame_count: int
    valid_frame_ratio: float
    invalid_frame_ratio: float
    zeroed_canonical_joint_frame_count: int
    zeroed_canonical_joint_frame_ratio: float


@dataclass(frozen=True, slots=True)
class ConfidenceMetrics:
    """Confidence metrics for a sample."""

    body_mean_confidence: float
    left_hand_mean_confidence: float
    right_hand_mean_confidence: float
    face_mean_confidence: float
    overall_mean_confidence: float
    body_nonzero_confidence_ratio: float
    left_hand_nonzero_confidence_ratio: float
    right_hand_nonzero_confidence_ratio: float
    face_nonzero_confidence_ratio: float
    overall_nonzero_confidence_ratio: float


@dataclass(frozen=True, slots=True)
class TextMetrics:
    """Textual metrics for a sample."""

    normalized_text: str
    character_count: int
    token_count: int


@dataclass(frozen=True, slots=True)
class LengthMetrics:
    """Length and duration metrics for a sample."""

    num_frames: int
    fps: float | None
    duration_seconds: float | None
    frames_per_token: float | None
    frames_per_character: float | None


@dataclass(frozen=True, slots=True)
class MetricBundle:
    """The fully composed metric bundle for a sample."""

    sample_id: str
    split: SampleSplit
    oob: OobMetrics
    coverage: CoverageMetrics
    hand: HandMetrics
    face: FaceMetrics
    valid: ValidMetrics
    confidence: ConfidenceMetrics
    text: TextMetrics
    length: LengthMetrics
