"""Typed models for sample-local metrics."""

from __future__ import annotations

from dataclasses import dataclass


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
class HandMetrics:
    """Hand presence metrics for a sample."""

    left_hand_nonzero_frame_count: int
    right_hand_nonzero_frame_count: int
    any_hand_nonzero_frame_count: int
    left_hand_nonzero_frame_ratio: float
    right_hand_nonzero_frame_ratio: float
    any_hand_nonzero_frame_ratio: float


@dataclass(frozen=True, slots=True)
class FaceMetrics:
    """Face presence metrics for a sample."""

    face_nonzero_frame_count: int
    face_missing_frame_count: int
    face_nonzero_frame_ratio: float
    face_missing_frame_ratio: float


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
    split: str
    oob: OobMetrics
    hand: HandMetrics
    face: FaceMetrics
    valid: ValidMetrics
    confidence: ConfidenceMetrics
    text: TextMetrics
    length: LengthMetrics
