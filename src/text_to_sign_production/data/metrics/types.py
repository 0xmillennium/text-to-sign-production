"""Typed models for sample-local metrics."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from text_to_sign_production.data._shared.identities import SampleSplit


class MetricFamily(enum.StrEnum):
    """Metric families owned by the metric computation layer."""

    ACTIVE_SIGNING_SPAN = "active_signing_span"
    OOB = "oob"
    UPPER_BODY_SUPPORT = "upper_body_support"
    COVERAGE = "coverage"
    HAND = "hand"
    FACE = "face"
    VALID = "valid"
    CONFIDENCE = "confidence"
    TEMPORAL_COHERENCE = "temporal_coherence"
    TEXT = "text"
    LENGTH = "length"


@dataclass(frozen=True, slots=True)
class MetricValidationIssue:
    """A specific issue found during metric bundle validation."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ActiveSigningSpanMetrics:
    """Deterministic signing-relevant span and its diagnostics."""

    start_frame_index: int
    end_frame_index_exclusive: int
    frame_count: int
    frame_ratio: float
    trimmed_prefix_frame_count: int
    trimmed_prefix_frame_ratio: float
    trimmed_suffix_frame_count: int
    trimmed_suffix_frame_ratio: float
    sustained_any_hand_evidence_frame_count: int
    sustained_any_hand_evidence_frame_ratio: float
    sustained_upper_body_evidence_frame_count: int
    sustained_upper_body_evidence_frame_ratio: float
    sustained_motion_evidence_frame_count: int
    sustained_motion_evidence_frame_ratio: float
    source: str


@dataclass(frozen=True, slots=True)
class OobMetrics:
    """Out-of-bounds metrics for a sample."""

    out_of_bounds_coordinate_count: int
    total_coordinate_slots: int
    out_of_bounds_ratio: float


@dataclass(frozen=True, slots=True)
class UpperBodySupportMetrics:
    """Binding upper-body/signing-relevant support metrics."""

    upper_body_support_landmark_coverage_ratio: float


@dataclass(frozen=True, slots=True)
class CoverageMetrics:
    """Diagnostic landmark completeness metrics for canonical pose channels."""

    full_body_landmark_coverage_ratio: float
    left_hand_landmark_coverage_ratio: float
    right_hand_landmark_coverage_ratio: float
    any_hand_landmark_coverage_ratio: float
    face_landmark_coverage_ratio: float


@dataclass(frozen=True, slots=True)
class HandMetrics:
    """Active-signing-span hand availability metrics for a sample."""

    whole_clip_left_hand_available_frame_count: int
    whole_clip_right_hand_available_frame_count: int
    whole_clip_any_hand_available_frame_count: int
    whole_clip_left_hand_available_frame_ratio: float
    whole_clip_right_hand_available_frame_ratio: float
    whole_clip_any_hand_available_frame_ratio: float
    active_span_any_hand_available_frame_count: int
    active_span_any_hand_available_frame_ratio: float
    max_active_span_any_hand_unavailable_run_count: int
    max_active_span_any_hand_unavailable_run_ratio: float


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
    """Availability-aware confidence quality metrics for a sample."""

    body_available_mean_confidence: float
    active_span_left_hand_available_mean_confidence: float
    active_span_right_hand_available_mean_confidence: float
    active_span_any_hand_available_mean_confidence: float
    face_available_mean_confidence: float
    overall_available_mean_confidence: float
    body_nonzero_confidence_ratio: float
    left_hand_nonzero_confidence_ratio: float
    right_hand_nonzero_confidence_ratio: float
    face_nonzero_confidence_ratio: float
    overall_nonzero_confidence_ratio: float


@dataclass(frozen=True, slots=True)
class TemporalCoherenceMetrics:
    """Active-span motion pathology metrics."""

    active_span_abrupt_motion_frame_ratio: float
    active_span_discontinuity_frame_ratio: float
    max_active_span_frozen_run_ratio: float


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
    active_signing_span: ActiveSigningSpanMetrics
    oob: OobMetrics
    upper_body_support: UpperBodySupportMetrics
    coverage: CoverageMetrics
    hand: HandMetrics
    face: FaceMetrics
    valid: ValidMetrics
    confidence: ConfidenceMetrics
    temporal_coherence: TemporalCoherenceMetrics
    text: TextMetrics
    length: LengthMetrics
