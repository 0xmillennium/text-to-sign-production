"""Typed models for sample-local metrics."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.legacy_data._shared.types import ValidationIssue


class MetricFamily(enum.StrEnum):
    """Metric families owned by the metric computation layer."""

    ACTIVE_SIGNING_SPAN = "active_signing_span"
    OOB = "oob"
    UPPER_BODY_SUPPORT = "upper_body_support"
    COVERAGE = "coverage"
    MANUAL_VISIBILITY = "manual_visibility"
    NON_MANUAL_VISIBILITY = "non_manual_visibility"
    VALID = "valid"
    CONFIDENCE = "confidence"
    KINEMATIC_NATURALNESS = "kinematic_naturalness"
    TRACKING_QUALITY = "tracking_quality"
    MANUAL_DETAIL = "manual_detail"
    NON_MANUAL_QUALITY = "non_manual_quality"
    GEOMETRY = "geometry"
    TEXT = "text"
    LENGTH = "length"


MetricValidationIssue = ValidationIssue


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

    active_span_upper_body_support_landmark_coverage_ratio: float
    full_clip_upper_body_support_landmark_coverage_ratio: float


@dataclass(frozen=True, slots=True)
class CoverageMetrics:
    """Diagnostic landmark completeness metrics for canonical pose channels."""

    full_body_landmark_coverage_ratio: float
    left_hand_landmark_coverage_ratio: float
    right_hand_landmark_coverage_ratio: float
    any_hand_landmark_coverage_ratio: float
    face_landmark_coverage_ratio: float


@dataclass(frozen=True, slots=True)
class ManualVisibilityMetrics:
    """Active-signing-span manual-articulator visibility metrics for a sample."""

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
class NonManualVisibilityMetrics:
    """Temporal non-manual visibility metrics for a sample."""

    face_available_frame_count: int
    face_unavailable_frame_count: int
    face_available_frame_ratio: float
    face_unavailable_frame_ratio: float
    active_span_face_available_frame_count: int
    active_span_face_available_frame_ratio: float
    max_active_span_face_unavailable_run_count: int
    max_active_span_face_unavailable_run_ratio: float


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

    active_span_body_available_mean_confidence: float
    full_clip_body_available_mean_confidence: float
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
class KinematicNaturalnessMetrics:
    """Active-span motion pathology metrics."""

    active_span_abrupt_motion_frame_ratio: float
    active_span_discontinuity_frame_ratio: float
    max_active_span_frozen_run_ratio: float


@dataclass(frozen=True, slots=True)
class TrackingQualityMetrics:
    """Persisted tracking-quality metrics for a sample."""

    tracked_target_missing_frame_ratio: float
    person_tracking_continuity_break_ratio: float
    person_tracking_reanchor_ratio: float


@dataclass(frozen=True, slots=True)
class ManualDetailMetrics:
    """Active-span fine manual-articulation detail metrics."""

    active_span_representative_hand_landmark_coverage_ratio: float
    active_span_representative_hand_fingertip_coverage_ratio: float
    active_span_representative_hand_distal_chain_coverage_ratio: float
    max_active_span_representative_hand_detail_dropout_run_ratio: float


@dataclass(frozen=True, slots=True)
class NonManualQualityMetrics:
    """Active-span non-manual landmark detail and overlap metrics."""

    active_span_face_landmark_coverage_ratio: float
    active_span_upper_face_landmark_coverage_ratio: float
    active_span_lower_face_landmark_coverage_ratio: float
    max_active_span_face_detail_dropout_run_ratio: float
    active_span_manual_face_overlap_frame_ratio: float


@dataclass(frozen=True, slots=True)
class GeometryMetrics:
    """Active-span sample-internal geometric consistency metrics."""

    active_span_upper_body_bone_length_outlier_frame_ratio: float
    active_span_representative_hand_bone_length_outlier_frame_ratio: float
    active_span_cross_channel_scale_outlier_frame_ratio: float


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
    manual_visibility: ManualVisibilityMetrics
    non_manual_visibility: NonManualVisibilityMetrics
    valid: ValidMetrics
    confidence: ConfidenceMetrics
    kinematic_naturalness: KinematicNaturalnessMetrics
    tracking_quality: TrackingQualityMetrics
    manual_detail: ManualDetailMetrics
    non_manual_quality: NonManualQualityMetrics
    geometry: GeometryMetrics
    text: TextMetrics
    length: LengthMetrics


@dataclass(frozen=True, slots=True)
class MetricObservationRecord:
    """Numeric metric observed on composed metric bundles."""

    family: MetricFamily
    metric_key: str
    observed_count: int
    missing_count: int


@dataclass(frozen=True, slots=True)
class MetricFamilyCountRecord:
    """Split-aware sample count for a metric family."""

    split: SampleSplit
    family: MetricFamily
    sample_count: int


@dataclass(frozen=True, slots=True)
class MetricDistributionRecord:
    """Split-aware numeric distribution for one metric."""

    split: SampleSplit | None
    family: MetricFamily
    metric_key: str
    sample_count: int
    missing_count: int
    unique_value_count: int
    minimum: float | None
    p5: float | None
    p25: float | None
    p50: float | None
    p75: float | None
    p95: float | None
    p99: float | None
    maximum: float | None


@dataclass(frozen=True, slots=True)
class MetricMissingnessRecord:
    """Missingness summary for one metric."""

    split: SampleSplit | None
    family: MetricFamily
    metric_key: str
    sample_count: int
    missing_count: int
    missing_ratio: float


@dataclass(frozen=True, slots=True)
class MetricCoverageSummaryRecord:
    """Coverage-family analytical summary."""

    split: SampleSplit | None
    family: MetricFamily
    metric_key: str
    sample_count: int
    minimum: float | None
    p50: float | None
    p95: float | None
    maximum: float | None


@dataclass(frozen=True, slots=True)
class MetricConfidenceSummaryRecord:
    """Confidence-family analytical summary."""

    split: SampleSplit | None
    metric_key: str
    sample_count: int
    minimum: float | None
    p50: float | None
    p95: float | None
    maximum: float | None


@dataclass(frozen=True, slots=True)
class MetricSaturationRecord:
    """Constant or near-constant metric value finding."""

    split: SampleSplit | None
    family: MetricFamily
    metric_key: str
    sample_count: int
    unique_value_count: int
    reasons: tuple[str, ...]
