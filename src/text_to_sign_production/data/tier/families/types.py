"""Type system for quality-family metric computation."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from text_to_sign_production.core.models import PreparedSample
from text_to_sign_production.data.tier.context import QualityContext
from text_to_sign_production.data.tier.facts import QualityFacts


class BindingQualityFamily(enum.StrEnum):
    """Metric families intended for later policy binding."""

    OOB = "oob"
    UPPER_BODY_SUPPORT = "upper_body_support"
    MANUAL_VISIBILITY = "manual_visibility"
    NON_MANUAL_VISIBILITY = "non_manual_visibility"
    CONFIDENCE = "confidence"
    KINEMATIC_NATURALNESS = "kinematic_naturalness"
    TRACKING_QUALITY = "tracking_quality"
    MANUAL_DETAIL = "manual_detail"
    NON_MANUAL_QUALITY = "non_manual_quality"
    GEOMETRY = "geometry"


class DiagnosticQualityFamily(enum.StrEnum):
    """Diagnostic metric families that remain non-binding."""

    TEXT = "text"
    LENGTH = "length"


@dataclass(frozen=True, slots=True)
class QualityMetricBuildInput:
    """Single explicit input contract for quality-family metric computation."""

    sample: PreparedSample
    quality_facts: QualityFacts
    quality_context: QualityContext


@dataclass(frozen=True, slots=True)
class OobMetrics:
    """Spatial out-of-bounds channel metrics."""

    body_out_of_bounds_frame_ratio: float
    left_hand_out_of_bounds_frame_ratio: float
    right_hand_out_of_bounds_frame_ratio: float
    face_out_of_bounds_frame_ratio: float
    max_channel_out_of_bounds_frame_ratio: float


@dataclass(frozen=True, slots=True)
class UpperBodySupportMetrics:
    """Active-span upper-body support metrics."""

    active_span_upper_body_available_frame_ratio: float
    active_span_upper_body_landmark_coverage_ratio: float
    max_active_span_upper_body_dropout_run_ratio: float


@dataclass(frozen=True, slots=True)
class ManualVisibilityMetrics:
    """Active-span coarse manual visibility metrics."""

    active_span_any_hand_available_frame_ratio: float
    active_span_both_hands_unavailable_frame_ratio: float
    max_active_span_any_hand_dropout_run_ratio: float


@dataclass(frozen=True, slots=True)
class NonManualVisibilityMetrics:
    """Active-span coarse face/non-manual visibility metrics."""

    active_span_face_available_frame_ratio: float
    max_active_span_face_unavailable_run_ratio: float


@dataclass(frozen=True, slots=True)
class ConfidenceMetrics:
    """Active-span confidence metrics."""

    active_span_body_mean_confidence: float
    active_span_hand_mean_confidence: float
    active_span_face_mean_confidence: float


@dataclass(frozen=True, slots=True)
class KinematicNaturalnessMetrics:
    """Representative-articulator motion metrics."""

    comparable_transition_abrupt_ratio: float
    comparable_transition_discontinuity_ratio: float
    active_span_frozen_frame_ratio: float


@dataclass(frozen=True, slots=True)
class TrackingQualityMetrics:
    """Tracking stability metrics."""

    tracked_target_missing_frame_ratio: float
    person_tracking_continuity_break_ratio: float
    person_tracking_reanchor_ratio: float


@dataclass(frozen=True, slots=True)
class ManualDetailMetrics:
    """Representative-hand fine detail metrics."""

    active_span_representative_hand_landmark_coverage_ratio: float
    active_span_representative_hand_fingertip_coverage_ratio: float
    active_span_representative_hand_distal_chain_coverage_ratio: float
    max_active_span_representative_hand_detail_dropout_run_ratio: float


@dataclass(frozen=True, slots=True)
class NonManualQualityMetrics:
    """Face/non-manual detail metrics."""

    active_span_face_landmark_coverage_ratio: float
    active_span_upper_face_landmark_coverage_ratio: float
    active_span_lower_face_landmark_coverage_ratio: float
    max_active_span_face_detail_dropout_run_ratio: float
    active_span_manual_face_overlap_ratio: float


@dataclass(frozen=True, slots=True)
class GeometryMetrics:
    """Sample-relative geometry consistency metrics."""

    active_span_upper_body_bone_length_outlier_frame_ratio: float
    active_span_representative_hand_bone_length_outlier_frame_ratio: float
    active_span_cross_channel_scale_outlier_frame_ratio: float


@dataclass(frozen=True, slots=True)
class TextDiagnosticMetrics:
    """Text-side diagnostic density metrics."""

    token_count: int
    character_count: int
    non_whitespace_character_count: int
    frames_per_token: float | None
    frames_per_character: float | None
    frames_per_non_whitespace_character: float | None


@dataclass(frozen=True, slots=True)
class LengthDiagnosticMetrics:
    """Length-side diagnostic metrics."""

    frame_count: int
    duration_seconds: float
    frames_per_second_observed: float | None


@dataclass(frozen=True, slots=True)
class QualityMetricBundle:
    """Composed quality-family metrics for one sample."""

    oob: OobMetrics
    upper_body_support: UpperBodySupportMetrics
    manual_visibility: ManualVisibilityMetrics
    non_manual_visibility: NonManualVisibilityMetrics
    confidence: ConfidenceMetrics
    kinematic_naturalness: KinematicNaturalnessMetrics
    tracking_quality: TrackingQualityMetrics
    manual_detail: ManualDetailMetrics
    non_manual_quality: NonManualQualityMetrics
    geometry: GeometryMetrics
    text: TextDiagnosticMetrics
    length: LengthDiagnosticMetrics


__all__ = [
    "BindingQualityFamily",
    "ConfidenceMetrics",
    "DiagnosticQualityFamily",
    "GeometryMetrics",
    "KinematicNaturalnessMetrics",
    "LengthDiagnosticMetrics",
    "ManualDetailMetrics",
    "ManualVisibilityMetrics",
    "NonManualQualityMetrics",
    "NonManualVisibilityMetrics",
    "OobMetrics",
    "QualityMetricBuildInput",
    "QualityMetricBundle",
    "TextDiagnosticMetrics",
    "TrackingQualityMetrics",
    "UpperBodySupportMetrics",
]
