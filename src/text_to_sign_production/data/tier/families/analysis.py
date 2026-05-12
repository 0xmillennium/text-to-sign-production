"""Read-only typed analysis helpers for quality-family metrics."""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sign_production.data.tier.families.types import (
    BindingQualityFamily,
    DiagnosticQualityFamily,
    QualityMetricBundle,
)


@dataclass(frozen=True, slots=True)
class MetricValueSummary:
    """One observed numeric metric value."""

    name: str
    value: float


@dataclass(frozen=True, slots=True)
class FamilyMetricSummary:
    """Typed read-only metric values for one family."""

    family: BindingQualityFamily | DiagnosticQualityFamily
    metrics: tuple[MetricValueSummary, ...]

    @property
    def metric_count(self) -> int:
        """Number of numeric metrics observed for this family."""
        return len(self.metrics)


@dataclass(frozen=True, slots=True)
class QualityMetricBundleSummary:
    """Typed read-only inventory of metric-family coverage."""

    binding_families: tuple[FamilyMetricSummary, ...]
    diagnostic_families: tuple[FamilyMetricSummary, ...]

    @property
    def binding_family_count(self) -> int:
        """Number of binding metric families in the bundle."""
        return len(self.binding_families)

    @property
    def diagnostic_family_count(self) -> int:
        """Number of diagnostic metric families in the bundle."""
        return len(self.diagnostic_families)

    @property
    def binding_metric_count(self) -> int:
        """Number of binding metrics in the bundle."""
        return sum(summary.metric_count for summary in self.binding_families)

    @property
    def diagnostic_metric_count(self) -> int:
        """Number of diagnostic metrics in the bundle."""
        return sum(summary.metric_count for summary in self.diagnostic_families)

    @property
    def metric_count(self) -> int:
        """Total numeric metric count across binding and diagnostic families."""
        return self.binding_metric_count + self.diagnostic_metric_count


def binding_metric_summary(
    bundle: QualityMetricBundle,
) -> tuple[FamilyMetricSummary, ...]:
    """Return typed binding-family metric values."""
    return (
        FamilyMetricSummary(
            BindingQualityFamily.OOB,
            (
                _metric(
                    "body_out_of_bounds_frame_ratio", bundle.oob.body_out_of_bounds_frame_ratio
                ),
                _metric(
                    "left_hand_out_of_bounds_frame_ratio",
                    bundle.oob.left_hand_out_of_bounds_frame_ratio,
                ),
                _metric(
                    "right_hand_out_of_bounds_frame_ratio",
                    bundle.oob.right_hand_out_of_bounds_frame_ratio,
                ),
                _metric(
                    "face_out_of_bounds_frame_ratio", bundle.oob.face_out_of_bounds_frame_ratio
                ),
                _metric(
                    "max_channel_out_of_bounds_frame_ratio",
                    bundle.oob.max_channel_out_of_bounds_frame_ratio,
                ),
            ),
        ),
        FamilyMetricSummary(
            BindingQualityFamily.UPPER_BODY_SUPPORT,
            (
                _metric(
                    "active_span_upper_body_available_frame_ratio",
                    bundle.upper_body_support.active_span_upper_body_available_frame_ratio,
                ),
                _metric(
                    "active_span_upper_body_landmark_coverage_ratio",
                    bundle.upper_body_support.active_span_upper_body_landmark_coverage_ratio,
                ),
                _metric(
                    "max_active_span_upper_body_dropout_run_ratio",
                    bundle.upper_body_support.max_active_span_upper_body_dropout_run_ratio,
                ),
            ),
        ),
        FamilyMetricSummary(
            BindingQualityFamily.MANUAL_VISIBILITY,
            (
                _metric(
                    "active_span_any_hand_available_frame_ratio",
                    bundle.manual_visibility.active_span_any_hand_available_frame_ratio,
                ),
                _metric(
                    "active_span_both_hands_unavailable_frame_ratio",
                    bundle.manual_visibility.active_span_both_hands_unavailable_frame_ratio,
                ),
                _metric(
                    "max_active_span_any_hand_dropout_run_ratio",
                    bundle.manual_visibility.max_active_span_any_hand_dropout_run_ratio,
                ),
            ),
        ),
        FamilyMetricSummary(
            BindingQualityFamily.NON_MANUAL_VISIBILITY,
            (
                _metric(
                    "active_span_face_available_frame_ratio",
                    bundle.non_manual_visibility.active_span_face_available_frame_ratio,
                ),
                _metric(
                    "max_active_span_face_unavailable_run_ratio",
                    bundle.non_manual_visibility.max_active_span_face_unavailable_run_ratio,
                ),
            ),
        ),
        FamilyMetricSummary(
            BindingQualityFamily.CONFIDENCE,
            (
                _metric(
                    "active_span_body_observed_mean_confidence",
                    bundle.confidence.active_span_body_observed_mean_confidence,
                ),
                _metric(
                    "active_span_hand_observed_mean_confidence",
                    bundle.confidence.active_span_hand_observed_mean_confidence,
                ),
                _metric(
                    "active_span_face_observed_mean_confidence",
                    bundle.confidence.active_span_face_observed_mean_confidence,
                ),
            ),
        ),
        FamilyMetricSummary(
            BindingQualityFamily.KINEMATIC_NATURALNESS,
            (
                _metric(
                    "comparable_transition_abrupt_ratio",
                    bundle.kinematic_naturalness.comparable_transition_abrupt_ratio,
                ),
                _metric(
                    "comparable_transition_discontinuity_ratio",
                    bundle.kinematic_naturalness.comparable_transition_discontinuity_ratio,
                ),
                _metric(
                    "active_span_frozen_frame_ratio",
                    bundle.kinematic_naturalness.active_span_frozen_frame_ratio,
                ),
                _metric(
                    "max_active_span_frozen_run_ratio",
                    bundle.kinematic_naturalness.max_active_span_frozen_run_ratio,
                ),
            ),
        ),
        FamilyMetricSummary(
            BindingQualityFamily.TRACKING_QUALITY,
            (
                _metric(
                    "tracked_target_missing_frame_ratio",
                    bundle.tracking_quality.tracked_target_missing_frame_ratio,
                ),
                _metric(
                    "person_tracking_continuity_break_ratio",
                    bundle.tracking_quality.person_tracking_continuity_break_ratio,
                ),
                _metric(
                    "person_tracking_reanchor_ratio",
                    bundle.tracking_quality.person_tracking_reanchor_ratio,
                ),
            ),
        ),
        FamilyMetricSummary(
            BindingQualityFamily.MANUAL_DETAIL,
            (
                _metric(
                    "active_span_representative_hand_landmark_coverage_ratio",
                    bundle.manual_detail.active_span_representative_hand_landmark_coverage_ratio,
                ),
                _metric(
                    "active_span_representative_hand_fingertip_coverage_ratio",
                    bundle.manual_detail.active_span_representative_hand_fingertip_coverage_ratio,
                ),
                _metric(
                    "active_span_representative_hand_distal_chain_coverage_ratio",
                    bundle.manual_detail.active_span_representative_hand_distal_chain_coverage_ratio,
                ),
                _metric(
                    "max_active_span_representative_hand_detail_dropout_run_ratio",
                    bundle.manual_detail.max_active_span_representative_hand_detail_dropout_run_ratio,
                ),
            ),
        ),
        FamilyMetricSummary(
            BindingQualityFamily.NON_MANUAL_QUALITY,
            (
                _metric(
                    "active_span_face_landmark_coverage_ratio",
                    bundle.non_manual_quality.active_span_face_landmark_coverage_ratio,
                ),
                _metric(
                    "active_span_upper_face_landmark_coverage_ratio",
                    bundle.non_manual_quality.active_span_upper_face_landmark_coverage_ratio,
                ),
                _metric(
                    "active_span_lower_face_landmark_coverage_ratio",
                    bundle.non_manual_quality.active_span_lower_face_landmark_coverage_ratio,
                ),
                _metric(
                    "max_active_span_face_detail_dropout_run_ratio",
                    bundle.non_manual_quality.max_active_span_face_detail_dropout_run_ratio,
                ),
                _metric(
                    "active_span_face_available_given_manual_frame_ratio",
                    bundle.non_manual_quality.active_span_face_available_given_manual_frame_ratio,
                ),
            ),
        ),
        FamilyMetricSummary(
            BindingQualityFamily.GEOMETRY,
            (
                _metric(
                    "active_span_upper_body_bone_length_outlier_frame_ratio",
                    bundle.geometry.active_span_upper_body_bone_length_outlier_frame_ratio,
                ),
                _metric(
                    "active_span_cross_channel_scale_outlier_frame_ratio",
                    bundle.geometry.active_span_cross_channel_scale_outlier_frame_ratio,
                ),
            ),
        ),
    )


def diagnostic_metric_summary(
    bundle: QualityMetricBundle,
) -> tuple[FamilyMetricSummary, ...]:
    """Return typed diagnostic-family metric values."""
    return (
        FamilyMetricSummary(
            DiagnosticQualityFamily.TEXT,
            _optional_metrics(
                _metric("token_count", bundle.text.token_count),
                _metric("character_count", bundle.text.character_count),
                _metric(
                    "non_whitespace_character_count",
                    bundle.text.non_whitespace_character_count,
                ),
                _optional_metric("frames_per_token", bundle.text.frames_per_token),
                _optional_metric("frames_per_character", bundle.text.frames_per_character),
                _optional_metric(
                    "frames_per_non_whitespace_character",
                    bundle.text.frames_per_non_whitespace_character,
                ),
            ),
        ),
        FamilyMetricSummary(
            DiagnosticQualityFamily.LENGTH,
            _optional_metrics(
                _metric("frame_count", bundle.length.frame_count),
                _metric("duration_seconds", bundle.length.duration_seconds),
                _optional_metric(
                    "frames_per_second_observed",
                    bundle.length.frames_per_second_observed,
                ),
            ),
        ),
    )


def summarize_quality_metric_bundle(bundle: QualityMetricBundle) -> QualityMetricBundleSummary:
    """Return a compact, read-only metric bundle inventory."""
    binding = binding_metric_summary(bundle)
    diagnostic = diagnostic_metric_summary(bundle)
    return QualityMetricBundleSummary(
        binding_families=binding,
        diagnostic_families=diagnostic,
    )


def _metric(name: str, value: int | float) -> MetricValueSummary:
    return MetricValueSummary(name=name, value=float(value))


def _optional_metric(name: str, value: int | float | None) -> MetricValueSummary | None:
    if value is None:
        return None
    return _metric(name, value)


def _optional_metrics(
    *metrics: MetricValueSummary | None,
) -> tuple[MetricValueSummary, ...]:
    return tuple(metric for metric in metrics if metric is not None)


__all__ = [
    "FamilyMetricSummary",
    "MetricValueSummary",
    "QualityMetricBundleSummary",
    "binding_metric_summary",
    "diagnostic_metric_summary",
    "summarize_quality_metric_bundle",
]
