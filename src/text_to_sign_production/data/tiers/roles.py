"""Explicit binding and diagnostic tier metric roles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from text_to_sign_production.data.metrics.types import MetricBundle
from text_to_sign_production.data.tiers.types import (
    BindingTierFamily,
    DiagnosticMetric,
    FilterLevel,
    MetricPolicyRole,
    TierMetricFailure,
)


@dataclass(frozen=True, slots=True)
class TierMetricRoleSpec:
    """One metric key's policy role in tier decisions and calibration."""

    role: MetricPolicyRole
    family: str
    metric_key: str
    metric_path: tuple[str, str]
    comparison: str | None = None
    threshold_attr: str | None = None
    reason_code: str | None = None


BINDING_TIER_METRICS: tuple[TierMetricRoleSpec, ...] = (
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.OOB.value,
        "out_of_bounds_ratio",
        ("oob", "out_of_bounds_ratio"),
        "<=",
        "max_out_of_bounds_ratio",
        "max_out_of_bounds_ratio_exceeded",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.COVERAGE.value,
        "signing_relevant_body_landmark_coverage_ratio",
        ("coverage", "signing_relevant_body_landmark_coverage_ratio"),
        ">=",
        "min_signing_relevant_body_landmark_coverage_ratio",
        "min_signing_relevant_body_landmark_coverage_ratio_not_met",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.HAND.value,
        "active_window_any_hand_available_frame_ratio",
        ("hand", "active_window_any_hand_available_frame_ratio"),
        ">=",
        "min_active_window_any_hand_available_frame_ratio",
        "min_active_window_any_hand_available_frame_ratio_not_met",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.HAND.value,
        "max_active_window_any_hand_unavailable_run_ratio",
        ("hand", "max_active_window_any_hand_unavailable_run_ratio"),
        "<=",
        "max_active_window_any_hand_unavailable_run_ratio",
        "max_active_window_any_hand_unavailable_run_ratio_exceeded",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.FACE.value,
        "face_available_frame_ratio",
        ("face", "face_available_frame_ratio"),
        ">=",
        "min_face_available_frame_ratio",
        "min_face_available_frame_ratio_not_met",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.CONFIDENCE.value,
        "body_available_mean_confidence",
        ("confidence", "body_available_mean_confidence"),
        ">=",
        "min_body_available_mean_confidence",
        "min_body_available_mean_confidence_not_met",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.CONFIDENCE.value,
        "active_window_any_hand_available_mean_confidence",
        ("confidence", "active_window_any_hand_available_mean_confidence"),
        ">=",
        "min_active_window_any_hand_available_mean_confidence",
        "min_active_window_any_hand_available_mean_confidence_not_met",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.TEXT.value,
        "character_count",
        ("text", "character_count"),
        ">=",
        "min_character_count",
        "min_character_count_not_met",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.TEXT.value,
        "token_count",
        ("text", "token_count"),
        ">=",
        "min_token_count",
        "min_token_count_not_met",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.LENGTH.value,
        "num_frames",
        ("length", "num_frames"),
        ">=",
        "min_num_frames",
        "min_num_frames_not_met",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.LENGTH.value,
        "duration_seconds",
        ("length", "duration_seconds"),
        ">=",
        "min_duration_seconds",
        "min_duration_seconds_not_met",
    ),
)

DIAGNOSTIC_TIER_METRICS: tuple[TierMetricRoleSpec, ...] = (
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        "analysis_window",
        DiagnosticMetric.ANALYSIS_WINDOW_START_FRAME_INDEX.value,
        ("analysis_window", "start_frame_index"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        "analysis_window",
        DiagnosticMetric.ANALYSIS_WINDOW_END_FRAME_INDEX_EXCLUSIVE.value,
        ("analysis_window", "end_frame_index_exclusive"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        "analysis_window",
        DiagnosticMetric.ANALYSIS_WINDOW_FRAME_COUNT.value,
        ("analysis_window", "frame_count"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        "analysis_window",
        DiagnosticMetric.ANALYSIS_WINDOW_FRAME_RATIO.value,
        ("analysis_window", "frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        "analysis_window",
        DiagnosticMetric.PRE_SIGN_EXCLUDED_FRAME_RATIO.value,
        ("analysis_window", "pre_sign_excluded_frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        "analysis_window",
        DiagnosticMetric.POST_SIGN_EXCLUDED_FRAME_RATIO.value,
        ("analysis_window", "post_sign_excluded_frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        "analysis_window",
        DiagnosticMetric.SUSTAINED_HAND_EVIDENCE_FRAME_RATIO.value,
        ("analysis_window", "sustained_hand_evidence_frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.COVERAGE.value,
        DiagnosticMetric.FULL_BODY_LANDMARK_COVERAGE_RATIO.value,
        ("coverage", "full_body_landmark_coverage_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.COVERAGE.value,
        DiagnosticMetric.LEFT_HAND_LANDMARK_COVERAGE_RATIO.value,
        ("coverage", "left_hand_landmark_coverage_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.COVERAGE.value,
        DiagnosticMetric.RIGHT_HAND_LANDMARK_COVERAGE_RATIO.value,
        ("coverage", "right_hand_landmark_coverage_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.COVERAGE.value,
        DiagnosticMetric.ANY_HAND_LANDMARK_COVERAGE_RATIO.value,
        ("coverage", "any_hand_landmark_coverage_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.COVERAGE.value,
        DiagnosticMetric.FACE_LANDMARK_COVERAGE_RATIO.value,
        ("coverage", "face_landmark_coverage_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.HAND.value,
        DiagnosticMetric.WHOLE_CLIP_ANY_HAND_AVAILABLE_FRAME_RATIO.value,
        ("hand", "whole_clip_any_hand_available_frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.CONFIDENCE.value,
        DiagnosticMetric.ACTIVE_WINDOW_LEFT_HAND_AVAILABLE_MEAN_CONFIDENCE.value,
        ("confidence", "active_window_left_hand_available_mean_confidence"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.CONFIDENCE.value,
        DiagnosticMetric.ACTIVE_WINDOW_RIGHT_HAND_AVAILABLE_MEAN_CONFIDENCE.value,
        ("confidence", "active_window_right_hand_available_mean_confidence"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.CONFIDENCE.value,
        DiagnosticMetric.FACE_AVAILABLE_MEAN_CONFIDENCE.value,
        ("confidence", "face_available_mean_confidence"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.CONFIDENCE.value,
        DiagnosticMetric.OVERALL_AVAILABLE_MEAN_CONFIDENCE.value,
        ("confidence", "overall_available_mean_confidence"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        "valid",
        DiagnosticMetric.VALID_FRAME_RATIO.value,
        ("valid", "valid_frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        "valid",
        DiagnosticMetric.INVALID_FRAME_RATIO.value,
        ("valid", "invalid_frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        "valid",
        DiagnosticMetric.ZEROED_CANONICAL_JOINT_FRAME_RATIO.value,
        ("valid", "zeroed_canonical_joint_frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.LENGTH.value,
        DiagnosticMetric.FRAMES_PER_TOKEN.value,
        ("length", "frames_per_token"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.LENGTH.value,
        DiagnosticMetric.FRAMES_PER_CHARACTER.value,
        ("length", "frames_per_character"),
    ),
)

BINDING_TIER_METRIC_KEYS_BY_FAMILY: dict[BindingTierFamily, frozenset[str]] = {
    family: frozenset(
        spec.metric_key
        for spec in BINDING_TIER_METRICS
        if spec.family == family.value
    )
    for family in BindingTierFamily
}

BINDING_TIER_METRICS_BY_FAMILY: dict[BindingTierFamily, tuple[TierMetricRoleSpec, ...]] = {
    family: tuple(
        spec
        for spec in BINDING_TIER_METRICS
        if spec.family == family.value
    )
    for family in BindingTierFamily
}


def get_metric_value(bundle: MetricBundle, spec: TierMetricRoleSpec) -> float | int | None:
    """Read a metric value using the canonical role registry path."""
    section_name, attr_name = spec.metric_path
    section: Any = getattr(bundle, section_name)
    return getattr(section, attr_name)


def get_threshold_value(thresholds: object, spec: TierMetricRoleSpec) -> float | int:
    """Read the threshold value named by a binding role spec."""
    if spec.threshold_attr is None:
        raise ValueError(f"Diagnostic metric {spec.family}.{spec.metric_key} has no threshold.")
    return getattr(thresholds, spec.threshold_attr)


def evaluate_binding_metric(
    bundle: MetricBundle,
    thresholds: object,
    spec: TierMetricRoleSpec,
    applied_level: FilterLevel,
) -> TierMetricFailure | None:
    """Evaluate a binding metric using canonical role metadata."""
    if spec.comparison is None or spec.reason_code is None:
        raise ValueError(f"Metric {spec.family}.{spec.metric_key} is not a binding metric.")

    actual_value = get_metric_value(bundle, spec)
    expected_value = get_threshold_value(thresholds, spec)
    passed = False
    if actual_value is not None:
        if spec.comparison == ">=":
            passed = actual_value >= expected_value
        elif spec.comparison == "<=":
            passed = actual_value <= expected_value
        else:
            raise ValueError(f"Unsupported comparison {spec.comparison!r}.")
    if passed:
        return None

    reason_code = spec.reason_code
    if actual_value is None and spec.metric_key == "duration_seconds":
        reason_code = "duration_seconds_missing"

    return TierMetricFailure(
        family=BindingTierFamily(spec.family),
        metric_key=spec.metric_key,
        reason_code=reason_code,
        actual_value=actual_value,
        expected_value=expected_value,
        comparison=spec.comparison,
        applied_level=applied_level,
    )
