"""Canonical tier metric role registry with explicit veto/diagnostic ownership."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeAlias, cast

from text_to_sign_production.data.metrics.types import MetricBundle
from text_to_sign_production.data.tiers.types import (
    BindingTierFamily,
    DiagnosticMetric,
    DiagnosticTierCategory,
    FilterLevel,
    MetricPolicyRole,
    TierMetricFailure,
)

RoleFamily: TypeAlias = BindingTierFamily | DiagnosticTierCategory


# ---------------------------------------------------------------------------
# Role spec type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TierMetricRoleSpec:
    """One metric key's binding veto or report-only diagnostic role."""

    role: MetricPolicyRole
    family: RoleFamily
    metric_key: str
    metric_path: tuple[str, str]
    comparison: str | None = None
    threshold_attr: str | None = None
    reason_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, MetricPolicyRole):
            raise TypeError(f"role must be MetricPolicyRole, got {type(self.role)!r}.")
        if not isinstance(self.family, BindingTierFamily | DiagnosticTierCategory):
            raise TypeError(
                "family must be BindingTierFamily or DiagnosticTierCategory, got "
                f"{type(self.family)!r}."
            )


# ---------------------------------------------------------------------------
# Binding registry
# ---------------------------------------------------------------------------


BINDING_TIER_FAMILIES: tuple[BindingTierFamily, ...] = tuple(BindingTierFamily)

BINDING_TIER_METRICS: tuple[TierMetricRoleSpec, ...] = (
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.OOB,
        "out_of_bounds_ratio",
        ("oob", "out_of_bounds_ratio"),
        "<=",
        "max_out_of_bounds_ratio",
        "max_out_of_bounds_ratio_exceeded",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.UPPER_BODY_SUPPORT,
        "active_span_upper_body_support_landmark_coverage_ratio",
        ("upper_body_support", "active_span_upper_body_support_landmark_coverage_ratio"),
        ">=",
        "min_active_span_upper_body_support_landmark_coverage_ratio",
        "min_active_span_upper_body_support_landmark_coverage_ratio_not_met",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.MANUAL_VISIBILITY,
        "active_span_any_hand_available_frame_ratio",
        ("manual_visibility", "active_span_any_hand_available_frame_ratio"),
        ">=",
        "min_active_span_any_hand_available_frame_ratio",
        "min_active_span_any_hand_available_frame_ratio_not_met",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.MANUAL_VISIBILITY,
        "max_active_span_any_hand_unavailable_run_ratio",
        ("manual_visibility", "max_active_span_any_hand_unavailable_run_ratio"),
        "<=",
        "max_active_span_any_hand_unavailable_run_ratio",
        "max_active_span_any_hand_unavailable_run_ratio_exceeded",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.CONFIDENCE,
        "active_span_body_available_mean_confidence",
        ("confidence", "active_span_body_available_mean_confidence"),
        ">=",
        "min_active_span_body_available_mean_confidence",
        "min_active_span_body_available_mean_confidence_not_met",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.CONFIDENCE,
        "active_span_any_hand_available_mean_confidence",
        ("confidence", "active_span_any_hand_available_mean_confidence"),
        ">=",
        "min_active_span_any_hand_available_mean_confidence",
        "min_active_span_any_hand_available_mean_confidence_not_met",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.NON_MANUAL_VISIBILITY,
        "active_span_face_available_frame_ratio",
        ("non_manual_visibility", "active_span_face_available_frame_ratio"),
        ">=",
        "min_active_span_face_available_frame_ratio",
        "min_active_span_face_available_frame_ratio_not_met",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.NON_MANUAL_VISIBILITY,
        "max_active_span_face_unavailable_run_ratio",
        ("non_manual_visibility", "max_active_span_face_unavailable_run_ratio"),
        "<=",
        "max_active_span_face_unavailable_run_ratio",
        "max_active_span_face_unavailable_run_ratio_exceeded",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.KINEMATIC_NATURALNESS,
        "active_span_abrupt_motion_frame_ratio",
        ("kinematic_naturalness", "active_span_abrupt_motion_frame_ratio"),
        "<=",
        "max_active_span_abrupt_motion_frame_ratio",
        "max_active_span_abrupt_motion_frame_ratio_exceeded",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.KINEMATIC_NATURALNESS,
        "active_span_discontinuity_frame_ratio",
        ("kinematic_naturalness", "active_span_discontinuity_frame_ratio"),
        "<=",
        "max_active_span_discontinuity_frame_ratio",
        "max_active_span_discontinuity_frame_ratio_exceeded",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.KINEMATIC_NATURALNESS,
        "max_active_span_frozen_run_ratio",
        ("kinematic_naturalness", "max_active_span_frozen_run_ratio"),
        "<=",
        "max_active_span_frozen_run_ratio",
        "max_active_span_frozen_run_ratio_exceeded",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.TRACKING_QUALITY,
        "tracked_target_missing_frame_ratio",
        ("tracking_quality", "tracked_target_missing_frame_ratio"),
        "<=",
        "max_tracked_target_missing_frame_ratio",
        "max_tracked_target_missing_frame_ratio_exceeded",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.TRACKING_QUALITY,
        "person_tracking_continuity_break_ratio",
        ("tracking_quality", "person_tracking_continuity_break_ratio"),
        "<=",
        "max_person_tracking_continuity_break_ratio",
        "max_person_tracking_continuity_break_ratio_exceeded",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.TRACKING_QUALITY,
        "person_tracking_reanchor_ratio",
        ("tracking_quality", "person_tracking_reanchor_ratio"),
        "<=",
        "max_person_tracking_reanchor_ratio",
        "max_person_tracking_reanchor_ratio_exceeded",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.MANUAL_DETAIL,
        "active_span_representative_hand_landmark_coverage_ratio",
        ("manual_detail", "active_span_representative_hand_landmark_coverage_ratio"),
        ">=",
        "min_active_span_representative_hand_landmark_coverage_ratio",
        "min_active_span_representative_hand_landmark_coverage_ratio_not_met",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.MANUAL_DETAIL,
        "active_span_representative_hand_fingertip_coverage_ratio",
        ("manual_detail", "active_span_representative_hand_fingertip_coverage_ratio"),
        ">=",
        "min_active_span_representative_hand_fingertip_coverage_ratio",
        "min_active_span_representative_hand_fingertip_coverage_ratio_not_met",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.MANUAL_DETAIL,
        "active_span_representative_hand_distal_chain_coverage_ratio",
        ("manual_detail", "active_span_representative_hand_distal_chain_coverage_ratio"),
        ">=",
        "min_active_span_representative_hand_distal_chain_coverage_ratio",
        "min_active_span_representative_hand_distal_chain_coverage_ratio_not_met",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.MANUAL_DETAIL,
        "max_active_span_representative_hand_detail_dropout_run_ratio",
        ("manual_detail", "max_active_span_representative_hand_detail_dropout_run_ratio"),
        "<=",
        "max_active_span_representative_hand_detail_dropout_run_ratio",
        "max_active_span_representative_hand_detail_dropout_run_ratio_exceeded",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.NON_MANUAL_QUALITY,
        "active_span_face_landmark_coverage_ratio",
        ("non_manual_quality", "active_span_face_landmark_coverage_ratio"),
        ">=",
        "min_active_span_face_landmark_coverage_ratio",
        "min_active_span_face_landmark_coverage_ratio_not_met",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.NON_MANUAL_QUALITY,
        "active_span_upper_face_landmark_coverage_ratio",
        ("non_manual_quality", "active_span_upper_face_landmark_coverage_ratio"),
        ">=",
        "min_active_span_upper_face_landmark_coverage_ratio",
        "min_active_span_upper_face_landmark_coverage_ratio_not_met",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.NON_MANUAL_QUALITY,
        "active_span_lower_face_landmark_coverage_ratio",
        ("non_manual_quality", "active_span_lower_face_landmark_coverage_ratio"),
        ">=",
        "min_active_span_lower_face_landmark_coverage_ratio",
        "min_active_span_lower_face_landmark_coverage_ratio_not_met",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.NON_MANUAL_QUALITY,
        "max_active_span_face_detail_dropout_run_ratio",
        ("non_manual_quality", "max_active_span_face_detail_dropout_run_ratio"),
        "<=",
        "max_active_span_face_detail_dropout_run_ratio",
        "max_active_span_face_detail_dropout_run_ratio_exceeded",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.NON_MANUAL_QUALITY,
        "active_span_manual_face_overlap_frame_ratio",
        ("non_manual_quality", "active_span_manual_face_overlap_frame_ratio"),
        ">=",
        "min_active_span_manual_face_overlap_frame_ratio",
        "min_active_span_manual_face_overlap_frame_ratio_not_met",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.GEOMETRY,
        "active_span_upper_body_bone_length_outlier_frame_ratio",
        ("geometry", "active_span_upper_body_bone_length_outlier_frame_ratio"),
        "<=",
        "max_active_span_upper_body_bone_length_outlier_frame_ratio",
        "max_active_span_upper_body_bone_length_outlier_frame_ratio_exceeded",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.GEOMETRY,
        "active_span_representative_hand_bone_length_outlier_frame_ratio",
        ("geometry", "active_span_representative_hand_bone_length_outlier_frame_ratio"),
        "<=",
        "max_active_span_representative_hand_bone_length_outlier_frame_ratio",
        "max_active_span_representative_hand_bone_length_outlier_frame_ratio_exceeded",
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.BINDING,
        BindingTierFamily.GEOMETRY,
        "active_span_cross_channel_scale_outlier_frame_ratio",
        ("geometry", "active_span_cross_channel_scale_outlier_frame_ratio"),
        "<=",
        "max_active_span_cross_channel_scale_outlier_frame_ratio",
        "max_active_span_cross_channel_scale_outlier_frame_ratio_exceeded",
    ),
)


# ---------------------------------------------------------------------------
# Diagnostic registry
# ---------------------------------------------------------------------------


DIAGNOSTIC_TIER_METRICS: tuple[TierMetricRoleSpec, ...] = (
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.ACTIVE_SIGNING_SPAN,
        DiagnosticMetric.ACTIVE_SIGNING_SPAN_START_FRAME_INDEX.value,
        ("active_signing_span", "start_frame_index"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.ACTIVE_SIGNING_SPAN,
        DiagnosticMetric.ACTIVE_SIGNING_SPAN_END_FRAME_INDEX_EXCLUSIVE.value,
        ("active_signing_span", "end_frame_index_exclusive"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.ACTIVE_SIGNING_SPAN,
        DiagnosticMetric.ACTIVE_SIGNING_SPAN_FRAME_COUNT.value,
        ("active_signing_span", "frame_count"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.ACTIVE_SIGNING_SPAN,
        DiagnosticMetric.ACTIVE_SIGNING_SPAN_FRAME_RATIO.value,
        ("active_signing_span", "frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.ACTIVE_SIGNING_SPAN,
        DiagnosticMetric.TRIMMED_PREFIX_FRAME_COUNT.value,
        ("active_signing_span", "trimmed_prefix_frame_count"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.ACTIVE_SIGNING_SPAN,
        DiagnosticMetric.TRIMMED_PREFIX_FRAME_RATIO.value,
        ("active_signing_span", "trimmed_prefix_frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.ACTIVE_SIGNING_SPAN,
        DiagnosticMetric.TRIMMED_SUFFIX_FRAME_COUNT.value,
        ("active_signing_span", "trimmed_suffix_frame_count"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.ACTIVE_SIGNING_SPAN,
        DiagnosticMetric.TRIMMED_SUFFIX_FRAME_RATIO.value,
        ("active_signing_span", "trimmed_suffix_frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.ACTIVE_SIGNING_SPAN,
        DiagnosticMetric.SUSTAINED_ANY_HAND_EVIDENCE_FRAME_RATIO.value,
        ("active_signing_span", "sustained_any_hand_evidence_frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.ACTIVE_SIGNING_SPAN,
        DiagnosticMetric.SUSTAINED_UPPER_BODY_EVIDENCE_FRAME_RATIO.value,
        ("active_signing_span", "sustained_upper_body_evidence_frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.ACTIVE_SIGNING_SPAN,
        DiagnosticMetric.SUSTAINED_MOTION_EVIDENCE_FRAME_RATIO.value,
        ("active_signing_span", "sustained_motion_evidence_frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.UPPER_BODY_SUPPORT,
        DiagnosticMetric.FULL_CLIP_UPPER_BODY_SUPPORT_LANDMARK_COVERAGE_RATIO.value,
        ("upper_body_support", "full_clip_upper_body_support_landmark_coverage_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.COVERAGE,
        DiagnosticMetric.FULL_BODY_LANDMARK_COVERAGE_RATIO.value,
        ("coverage", "full_body_landmark_coverage_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.COVERAGE,
        DiagnosticMetric.LEFT_HAND_LANDMARK_COVERAGE_RATIO.value,
        ("coverage", "left_hand_landmark_coverage_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.COVERAGE,
        DiagnosticMetric.RIGHT_HAND_LANDMARK_COVERAGE_RATIO.value,
        ("coverage", "right_hand_landmark_coverage_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.COVERAGE,
        DiagnosticMetric.ANY_HAND_LANDMARK_COVERAGE_RATIO.value,
        ("coverage", "any_hand_landmark_coverage_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.COVERAGE,
        DiagnosticMetric.FACE_LANDMARK_COVERAGE_RATIO.value,
        ("coverage", "face_landmark_coverage_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.MANUAL_VISIBILITY,
        DiagnosticMetric.WHOLE_CLIP_LEFT_HAND_AVAILABLE_FRAME_RATIO.value,
        ("manual_visibility", "whole_clip_left_hand_available_frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.MANUAL_VISIBILITY,
        DiagnosticMetric.WHOLE_CLIP_RIGHT_HAND_AVAILABLE_FRAME_RATIO.value,
        ("manual_visibility", "whole_clip_right_hand_available_frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.MANUAL_VISIBILITY,
        DiagnosticMetric.WHOLE_CLIP_ANY_HAND_AVAILABLE_FRAME_RATIO.value,
        ("manual_visibility", "whole_clip_any_hand_available_frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.NON_MANUAL_VISIBILITY,
        DiagnosticMetric.FACE_AVAILABLE_FRAME_RATIO.value,
        ("non_manual_visibility", "face_available_frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.NON_MANUAL_VISIBILITY,
        DiagnosticMetric.FACE_UNAVAILABLE_FRAME_RATIO.value,
        ("non_manual_visibility", "face_unavailable_frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.CONFIDENCE,
        DiagnosticMetric.ACTIVE_SPAN_LEFT_HAND_AVAILABLE_MEAN_CONFIDENCE.value,
        ("confidence", "active_span_left_hand_available_mean_confidence"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.CONFIDENCE,
        DiagnosticMetric.ACTIVE_SPAN_RIGHT_HAND_AVAILABLE_MEAN_CONFIDENCE.value,
        ("confidence", "active_span_right_hand_available_mean_confidence"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.CONFIDENCE,
        DiagnosticMetric.FULL_CLIP_BODY_AVAILABLE_MEAN_CONFIDENCE.value,
        ("confidence", "full_clip_body_available_mean_confidence"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.CONFIDENCE,
        DiagnosticMetric.FACE_AVAILABLE_MEAN_CONFIDENCE.value,
        ("confidence", "face_available_mean_confidence"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.CONFIDENCE,
        DiagnosticMetric.OVERALL_AVAILABLE_MEAN_CONFIDENCE.value,
        ("confidence", "overall_available_mean_confidence"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.CONFIDENCE,
        DiagnosticMetric.BODY_NONZERO_CONFIDENCE_RATIO.value,
        ("confidence", "body_nonzero_confidence_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.CONFIDENCE,
        DiagnosticMetric.LEFT_HAND_NONZERO_CONFIDENCE_RATIO.value,
        ("confidence", "left_hand_nonzero_confidence_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.CONFIDENCE,
        DiagnosticMetric.RIGHT_HAND_NONZERO_CONFIDENCE_RATIO.value,
        ("confidence", "right_hand_nonzero_confidence_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.CONFIDENCE,
        DiagnosticMetric.FACE_NONZERO_CONFIDENCE_RATIO.value,
        ("confidence", "face_nonzero_confidence_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        BindingTierFamily.CONFIDENCE,
        DiagnosticMetric.OVERALL_NONZERO_CONFIDENCE_RATIO.value,
        ("confidence", "overall_nonzero_confidence_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.VALID,
        DiagnosticMetric.VALID_FRAME_COUNT.value,
        ("valid", "valid_frame_count"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.VALID,
        DiagnosticMetric.VALID_FRAME_RATIO.value,
        ("valid", "valid_frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.VALID,
        DiagnosticMetric.INVALID_FRAME_COUNT.value,
        ("valid", "invalid_frame_count"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.VALID,
        DiagnosticMetric.INVALID_FRAME_RATIO.value,
        ("valid", "invalid_frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.VALID,
        DiagnosticMetric.ZEROED_CANONICAL_JOINT_FRAME_COUNT.value,
        ("valid", "zeroed_canonical_joint_frame_count"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.VALID,
        DiagnosticMetric.ZEROED_CANONICAL_JOINT_FRAME_RATIO.value,
        ("valid", "zeroed_canonical_joint_frame_ratio"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.TEXT,
        DiagnosticMetric.CHARACTER_COUNT.value,
        ("text", "character_count"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.TEXT,
        DiagnosticMetric.TOKEN_COUNT.value,
        ("text", "token_count"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.LENGTH,
        DiagnosticMetric.NUM_FRAMES.value,
        ("length", "num_frames"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.LENGTH,
        DiagnosticMetric.DURATION_SECONDS.value,
        ("length", "duration_seconds"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.LENGTH,
        DiagnosticMetric.FRAMES_PER_TOKEN.value,
        ("length", "frames_per_token"),
    ),
    TierMetricRoleSpec(
        MetricPolicyRole.DIAGNOSTIC,
        DiagnosticTierCategory.LENGTH,
        DiagnosticMetric.FRAMES_PER_CHARACTER.value,
        ("length", "frames_per_character"),
    ),
)


def _binding_specs_for_family(family: BindingTierFamily) -> tuple[TierMetricRoleSpec, ...]:
    return tuple(spec for spec in BINDING_TIER_METRICS if spec.family is family)


def _binding_family(spec: TierMetricRoleSpec) -> BindingTierFamily:
    if not isinstance(spec.family, BindingTierFamily):
        raise TypeError(f"Binding spec has non-binding family {spec.family!r}.")
    return spec.family


# ---------------------------------------------------------------------------
# Derived indexes
# ---------------------------------------------------------------------------


BINDING_TIER_METRIC_KEYS_BY_FAMILY: Mapping[BindingTierFamily, frozenset[str]] = MappingProxyType(
    {
        family: frozenset(spec.metric_key for spec in _binding_specs_for_family(family))
        for family in BINDING_TIER_FAMILIES
    }
)

BINDING_TIER_METRICS_BY_FAMILY: Mapping[BindingTierFamily, tuple[TierMetricRoleSpec, ...]] = (
    MappingProxyType(
        {family: _binding_specs_for_family(family) for family in BINDING_TIER_FAMILIES}
    )
)


# ---------------------------------------------------------------------------
# Registry self-validation
# ---------------------------------------------------------------------------


def _assert_binding_specs_well_formed() -> None:
    seen_keys: set[tuple[BindingTierFamily, str]] = set()
    for spec in BINDING_TIER_METRICS:
        if spec.role != MetricPolicyRole.BINDING:
            raise RuntimeError(f"Binding registry contains non-binding spec: {spec!r}")
        family = _binding_family(spec)
        key = (family, spec.metric_key)
        if key in seen_keys:
            raise RuntimeError(
                f"Duplicate binding tier metric spec: {family.value}.{spec.metric_key}"
            )
        seen_keys.add(key)
        if spec.comparison not in (">=", "<="):
            raise RuntimeError(
                f"Binding spec has invalid comparison: {family.value}.{spec.metric_key}"
            )
        if spec.threshold_attr is None or spec.reason_code is None:
            raise RuntimeError(
                f"Binding spec is missing policy metadata: {family.value}.{spec.metric_key}"
            )


def assert_role_registry_complete() -> None:
    """Verify enums and binding role specs stay synchronized with this registry."""
    _assert_binding_specs_well_formed()
    missing = [
        family.value
        for family in BINDING_TIER_FAMILIES
        if not BINDING_TIER_METRICS_BY_FAMILY[family]
    ]
    if missing:
        raise RuntimeError(f"Binding families without metric role specs: {missing}")


# ---------------------------------------------------------------------------
# Read-only lookup helpers
# ---------------------------------------------------------------------------


def get_binding_metric_specs(family: BindingTierFamily) -> tuple[TierMetricRoleSpec, ...]:
    """Return immutable binding metric role specs for one family."""
    return BINDING_TIER_METRICS_BY_FAMILY[family]


def get_binding_metric_keys(family: BindingTierFamily) -> frozenset[str]:
    """Return immutable binding metric keys for one family."""
    return BINDING_TIER_METRIC_KEYS_BY_FAMILY[family]


# ---------------------------------------------------------------------------
# Metric / threshold access helpers
# ---------------------------------------------------------------------------


def get_metric_value(bundle: MetricBundle, spec: TierMetricRoleSpec) -> float | int | None:
    """Read a metric value using the canonical role registry path."""
    section_name, attr_name = spec.metric_path
    section: Any = getattr(bundle, section_name)
    return cast(float | int | None, getattr(section, attr_name))


def get_threshold_value(thresholds: object, spec: TierMetricRoleSpec) -> float | int:
    """Read the threshold value named by a binding role spec."""
    if spec.threshold_attr is None:
        raise ValueError(f"Diagnostic metric {spec.family}.{spec.metric_key} has no threshold.")
    return cast(float | int, getattr(thresholds, spec.threshold_attr))


# ---------------------------------------------------------------------------
# Generic evaluator
# ---------------------------------------------------------------------------


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

    return TierMetricFailure(
        family=_binding_family(spec),
        metric_key=spec.metric_key,
        reason_code=spec.reason_code,
        actual_value=actual_value,
        expected_value=expected_value,
        comparison=spec.comparison,
        applied_level=applied_level,
    )


assert_role_registry_complete()
