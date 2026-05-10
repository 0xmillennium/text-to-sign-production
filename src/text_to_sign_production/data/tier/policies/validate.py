"""Tier-domain validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields

from text_to_sign_production.core.ids import SampleStatus, TierName
from text_to_sign_production.data.tier.families import (
    BindingQualityFamily,
    DiagnosticQualityFamily,
)
from text_to_sign_production.data.tier.policies.gates import GateDecisionBundle
from text_to_sign_production.data.tier.policies.filters import TierFiltersConfig
from text_to_sign_production.data.tier.policies.policies import TierPoliciesConfig
from text_to_sign_production.data.tier.policies.roles import BINDING_TIER_FAMILIES
from text_to_sign_production.data.tier.policies.types import (
    TierDecisionBundle,
    TierStatus,
    TierValidationIssue,
    TierValidationIssueCode,
)


def validate_tier_filters_config(config: TierFiltersConfig) -> tuple[TierValidationIssue, ...]:
    """Validate tier filter shape and monotonic threshold strictness."""
    issues: list[TierValidationIssue] = []
    for config_field in fields(config):
        section = getattr(config, config_field.name)
        levels = set(section.thresholds_by_level)
        expected_levels = set(TierName)
        if levels != expected_levels:
            issues.append(
                TierValidationIssue(
                    code=TierValidationIssueCode.INVALID_FILTER_CONFIG,
                    message="Tier filter section must define every tier level exactly once.",
                    field_path=config_field.name,
                )
            )
        for level, thresholds in section.thresholds_by_level.items():
            for threshold_field in fields(thresholds):
                value = getattr(thresholds, threshold_field.name)
                path = f"{config_field.name}.{level.value}.{threshold_field.name}"
                if not isinstance(value, int | float) or isinstance(value, bool):
                    issues.append(
                        TierValidationIssue(
                            code=TierValidationIssueCode.INVALID_THRESHOLD,
                            message="Tier threshold must be numeric.",
                            field_path=path,
                        )
                    )
                elif not 0.0 <= float(value) <= 1.0:
                    issues.append(
                        TierValidationIssue(
                            code=TierValidationIssueCode.INVALID_THRESHOLD,
                            message="Tier threshold ratio must be within [0, 1].",
                            field_path=path,
                        )
                    )
    issues.extend(_validate_monotonicity(config))
    return tuple(issues)


def validate_tier_policies_config(config: TierPoliciesConfig) -> tuple[TierValidationIssue, ...]:
    """Validate tier policy consistency."""
    issues: list[TierValidationIssue] = []
    expected_tiers = set(TierName)
    if set(config.policies_by_tier) != expected_tiers:
        issues.append(
            TierValidationIssue(
                code=TierValidationIssueCode.INVALID_POLICY_CONFIG,
                message="Tier policies must define every tier exactly once.",
                field_path="policies_by_tier",
            )
        )
    if tuple(config.tier_order) != tuple(TierName):
        issues.append(
            TierValidationIssue(
                code=TierValidationIssueCode.INVALID_POLICY_CONFIG,
                message="Tier order must be loose, clean, tight.",
                field_path="tier_order",
            )
        )
    expected_families = set(BINDING_TIER_FAMILIES)
    for tier, policy in config.policies_by_tier.items():
        if policy.tier_name != tier:
            issues.append(
                TierValidationIssue(
                    code=TierValidationIssueCode.INVALID_POLICY_CONFIG,
                    message="Tier policy key and policy name must match.",
                    field_path=f"policies_by_tier.{tier.value}.tier_name",
                )
            )
        family_keys = set(policy.family_filter_levels)
        if family_keys != expected_families:
            issues.append(
                TierValidationIssue(
                    code=TierValidationIssueCode.MISSING_FAMILY_POLICY,
                    message="Tier policy must define every binding family exactly once.",
                    field_path=f"policies_by_tier.{tier.value}.family_filter_levels",
                )
            )
        for family, level in policy.family_filter_levels.items():
            if isinstance(family, DiagnosticQualityFamily):
                issues.append(
                    TierValidationIssue(
                        code=TierValidationIssueCode.DIAGNOSTIC_FAMILY_BOUND,
                        message="Diagnostic families must not participate in tier policy.",
                        field_path=f"policies_by_tier.{tier.value}.family_filter_levels",
                    )
                )
            if not isinstance(family, BindingQualityFamily):
                issues.append(
                    TierValidationIssue(
                        code=TierValidationIssueCode.INVALID_POLICY_CONFIG,
                        message="Tier policy family key must be a binding quality family.",
                        field_path=f"policies_by_tier.{tier.value}.family_filter_levels",
                    )
                )
            if not isinstance(level, TierName):
                issues.append(
                    TierValidationIssue(
                        code=TierValidationIssueCode.INVALID_POLICY_CONFIG,
                        message="Tier policy family level must be a known tier level.",
                        field_path=f"policies_by_tier.{tier.value}.{family}",
                    )
                )
    return tuple(issues)


def validate_tier_decision_bundle(
    bundle: TierDecisionBundle,
    gate_decisions: GateDecisionBundle | None = None,
) -> tuple[TierValidationIssue, ...]:
    """Validate structural consistency of a tier decision bundle."""
    issues: list[TierValidationIssue] = []
    if (
        gate_decisions is not None
        and gate_decisions.final_status is not SampleStatus.PASSED
        and bundle.selected_tier is not None
    ):
        issues.append(
            TierValidationIssue(
                code=TierValidationIssueCode.GATE_FAILED_TIER_SELECTED,
                message="Gate-failed samples must not receive a selected tier.",
                field_path="selected_tier",
            )
        )
    if bundle.status is TierStatus.SKIPPED:
        if bundle.selected_tier is not None:
            issues.append(
                TierValidationIssue(
                    code=TierValidationIssueCode.INVALID_DECISION_BUNDLE,
                    message="Skipped tier bundles must not have a selected tier.",
                    field_path="selected_tier",
                )
            )
        return tuple(issues)
    if bundle.status is TierStatus.PASSED and bundle.selected_tier is None:
        issues.append(
            TierValidationIssue(
                code=TierValidationIssueCode.INVALID_DECISION_BUNDLE,
                message="Passed tier bundles must have a selected tier.",
                field_path="selected_tier",
            )
        )
    if bundle.status is TierStatus.FAILED and bundle.selected_tier is not None:
        issues.append(
            TierValidationIssue(
                code=TierValidationIssueCode.INVALID_DECISION_BUNDLE,
                message="Failed tier bundles must not have a selected tier.",
                field_path="selected_tier",
            )
        )

    families = tuple(decision.family for decision in bundle.family_decisions)
    if len(set(families)) != len(families):
        issues.append(
            TierValidationIssue(
                code=TierValidationIssueCode.INVALID_DECISION_BUNDLE,
                message="Tier bundle contains duplicate family decisions.",
                field_path="family_decisions",
            )
        )
    if set(families) != set(BINDING_TIER_FAMILIES):
        issues.append(
            TierValidationIssue(
                code=TierValidationIssueCode.INVALID_DECISION_BUNDLE,
                message="Non-skipped tier bundle must contain exactly the binding families.",
                field_path="family_decisions",
            )
        )
    if bundle.selected_tier is not None:
        for decision in bundle.family_decisions:
            if bundle.selected_tier not in decision.supported_tiers:
                issues.append(
                    TierValidationIssue(
                        code=TierValidationIssueCode.SELECTED_TIER_CONTRADICTION,
                        message="Selected tier is not supported by a binding family decision.",
                        field_path=f"family_decisions.{decision.family.value}",
                    )
                )
    for decision in bundle.family_decisions:
        if decision.best_supported_tier is not None and (
            decision.best_supported_tier not in decision.supported_tiers
        ):
            issues.append(
                TierValidationIssue(
                    code=TierValidationIssueCode.INVALID_DECISION_BUNDLE,
                    message="Family best-supported tier must appear in supported_tiers.",
                    field_path=f"family_decisions.{decision.family.value}.best_supported_tier",
                )
            )
    return tuple(issues)


def _validate_monotonicity(config: TierFiltersConfig) -> tuple[TierValidationIssue, ...]:
    checks = (
        (config.oob.thresholds_by_level, "max_out_of_bounds_ratio", "nonincreasing", "oob"),
        (
            config.upper_body_support.thresholds_by_level,
            "min_active_span_upper_body_support_landmark_coverage_ratio",
            "nondecreasing",
            "upper_body_support",
        ),
        (
            config.manual_visibility.thresholds_by_level,
            "min_active_span_any_hand_available_frame_ratio",
            "nondecreasing",
            "manual_visibility",
        ),
        (
            config.manual_visibility.thresholds_by_level,
            "max_active_span_any_hand_unavailable_run_ratio",
            "nonincreasing",
            "manual_visibility",
        ),
        (
            config.non_manual_visibility.thresholds_by_level,
            "min_active_span_face_available_frame_ratio",
            "nondecreasing",
            "non_manual_visibility",
        ),
        (
            config.non_manual_visibility.thresholds_by_level,
            "max_active_span_face_unavailable_run_ratio",
            "nonincreasing",
            "non_manual_visibility",
        ),
        (
            config.confidence.thresholds_by_level,
            "min_active_span_body_available_mean_confidence",
            "nondecreasing",
            "confidence",
        ),
        (
            config.confidence.thresholds_by_level,
            "min_active_span_any_hand_available_mean_confidence",
            "nondecreasing",
            "confidence",
        ),
        (
            config.kinematic_naturalness.thresholds_by_level,
            "max_active_span_abrupt_motion_frame_ratio",
            "nonincreasing",
            "kinematic_naturalness",
        ),
        (
            config.kinematic_naturalness.thresholds_by_level,
            "max_active_span_discontinuity_frame_ratio",
            "nonincreasing",
            "kinematic_naturalness",
        ),
        (
            config.kinematic_naturalness.thresholds_by_level,
            "max_active_span_frozen_run_ratio",
            "nonincreasing",
            "kinematic_naturalness",
        ),
        (
            config.tracking_quality.thresholds_by_level,
            "max_tracked_target_missing_frame_ratio",
            "nonincreasing",
            "tracking_quality",
        ),
        (
            config.tracking_quality.thresholds_by_level,
            "max_person_tracking_continuity_break_ratio",
            "nonincreasing",
            "tracking_quality",
        ),
        (
            config.tracking_quality.thresholds_by_level,
            "max_person_tracking_reanchor_ratio",
            "nonincreasing",
            "tracking_quality",
        ),
        (
            config.manual_detail.thresholds_by_level,
            "min_active_span_representative_hand_landmark_coverage_ratio",
            "nondecreasing",
            "manual_detail",
        ),
        (
            config.manual_detail.thresholds_by_level,
            "min_active_span_representative_hand_fingertip_coverage_ratio",
            "nondecreasing",
            "manual_detail",
        ),
        (
            config.manual_detail.thresholds_by_level,
            "min_active_span_representative_hand_distal_chain_coverage_ratio",
            "nondecreasing",
            "manual_detail",
        ),
        (
            config.manual_detail.thresholds_by_level,
            "max_active_span_representative_hand_detail_dropout_run_ratio",
            "nonincreasing",
            "manual_detail",
        ),
        (
            config.non_manual_quality.thresholds_by_level,
            "min_active_span_face_landmark_coverage_ratio",
            "nondecreasing",
            "non_manual_quality",
        ),
        (
            config.non_manual_quality.thresholds_by_level,
            "min_active_span_upper_face_landmark_coverage_ratio",
            "nondecreasing",
            "non_manual_quality",
        ),
        (
            config.non_manual_quality.thresholds_by_level,
            "min_active_span_lower_face_landmark_coverage_ratio",
            "nondecreasing",
            "non_manual_quality",
        ),
        (
            config.non_manual_quality.thresholds_by_level,
            "max_active_span_face_detail_dropout_run_ratio",
            "nonincreasing",
            "non_manual_quality",
        ),
        (
            config.non_manual_quality.thresholds_by_level,
            "min_active_span_manual_face_overlap_frame_ratio",
            "nondecreasing",
            "non_manual_quality",
        ),
        (
            config.geometry.thresholds_by_level,
            "max_active_span_upper_body_bone_length_outlier_frame_ratio",
            "nonincreasing",
            "geometry",
        ),
        (
            config.geometry.thresholds_by_level,
            "max_active_span_representative_hand_bone_length_outlier_frame_ratio",
            "nonincreasing",
            "geometry",
        ),
        (
            config.geometry.thresholds_by_level,
            "max_active_span_cross_channel_scale_outlier_frame_ratio",
            "nonincreasing",
            "geometry",
        ),
    )
    issues: list[TierValidationIssue] = []
    for thresholds_by_level, attr_name, direction, family_name in checks:
        values = _ordered_values(thresholds_by_level, attr_name)
        expected = sorted(values) if direction == "nondecreasing" else sorted(values, reverse=True)
        if values != expected:
            issues.append(
                TierValidationIssue(
                    code=TierValidationIssueCode.NON_MONOTONIC_THRESHOLD,
                    message=(
                        "Tier thresholds must be monotonic across loose, clean, tight "
                        f"({direction})."
                    ),
                    field_path=f"{family_name}.{attr_name}",
                )
            )
    return tuple(issues)


def _ordered_values(
    thresholds_by_level: Mapping[TierName, object],
    attr_name: str,
) -> list[float]:
    return [float(getattr(thresholds_by_level[tier], attr_name)) for tier in TierName]


__all__ = [
    "validate_tier_decision_bundle",
    "validate_tier_filters_config",
    "validate_tier_policies_config",
]
