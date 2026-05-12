"""Tier-domain validation."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Literal

from text_to_sign_production.core.ids import TierName
from text_to_sign_production.core.models import CheckpointAdmission
from text_to_sign_production.data.tier.families.types import (
    BindingQualityFamily,
    DiagnosticQualityFamily,
)
from text_to_sign_production.data.tier.policies.filters import TierFiltersConfig
from text_to_sign_production.data.tier.policies.policies import TierPoliciesConfig
from text_to_sign_production.data.tier.policies.roles import BINDING_TIER_FAMILIES
from text_to_sign_production.data.tier.policies.types import (
    TierDecisionBundle,
    TierStatus,
    TierValidationIssue,
    TierValidationIssueCode,
)

MonotonicDirection = Literal["nondecreasing", "nonincreasing"]


@dataclass(frozen=True, slots=True)
class MonotonicThresholdCheck:
    """One auditable monotonicity rule for a tier filter threshold."""

    thresholds_by_level: Mapping[TierName, object]
    attr_name: str
    direction: MonotonicDirection
    family_name: str


# Public validators


def validate_tier_filters_config(config: TierFiltersConfig) -> tuple[TierValidationIssue, ...]:
    """Validate tier filter shape and monotonic threshold strictness."""
    return (*_validate_filter_shape(config), *_validate_monotonicity(config))


def validate_tier_policies_config(config: TierPoliciesConfig) -> tuple[TierValidationIssue, ...]:
    """Validate tier policy consistency."""
    issues: list[TierValidationIssue] = []
    issues.extend(_validate_policy_tier_coverage(config))
    issues.extend(_validate_policy_family_bindings(config))
    return tuple(issues)


def validate_tier_decision_bundle(
    bundle: TierDecisionBundle,
    checkpoint_admission: CheckpointAdmission,
) -> tuple[TierValidationIssue, ...]:
    """Validate structural consistency of a checkpoint-admitted tier decision bundle."""
    issues: list[TierValidationIssue] = []
    issues.extend(_validate_checkpoint_admission(checkpoint_admission))
    issues.extend(_validate_decision_status(bundle))
    issues.extend(_validate_leakage_decision(bundle))
    if bundle.status is not TierStatus.SKIPPED:
        issues.extend(_validate_family_decisions(bundle))
    return tuple(issues)


# Config shape


def _validate_filter_shape(config: TierFiltersConfig) -> tuple[TierValidationIssue, ...]:
    """Validate filter section coverage and scalar threshold ranges."""
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
    return tuple(issues)


# Policy-family binding


def _validate_policy_tier_coverage(
    config: TierPoliciesConfig,
) -> tuple[TierValidationIssue, ...]:
    """Validate top-level tier policy coverage and ordering."""
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
    for tier, policy in config.policies_by_tier.items():
        if policy.tier_name != tier:
            issues.append(
                TierValidationIssue(
                    code=TierValidationIssueCode.INVALID_POLICY_CONFIG,
                    message="Tier policy key and policy name must match.",
                    field_path=f"policies_by_tier.{tier.value}.tier_name",
                )
            )
    return tuple(issues)


def _validate_policy_family_bindings(
    config: TierPoliciesConfig,
) -> tuple[TierValidationIssue, ...]:
    """Validate binding-family policy mappings inside each tier policy."""
    issues: list[TierValidationIssue] = []
    expected_families = set(BINDING_TIER_FAMILIES)
    for tier, policy in config.policies_by_tier.items():
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


# Checkpoint admission and decision validation


def _validate_checkpoint_admission(
    checkpoint_admission: CheckpointAdmission,
) -> tuple[TierValidationIssue, ...]:
    """Validate the admission fact available to checkpoint-only tier flow."""
    issues: list[TierValidationIssue] = []
    if not checkpoint_admission.sample_id.strip():
        issues.append(
            TierValidationIssue(
                code=TierValidationIssueCode.INVALID_CHECKPOINT_ADMISSION,
                message="Checkpoint admission sample_id must be non-empty.",
                field_path="checkpoint_admission.sample_id",
            ),
        )
    if checkpoint_admission.source_manifest_path is None:
        issues.append(
            TierValidationIssue(
                code=TierValidationIssueCode.INVALID_CHECKPOINT_ADMISSION,
                message="Checkpoint admission must carry source_manifest_path.",
                field_path="checkpoint_admission.source_manifest_path",
            )
        )
    if not checkpoint_admission.source_manifest_sha256:
        issues.append(
            TierValidationIssue(
                code=TierValidationIssueCode.INVALID_CHECKPOINT_ADMISSION,
                message="Checkpoint admission must carry source_manifest_sha256.",
                field_path="checkpoint_admission.source_manifest_sha256",
            )
        )
    if not checkpoint_admission.payload_ref:
        issues.append(
            TierValidationIssue(
                code=TierValidationIssueCode.INVALID_CHECKPOINT_ADMISSION,
                message="Checkpoint admission must carry payload_ref.",
                field_path="checkpoint_admission.payload_ref",
            )
        )
    if checkpoint_admission.gate_detail_available:
        issues.append(
            TierValidationIssue(
                code=TierValidationIssueCode.INVALID_CHECKPOINT_ADMISSION,
                message="Checkpoint-only tier flow cannot claim gate detail availability.",
                field_path="checkpoint_admission.gate_detail_available",
            )
        )
    return tuple(issues)


def _validate_decision_status(bundle: TierDecisionBundle) -> tuple[TierValidationIssue, ...]:
    """Validate selected-tier/status consistency independent of family details."""
    issues: list[TierValidationIssue] = []
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
    return tuple(issues)


def _validate_leakage_decision(bundle: TierDecisionBundle) -> tuple[TierValidationIssue, ...]:
    """Validate leakage admissibility consistency in the final tier decision."""
    if bundle.status is TierStatus.SKIPPED:
        return ()
    if bundle.leakage_decision is None:
        return (
            TierValidationIssue(
                code=TierValidationIssueCode.INVALID_DECISION_BUNDLE,
                message="Non-skipped tier bundles must carry leakage admissibility.",
                field_path="leakage_decision",
            ),
        )
    if (
        bundle.selected_tier is not None
        and bundle.selected_tier not in bundle.leakage_decision.admissible_tiers
    ):
        return (
            TierValidationIssue(
                code=TierValidationIssueCode.SELECTED_TIER_CONTRADICTION,
                message="Selected tier is not admissible under sample-local leakage policy.",
                field_path="leakage_decision.admissible_tiers",
            ),
        )
    return ()


def _validate_family_decisions(bundle: TierDecisionBundle) -> tuple[TierValidationIssue, ...]:
    """Validate binding-family decision coverage and selected-tier support."""
    issues: list[TierValidationIssue] = []
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
                        field_path=f"family_decisions.{_family_value(decision.family)}",
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
                    field_path=(
                        f"family_decisions.{_family_value(decision.family)}.best_supported_tier"
                    ),
                )
            )
    return tuple(issues)


def _family_value(family: object) -> str:
    if isinstance(family, enum.Enum):
        return str(family.value)
    return str(family)


# Monotonicity logic


def _validate_monotonicity(config: TierFiltersConfig) -> tuple[TierValidationIssue, ...]:
    checks = (
        MonotonicThresholdCheck(
            config.oob.thresholds_by_level,
            "max_out_of_bounds_ratio",
            "nonincreasing",
            "oob",
        ),
        MonotonicThresholdCheck(
            config.upper_body_support.thresholds_by_level,
            "min_active_span_upper_body_support_landmark_coverage_ratio",
            "nondecreasing",
            "upper_body_support",
        ),
        MonotonicThresholdCheck(
            config.manual_visibility.thresholds_by_level,
            "min_active_span_any_hand_available_frame_ratio",
            "nondecreasing",
            "manual_visibility",
        ),
        MonotonicThresholdCheck(
            config.manual_visibility.thresholds_by_level,
            "max_active_span_any_hand_unavailable_run_ratio",
            "nonincreasing",
            "manual_visibility",
        ),
        MonotonicThresholdCheck(
            config.non_manual_visibility.thresholds_by_level,
            "min_active_span_face_available_frame_ratio",
            "nondecreasing",
            "non_manual_visibility",
        ),
        MonotonicThresholdCheck(
            config.non_manual_visibility.thresholds_by_level,
            "max_active_span_face_unavailable_run_ratio",
            "nonincreasing",
            "non_manual_visibility",
        ),
        MonotonicThresholdCheck(
            config.confidence.thresholds_by_level,
            "min_active_span_body_observed_mean_confidence",
            "nondecreasing",
            "confidence",
        ),
        MonotonicThresholdCheck(
            config.confidence.thresholds_by_level,
            "min_active_span_any_hand_observed_mean_confidence",
            "nondecreasing",
            "confidence",
        ),
        MonotonicThresholdCheck(
            config.kinematic_naturalness.thresholds_by_level,
            "max_comparable_transition_abrupt_ratio",
            "nonincreasing",
            "kinematic_naturalness",
        ),
        MonotonicThresholdCheck(
            config.kinematic_naturalness.thresholds_by_level,
            "max_comparable_transition_discontinuity_ratio",
            "nonincreasing",
            "kinematic_naturalness",
        ),
        MonotonicThresholdCheck(
            config.kinematic_naturalness.thresholds_by_level,
            "max_active_span_frozen_run_ratio",
            "nonincreasing",
            "kinematic_naturalness",
        ),
        MonotonicThresholdCheck(
            config.tracking_quality.thresholds_by_level,
            "max_tracked_target_missing_frame_ratio",
            "nonincreasing",
            "tracking_quality",
        ),
        MonotonicThresholdCheck(
            config.tracking_quality.thresholds_by_level,
            "max_person_tracking_continuity_break_ratio",
            "nonincreasing",
            "tracking_quality",
        ),
        MonotonicThresholdCheck(
            config.tracking_quality.thresholds_by_level,
            "max_person_tracking_reanchor_ratio",
            "nonincreasing",
            "tracking_quality",
        ),
        MonotonicThresholdCheck(
            config.manual_detail.thresholds_by_level,
            "min_active_span_representative_hand_landmark_coverage_ratio",
            "nondecreasing",
            "manual_detail",
        ),
        MonotonicThresholdCheck(
            config.manual_detail.thresholds_by_level,
            "min_active_span_representative_hand_fingertip_coverage_ratio",
            "nondecreasing",
            "manual_detail",
        ),
        MonotonicThresholdCheck(
            config.manual_detail.thresholds_by_level,
            "min_active_span_representative_hand_distal_chain_coverage_ratio",
            "nondecreasing",
            "manual_detail",
        ),
        MonotonicThresholdCheck(
            config.manual_detail.thresholds_by_level,
            "max_active_span_representative_hand_detail_dropout_run_ratio",
            "nonincreasing",
            "manual_detail",
        ),
        MonotonicThresholdCheck(
            config.non_manual_quality.thresholds_by_level,
            "min_active_span_face_landmark_coverage_ratio",
            "nondecreasing",
            "non_manual_quality",
        ),
        MonotonicThresholdCheck(
            config.non_manual_quality.thresholds_by_level,
            "min_active_span_upper_face_landmark_coverage_ratio",
            "nondecreasing",
            "non_manual_quality",
        ),
        MonotonicThresholdCheck(
            config.non_manual_quality.thresholds_by_level,
            "min_active_span_lower_face_landmark_coverage_ratio",
            "nondecreasing",
            "non_manual_quality",
        ),
        MonotonicThresholdCheck(
            config.non_manual_quality.thresholds_by_level,
            "max_active_span_face_detail_dropout_run_ratio",
            "nonincreasing",
            "non_manual_quality",
        ),
        MonotonicThresholdCheck(
            config.non_manual_quality.thresholds_by_level,
            "min_active_span_face_available_given_manual_frame_ratio",
            "nondecreasing",
            "non_manual_quality",
        ),
        MonotonicThresholdCheck(
            config.geometry.thresholds_by_level,
            "max_active_span_upper_body_bone_length_outlier_frame_ratio",
            "nonincreasing",
            "geometry",
        ),
        MonotonicThresholdCheck(
            config.geometry.thresholds_by_level,
            "max_active_span_cross_channel_scale_outlier_frame_ratio",
            "nonincreasing",
            "geometry",
        ),
    )
    issues: list[TierValidationIssue] = []
    for check in checks:
        values = _ordered_values(check.thresholds_by_level, check.attr_name)
        expected = (
            sorted(values) if check.direction == "nondecreasing" else sorted(values, reverse=True)
        )
        if values != expected:
            issues.append(
                TierValidationIssue(
                    code=TierValidationIssueCode.NON_MONOTONIC_THRESHOLD,
                    message=(
                        "Tier thresholds must be monotonic across loose, clean, tight "
                        f"({check.direction})."
                    ),
                    field_path=f"{check.family_name}.{check.attr_name}",
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
