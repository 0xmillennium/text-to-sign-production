"""Validation for typed tier decision bundles."""

from __future__ import annotations

from collections.abc import Callable

from text_to_sign_production.data.leakages.severity import LEAKAGE_SEVERITY_RANK
from text_to_sign_production.data.tiers.types import (
    BindingTierFamily,
    FilterLevel,
    TierBundle,
    TierMetricFailure,
    TierName,
    TierValidationIssue,
)

_TIER_NAMES: tuple[TierName, ...] = tuple(TierName)
_FAMILY_NAMES: tuple[BindingTierFamily, ...] = tuple(BindingTierFamily)
_FAMILY_DISPLAY_NAMES: tuple[str, ...] = tuple(family.value for family in BindingTierFamily)

_METRIC_RULES: dict[BindingTierFamily, dict[str, tuple[str, str]]] = {
    BindingTierFamily.OOB: {
        "out_of_bounds_ratio": ("max_out_of_bounds_ratio_exceeded", "<="),
    },
    BindingTierFamily.COVERAGE: {
        "body_landmark_coverage_ratio": (
            "min_body_landmark_coverage_ratio_not_met",
            ">=",
        ),
        "any_hand_landmark_coverage_ratio": (
            "min_any_hand_landmark_coverage_ratio_not_met",
            ">=",
        ),
        "face_landmark_coverage_ratio": (
            "min_face_landmark_coverage_ratio_not_met",
            ">=",
        ),
    },
    BindingTierFamily.HAND: {
        "any_hand_available_frame_ratio": (
            "min_any_hand_available_frame_ratio_not_met",
            ">=",
        ),
        "max_any_hand_unavailable_run_ratio": (
            "max_any_hand_unavailable_run_ratio_exceeded",
            "<=",
        ),
    },
    BindingTierFamily.FACE: {
        "face_available_frame_ratio": ("min_face_available_frame_ratio_not_met", ">="),
    },
    BindingTierFamily.CONFIDENCE: {
        "body_mean_confidence": ("min_body_mean_confidence_not_met", ">="),
        "left_hand_mean_confidence": ("min_left_hand_mean_confidence_not_met", ">="),
        "right_hand_mean_confidence": ("min_right_hand_mean_confidence_not_met", ">="),
        "face_mean_confidence": ("min_face_mean_confidence_not_met", ">="),
    },
    BindingTierFamily.TEXT: {
        "character_count": ("min_character_count_not_met", ">="),
        "token_count": ("min_token_count_not_met", ">="),
    },
    BindingTierFamily.LENGTH: {
        "num_frames": ("min_num_frames_not_met", ">="),
        "duration_seconds": ("min_duration_seconds_not_met", ">="),
    },
}


def validate_tier_bundle(bundle: TierBundle) -> list[TierValidationIssue]:
    """Validate a typed tier decision bundle without producing artifacts."""
    issues: list[TierValidationIssue] = []
    seen_decisions: set[tuple[object, str, object]] = set()
    tiers_by_sample: dict[tuple[object, str], set[TierName]] = {}

    def add(code: str, message: str) -> None:
        issues.append(TierValidationIssue(code=code, message=message))

    for decision in bundle.decisions:
        decision_key = (decision.split, decision.sample_id, decision.tier_name)
        if decision_key in seen_decisions:
            add("duplicate_decision", f"Duplicate TierDecision for {decision_key}")
        seen_decisions.add(decision_key)
        tiers_by_sample.setdefault((decision.split, decision.sample_id), set()).add(
            decision.tier_name
        )

        if decision.tier_name not in _TIER_NAMES:
            add("unknown_tier_name", f"Unknown tier name {decision.tier_name!r} for {decision_key}")

        actual_families = set(decision.applied_family_levels)
        expected_families = set(_FAMILY_NAMES)
        if actual_families != expected_families:
            add(
                "invalid_applied_family_levels",
                f"{decision_key} applied_family_levels must contain exactly "
                f"{list(_FAMILY_DISPLAY_NAMES)} "
                f"(missing={_display_families(expected_families - actual_families)}, "
                f"unknown={_display_families(actual_families - expected_families)})",
            )

        for family, level in decision.applied_family_levels.items():
            if family not in _FAMILY_NAMES:
                continue
            if not isinstance(level, FilterLevel):
                add(
                    "invalid_applied_level",
                    f"{decision_key} family {family!r} has invalid level {level!r}",
                )

        expected_included = not decision.metric_failures and decision.leakage_failure is None
        if decision.included != expected_included:
            add(
                "included_mismatch",
                f"{decision_key} included must equal absence of metric and leakage failures",
            )

        if decision.included:
            if decision.metric_failures or decision.leakage_failure is not None:
                add("included_with_failures", f"{decision_key} is included with failures")
        elif not decision.metric_failures and decision.leakage_failure is None:
            add("excluded_without_failures", f"{decision_key} is excluded without failures")

        for failure in decision.metric_failures:
            _validate_metric_failure(failure, decision_key, add)

        if decision.leakage_failure is not None:
            leakage_failure = decision.leakage_failure
            if leakage_failure.reason_code != "max_allowed_leakage_severity_exceeded":
                add(
                    "invalid_leakage_reason_code",
                    f"{decision_key} has invalid leakage reason {leakage_failure.reason_code!r}",
                )

            matched_keys = tuple(
                (ref.split, ref.sample_id) for ref in leakage_failure.matched_samples
            )
            if matched_keys != tuple(sorted(set(matched_keys))):
                add(
                    "invalid_leakage_matched_samples",
                    f"{decision_key} leakage matched_samples must be unique and sorted",
                )

            if (
                LEAKAGE_SEVERITY_RANK[leakage_failure.actual_max_severity]
                <= LEAKAGE_SEVERITY_RANK[leakage_failure.allowed_max_severity]
            ):
                add(
                    "invalid_leakage_severity_comparison",
                    f"{decision_key} leakage actual severity must exceed allowed severity",
                )

            if decision.max_leakage_severity != leakage_failure.actual_max_severity:
                add(
                    "max_leakage_severity_mismatch",
                    f"{decision_key} max_leakage_severity must match leakage failure severity",
                )

        if (
            decision.leakage_failure is None
            and decision.max_leakage_severity not in LEAKAGE_SEVERITY_RANK
        ):
            add(
                "invalid_max_leakage_severity",
                f"{decision_key} has invalid max_leakage_severity "
                f"{decision.max_leakage_severity!r}",
            )

    expected_tiers = set(_TIER_NAMES)
    for sample_key, tier_names in sorted(tiers_by_sample.items()):
        if tier_names != expected_tiers:
            add(
                "incomplete_sample_tier_decisions",
                f"{sample_key} must have exactly one decision for each tier "
                f"{[tier.value for tier in _TIER_NAMES]} "
                f"(missing={sorted(tier.value for tier in expected_tiers - tier_names)}, "
                f"unknown={sorted(tier.value for tier in tier_names - expected_tiers)})",
            )

    return issues


def _display_families(families: set[BindingTierFamily]) -> list[str]:
    return sorted(family.value for family in families)


def _validate_metric_failure(
    failure: TierMetricFailure,
    decision_key: tuple[str, str, str],
    add: Callable[[str, str], None],
) -> None:
    if failure.family not in _METRIC_RULES:
        add("invalid_metric_family", f"{decision_key} has invalid family {failure.family!r}")
        return

    family_rules = _METRIC_RULES[failure.family]
    expected_rule: tuple[str, str] | None
    if (
        failure.metric_key == "duration_seconds"
        and failure.reason_code == "duration_seconds_missing"
    ):
        expected_rule = ("duration_seconds_missing", ">=")
    else:
        expected_rule = family_rules.get(failure.metric_key)

    if expected_rule is None:
        add(
            "invalid_metric_key",
            f"{decision_key} family {failure.family!r} has invalid metric {failure.metric_key!r}",
        )
        return

    expected_reason, expected_comparison = expected_rule
    if failure.reason_code != expected_reason:
        add(
            "invalid_metric_reason_code",
            f"{decision_key} {failure.family}.{failure.metric_key} reason must be "
            f"{expected_reason!r}, got {failure.reason_code!r}",
        )
    if failure.comparison != expected_comparison:
        add(
            "invalid_metric_comparison",
            f"{decision_key} {failure.family}.{failure.metric_key} comparison must be "
            f"{expected_comparison!r}, got {failure.comparison!r}",
        )
    if not isinstance(failure.applied_level, FilterLevel):
        add(
            "invalid_metric_applied_level",
            f"{decision_key} {failure.family}.{failure.metric_key} has invalid applied level "
            f"{failure.applied_level!r}",
        )
