"""Validation for typed tier decision bundles."""

from __future__ import annotations

from collections.abc import Callable

from text_to_sign_production.data.leakages.types import LeakageSeverity
from text_to_sign_production.data.tiers.types import (
    FilterLevel,
    MetricFamily,
    TierBundle,
    TierMetricFailure,
    TierValidationIssue,
)

_TIER_NAMES = ("loose", "clean", "tight")
_FAMILY_NAMES = tuple(family.value for family in MetricFamily)
_SEVERITY_RANK = {
    LeakageSeverity.NONE: 0,
    LeakageSeverity.LOW: 1,
    LeakageSeverity.MEDIUM: 2,
    LeakageSeverity.HIGH: 3,
}

_METRIC_RULES: dict[str, dict[str, tuple[str, str]]] = {
    "oob": {
        "out_of_bounds_ratio": ("max_out_of_bounds_ratio_exceeded", "<="),
    },
    "hand": {
        "any_hand_nonzero_frame_ratio": (
            "min_any_hand_nonzero_frame_ratio_not_met",
            ">=",
        ),
    },
    "face": {
        "face_nonzero_frame_ratio": ("min_face_nonzero_frame_ratio_not_met", ">="),
        "face_missing_frame_ratio": ("max_face_missing_frame_ratio_exceeded", "<="),
    },
    "valid": {
        "valid_frame_ratio": ("min_valid_frame_ratio_not_met", ">="),
        "zeroed_canonical_joint_frame_ratio": (
            "max_zeroed_canonical_joint_frame_ratio_exceeded",
            "<=",
        ),
    },
    "confidence": {
        "overall_mean_confidence": ("min_overall_mean_confidence_not_met", ">="),
        "overall_nonzero_confidence_ratio": (
            "min_overall_nonzero_confidence_ratio_not_met",
            ">=",
        ),
    },
    "text": {
        "character_count": ("min_character_count_not_met", ">="),
        "token_count": ("min_token_count_not_met", ">="),
    },
    "length": {
        "num_frames": ("min_num_frames_not_met", ">="),
        "duration_seconds": ("min_duration_seconds_not_met", ">="),
    },
}


def validate_tier_bundle(bundle: TierBundle) -> list[TierValidationIssue]:
    """Validate a typed tier decision bundle without producing artifacts."""
    issues: list[TierValidationIssue] = []
    seen_decisions: set[tuple[str, str, str]] = set()
    tiers_by_sample: dict[tuple[str, str], set[str]] = {}

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
                f"{list(_FAMILY_NAMES)} (missing={sorted(expected_families - actual_families)}, "
                f"unknown={sorted(actual_families - expected_families)})",
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
                _SEVERITY_RANK[leakage_failure.actual_max_severity]
                <= _SEVERITY_RANK[leakage_failure.allowed_max_severity]
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

        if decision.leakage_failure is None and decision.max_leakage_severity not in _SEVERITY_RANK:
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
                f"{list(_TIER_NAMES)} (missing={sorted(expected_tiers - tier_names)}, "
                f"unknown={sorted(tier_names - expected_tiers)})",
            )

    return issues


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
