"""Validation for typed tier decision bundles."""

from __future__ import annotations

from collections.abc import Callable

from text_to_sign_production.data._shared.validate import is_sorted_unique_sequence
from text_to_sign_production.data.leakages.severity import LEAKAGE_SEVERITY_RANK
from text_to_sign_production.data.tiers.roles import (
    BINDING_TIER_FAMILIES,
    BINDING_TIER_METRIC_KEYS_BY_FAMILY,
    BINDING_TIER_METRICS_BY_FAMILY,
)
from text_to_sign_production.data.tiers.types import (
    BindingTierFamily,
    FilterLevel,
    TierBundle,
    TierMembership,
    TierMetricFailure,
    TierName,
    TierValidationIssue,
    binding_tier_family_display_label,
)

_TIER_NAMES: tuple[TierName, ...] = tuple(TierName)
_FAMILY_NAMES: tuple[BindingTierFamily, ...] = BINDING_TIER_FAMILIES
_FAMILY_DISPLAY_NAMES: tuple[str, ...] = tuple(
    binding_tier_family_display_label(family) for family in BindingTierFamily
)

_METRIC_RULES: dict[BindingTierFamily, dict[str, tuple[str, str]]] = {
    family: {spec.metric_key: (str(spec.reason_code), str(spec.comparison)) for spec in specs}
    for family, specs in BINDING_TIER_METRICS_BY_FAMILY.items()
}

for _family, _metric_rules in _METRIC_RULES.items():
    _configured = frozenset(_metric_rules)
    _declared = BINDING_TIER_METRIC_KEYS_BY_FAMILY[_family]
    if _configured != _declared:
        raise RuntimeError(
            f"Tier validation rules for {_family.value!r} do not match explicit binding roles "
            f"(rules={sorted(_configured)}, roles={sorted(_declared)})"
        )


def validate_tier_bundle(bundle: TierBundle) -> list[TierValidationIssue]:
    """Validate typed tier memberships and decision details without producing artifacts."""
    issues: list[TierValidationIssue] = []
    seen_memberships: set[tuple[object, str, object]] = set()
    seen_details: set[tuple[object, str, object]] = set()
    tiers_by_sample: dict[tuple[object, str], set[TierName]] = {}
    membership_by_key: dict[tuple[object, str, object], TierMembership] = {}

    def add(code: str, message: str) -> None:
        issues.append(TierValidationIssue(code=code, message=message))

    for membership in bundle.memberships:
        decision_key = (membership.split, membership.sample_id, membership.tier_name)
        if decision_key in seen_memberships:
            add("duplicate_membership", f"Duplicate TierMembershipRecord for {decision_key}")
        seen_memberships.add(decision_key)
        tiers_by_sample.setdefault((membership.split, membership.sample_id), set()).add(
            membership.tier_name
        )
        membership_by_key[decision_key] = membership.membership

        if membership.tier_name not in _TIER_NAMES:
            add(
                "unknown_tier_name",
                f"Unknown tier name {membership.tier_name!r} for {decision_key}",
            )
        if not isinstance(membership.membership, TierMembership):
            add(
                "invalid_membership",
                f"{decision_key} has invalid membership {membership.membership!r}",
            )

    for detail in bundle.decision_details:
        decision_key = (detail.split, detail.sample_id, detail.tier_name)
        if decision_key in seen_details:
            add("duplicate_decision_detail", f"Duplicate TierDecisionDetail for {decision_key}")
        seen_details.add(decision_key)

        if detail.tier_name not in _TIER_NAMES:
            add("unknown_tier_name", f"Unknown tier name {detail.tier_name!r} for {decision_key}")

        if membership_by_key.get(decision_key) != detail.membership:
            add(
                "membership_detail_mismatch",
                f"{decision_key} membership record must match decision detail membership",
            )

        actual_families = set(detail.applied_family_levels)
        expected_families = set(_FAMILY_NAMES)
        if actual_families != expected_families:
            add(
                "invalid_applied_family_levels",
                f"{decision_key} applied_family_levels must contain exactly "
                f"{list(_FAMILY_DISPLAY_NAMES)} "
                f"(missing={_display_families(expected_families - actual_families)}, "
                f"unknown={_display_families(actual_families - expected_families)})",
            )

        for family, level in detail.applied_family_levels.items():
            if family not in _FAMILY_NAMES:
                continue
            if not isinstance(level, FilterLevel):
                add(
                    "invalid_applied_level",
                    f"{decision_key} family {family!r} has invalid level {level!r}",
                )

        expected_membership = (
            TierMembership.INCLUDED
            if not detail.metric_failures and detail.leakage_failure is None
            else TierMembership.EXCLUDED
        )
        if detail.membership != expected_membership:
            add(
                "membership_mismatch",
                f"{decision_key} membership must equal absence or presence of failures",
            )

        if detail.membership is TierMembership.INCLUDED:
            if detail.metric_failures or detail.leakage_failure is not None:
                add("included_with_failures", f"{decision_key} is included with failures")
        elif not detail.metric_failures and detail.leakage_failure is None:
            add("excluded_without_failures", f"{decision_key} is excluded without failures")

        for failure in detail.metric_failures:
            _validate_metric_failure(failure, decision_key, add)

        if detail.leakage_failure is not None:
            leakage_failure = detail.leakage_failure
            if leakage_failure.reason_code != "max_allowed_leakage_severity_exceeded":
                add(
                    "invalid_leakage_reason_code",
                    f"{decision_key} has invalid leakage reason {leakage_failure.reason_code!r}",
                )

            matched_keys = tuple(
                (ref.split, ref.sample_id) for ref in leakage_failure.matched_samples
            )
            if not is_sorted_unique_sequence(matched_keys):
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

            if detail.max_leakage_severity != leakage_failure.actual_max_severity:
                add(
                    "max_leakage_severity_mismatch",
                    f"{decision_key} max_leakage_severity must match leakage failure severity",
                )

        if (
            detail.leakage_failure is None
            and detail.max_leakage_severity not in LEAKAGE_SEVERITY_RANK
        ):
            add(
                "invalid_max_leakage_severity",
                f"{decision_key} has invalid max_leakage_severity {detail.max_leakage_severity!r}",
            )

    if seen_memberships != seen_details:
        add(
            "membership_detail_keys_mismatch",
            "Tier memberships and decision details must have identical split/sample/tier keys",
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
    return sorted(binding_tier_family_display_label(family) for family in families)


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
