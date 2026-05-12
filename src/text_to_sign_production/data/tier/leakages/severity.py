"""Deterministic leakage severity rules."""

from __future__ import annotations

from text_to_sign_production.data.tier.leakages.types import LeakageRelation, LeakageSeverity


def classify_leakage_severity(relations: tuple[LeakageRelation, ...]) -> LeakageSeverity:
    """Classify a pair's relation set."""
    if not relations:
        return LeakageSeverity.NONE
    if LeakageRelation.SAME_SOURCE_SENTENCE in relations:
        return LeakageSeverity.HIGH
    if LeakageRelation.EXACT_TEXT in relations:
        return LeakageSeverity.MEDIUM
    return LeakageSeverity.LOW


def max_leakage_severity(values: tuple[LeakageSeverity, ...]) -> LeakageSeverity:
    """Return the maximum severity in deterministic order."""
    order = {
        LeakageSeverity.NONE: 0,
        LeakageSeverity.LOW: 1,
        LeakageSeverity.MEDIUM: 2,
        LeakageSeverity.HIGH: 3,
    }
    return max(values, key=lambda value: order[value], default=LeakageSeverity.NONE)


__all__ = ["classify_leakage_severity", "max_leakage_severity"]
