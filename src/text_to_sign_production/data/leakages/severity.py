"""Deterministic leakage severity mapping."""

from __future__ import annotations

from text_to_sign_production.data.leakages.types import LeakageRelation, LeakageSeverity


def classify_leakage_severity(
    relations: tuple[LeakageRelation, ...] | None,
) -> LeakageSeverity:
    """Map a set of exact relations to a deterministic severity."""
    if not relations:
        return LeakageSeverity.NONE

    if LeakageRelation.SAME_SOURCE_SENTENCE in relations:
        return LeakageSeverity.HIGH

    if LeakageRelation.EXACT_NORMALIZED_TEXT in relations:
        return LeakageSeverity.MEDIUM

    if LeakageRelation.SAME_SOURCE_VIDEO in relations:
        return LeakageSeverity.LOW

    return LeakageSeverity.NONE
