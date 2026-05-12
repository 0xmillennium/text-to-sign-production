"""Read-only tier decision analysis helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import cast

from text_to_sign_production.core.ids import TierMembership, TierName
from text_to_sign_production.data.tier.families.types import BindingQualityFamily
from text_to_sign_production.data.tier.policies.types import (
    TierDecisionBundle,
    TierFamilyDecision,
    TierIssueCode,
    TierStatus,
)


@dataclass(frozen=True, slots=True)
class SelectedTierSummary:
    """Compact selected-tier summary."""

    status: TierStatus
    selected_tier: TierName | None
    family_count: int
    issue_count: int


@dataclass(frozen=True, slots=True)
class WeakestLinkSummary:
    """Families that constrain the selected tier."""

    weakest_tier: TierName | None
    families: tuple[BindingQualityFamily, ...]


@dataclass(frozen=True, slots=True)
class TierIssueFrequency:
    """Frequency of one tier issue code."""

    code: TierIssueCode
    count: int


def summarize_selected_tier(bundle: TierDecisionBundle) -> SelectedTierSummary:
    """Summarize selected/skipped tier state without re-evaluating policy."""
    return SelectedTierSummary(
        status=bundle.status,
        selected_tier=bundle.selected_tier,
        family_count=len(bundle.family_decisions),
        issue_count=len(bundle.issues),
    )


def summarize_weakest_links(bundle: TierDecisionBundle) -> WeakestLinkSummary:
    """Return binding families with the weakest best-supported tier."""
    decisions = tuple(
        decision for decision in bundle.family_decisions if decision.best_supported_tier is not None
    )
    if not decisions:
        return WeakestLinkSummary(weakest_tier=None, families=())
    tier_order: tuple[TierName, ...] = (TierName.LOOSE, TierName.CLEAN, TierName.TIGHT)
    order: dict[TierName, int] = {tier: index for index, tier in enumerate(tier_order)}
    weakest_indexes: list[int] = []
    for decision in decisions:
        best = decision.best_supported_tier
        if best is not None:
            weakest_indexes.append(order[best])
    weakest_index = min(weakest_indexes)
    weakest_tier = tier_order[weakest_index]
    return WeakestLinkSummary(
        weakest_tier=weakest_tier,
        families=tuple(
            cast(BindingQualityFamily, decision.family)
            for decision in decisions
            if decision.best_supported_tier == weakest_tier
        ),
    )


def skipped_vs_selected_summary(
    bundles: tuple[TierDecisionBundle, ...],
) -> dict[TierStatus, int]:
    """Count selected, failed, and skipped bundles."""
    counter = Counter(bundle.status for bundle in bundles)
    return {status: counter[status] for status in TierStatus}


def family_decision(
    bundle: TierDecisionBundle,
    family: BindingQualityFamily,
) -> TierFamilyDecision | None:
    """Inspect one family decision by name."""
    for decision in bundle.family_decisions:
        if decision.family is family:
            return decision
    return None


def issue_frequency_summary(bundle: TierDecisionBundle) -> tuple[TierIssueFrequency, ...]:
    """Count tier issue codes in a decision bundle."""
    counter = Counter(issue.code for issue in bundle.issues)
    return tuple(
        TierIssueFrequency(code=code, count=count)
        for code, count in sorted(counter.items(), key=lambda item: item[0].value)
    )


def tier_membership_for_decision(
    decision: TierDecisionBundle,
    tier: TierName,
) -> TierMembership:
    """Project a tier decision into included/excluded membership for one tier."""
    selected_tier = decision.selected_tier
    if selected_tier is None:
        return TierMembership.EXCLUDED
    tier_order = tuple(TierName)
    if tier_order.index(tier) <= tier_order.index(selected_tier):
        return TierMembership.INCLUDED
    return TierMembership.EXCLUDED


__all__ = [
    "SelectedTierSummary",
    "TierIssueFrequency",
    "WeakestLinkSummary",
    "family_decision",
    "issue_frequency_summary",
    "skipped_vs_selected_summary",
    "summarize_selected_tier",
    "summarize_weakest_links",
    "tier_membership_for_decision",
]
