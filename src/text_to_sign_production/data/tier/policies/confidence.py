"""Confidence tier threshold evaluation."""

from __future__ import annotations

from text_to_sign_production.data.tier.families import BindingQualityFamily, ConfidenceMetrics
from text_to_sign_production.data.tier.policies.filters import ConfidenceTierFilters
from text_to_sign_production.data.tier.policies.policies import TierPoliciesConfig
from text_to_sign_production.data.tier.policies.types import (
    FamilyTierDecision,
    TierIssue,
    TierIssueCode,
    TierName,
    TierStatus,
)

_FAMILY = BindingQualityFamily.CONFIDENCE


def evaluate_confidence_tier(
    metric: ConfidenceMetrics,
    filters: ConfidenceTierFilters,
    policies: TierPoliciesConfig,
) -> FamilyTierDecision:
    """Evaluate up to which tier the confidence profile is sufficient."""
    supported: list[TierName] = []
    issues: list[TierIssue] = []
    for tier in policies.tier_order:
        level = policies.filter_level_for(tier, _FAMILY)
        thresholds = filters.thresholds_for(level)
        tier_issues: list[TierIssue] = []
        if metric.active_span_body_mean_confidence < (
            thresholds.min_active_span_body_available_mean_confidence
        ):
            tier_issues.append(
                _issue(
                    tier,
                    "active_span_body_mean_confidence",
                    metric.active_span_body_mean_confidence,
                    thresholds.min_active_span_body_available_mean_confidence,
                    "Body confidence is too low.",
                )
            )
        if metric.active_span_hand_mean_confidence < (
            thresholds.min_active_span_any_hand_available_mean_confidence
        ):
            tier_issues.append(
                _issue(
                    tier,
                    "active_span_hand_mean_confidence",
                    metric.active_span_hand_mean_confidence,
                    thresholds.min_active_span_any_hand_available_mean_confidence,
                    "Hand confidence is too low.",
                )
            )
        if tier_issues:
            issues.extend(tier_issues)
        else:
            supported.append(tier)
    return _decision(supported, issues)


def _issue(
    tier: TierName,
    metric_name: str,
    observed: float,
    threshold: float,
    message: str,
) -> TierIssue:
    return TierIssue(
        code=TierIssueCode.FAMILY_THRESHOLD_NOT_MET,
        message=message,
        family=_FAMILY,
        tier=tier,
        metric_name=metric_name,
        observed_value=observed,
        threshold_value=threshold,
    )


def _decision(supported: list[TierName], issues: list[TierIssue]) -> FamilyTierDecision:
    best = supported[-1] if supported else None
    status = TierStatus.PASSED if best is not None else TierStatus.FAILED
    if best is None:
        issues.append(
            TierIssue(
                code=TierIssueCode.FAMILY_UNSUPPORTED,
                message="Confidence does not support any configured tier.",
                family=_FAMILY,
            )
        )
    return FamilyTierDecision(_FAMILY, status, tuple(supported), best, tuple(issues))


__all__ = ["evaluate_confidence_tier"]
