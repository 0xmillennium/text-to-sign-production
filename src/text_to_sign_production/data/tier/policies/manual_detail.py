"""Manual-detail tier threshold evaluation."""

from __future__ import annotations

from text_to_sign_production.data.tier.families import BindingQualityFamily, ManualDetailMetrics
from text_to_sign_production.data.tier.policies.filters import ManualDetailTierFilters
from text_to_sign_production.data.tier.policies.policies import TierPoliciesConfig
from text_to_sign_production.data.tier.policies.types import (
    FamilyTierDecision,
    TierIssue,
    TierIssueCode,
    TierName,
    TierStatus,
)

_FAMILY = BindingQualityFamily.MANUAL_DETAIL


def evaluate_manual_detail_tier(
    metric: ManualDetailMetrics,
    filters: ManualDetailTierFilters,
    policies: TierPoliciesConfig,
) -> FamilyTierDecision:
    """Evaluate up to which tier representative-hand detail is sufficient."""
    supported: list[TierName] = []
    issues: list[TierIssue] = []
    for tier in policies.tier_order:
        level = policies.filter_level_for(tier, _FAMILY)
        thresholds = filters.thresholds_for(level)
        tier_issues: list[TierIssue] = []
        min_checks = (
            (
                "active_span_representative_hand_landmark_coverage_ratio",
                metric.active_span_representative_hand_landmark_coverage_ratio,
                thresholds.min_active_span_representative_hand_landmark_coverage_ratio,
                "Representative-hand landmark coverage is too low.",
            ),
            (
                "active_span_representative_hand_fingertip_coverage_ratio",
                metric.active_span_representative_hand_fingertip_coverage_ratio,
                thresholds.min_active_span_representative_hand_fingertip_coverage_ratio,
                "Representative-hand fingertip coverage is too low.",
            ),
            (
                "active_span_representative_hand_distal_chain_coverage_ratio",
                metric.active_span_representative_hand_distal_chain_coverage_ratio,
                thresholds.min_active_span_representative_hand_distal_chain_coverage_ratio,
                "Representative-hand distal-chain coverage is too low.",
            ),
        )
        tier_issues.extend(
            _issue(tier, metric_name, observed, threshold, message)
            for metric_name, observed, threshold, message in min_checks
            if observed < threshold
        )
        if (
            metric.max_active_span_representative_hand_detail_dropout_run_ratio
            > thresholds.max_active_span_representative_hand_detail_dropout_run_ratio
        ):
            tier_issues.append(
                _issue(
                    tier,
                    "max_active_span_representative_hand_detail_dropout_run_ratio",
                    metric.max_active_span_representative_hand_detail_dropout_run_ratio,
                    thresholds.max_active_span_representative_hand_detail_dropout_run_ratio,
                    "Representative-hand detail dropout run is too high.",
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
                message="Manual detail does not support any configured tier.",
                family=_FAMILY,
            )
        )
    return FamilyTierDecision(_FAMILY, status, tuple(supported), best, tuple(issues))


__all__ = ["evaluate_manual_detail_tier"]
