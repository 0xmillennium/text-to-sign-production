"""Tracking-quality tier threshold evaluation."""

from __future__ import annotations

from text_to_sign_production.data.tier.families import (
    BindingQualityFamily,
    TrackingQualityMetrics,
)
from text_to_sign_production.data.tier.policies.filters import (
    TrackingQualityTierFilters,
)
from text_to_sign_production.data.tier.policies.policies import TierPoliciesConfig
from text_to_sign_production.data.tier.policies.types import (
    FamilyTierDecision,
    TierIssue,
    TierIssueCode,
    TierName,
    TierStatus,
)

_FAMILY = BindingQualityFamily.TRACKING_QUALITY


def evaluate_tracking_quality_tier(
    metric: TrackingQualityMetrics,
    filters: TrackingQualityTierFilters,
    policies: TierPoliciesConfig,
) -> FamilyTierDecision:
    """Evaluate up to which tier tracking quality is sufficient."""
    supported: list[TierName] = []
    issues: list[TierIssue] = []
    for tier in policies.tier_order:
        level = policies.filter_level_for(tier, _FAMILY)
        thresholds = filters.thresholds_for(level)
        checks = (
            (
                "tracked_target_missing_frame_ratio",
                metric.tracked_target_missing_frame_ratio,
                thresholds.max_tracked_target_missing_frame_ratio,
                "Tracked-target missing ratio is too high.",
            ),
            (
                "person_tracking_continuity_break_ratio",
                metric.person_tracking_continuity_break_ratio,
                thresholds.max_person_tracking_continuity_break_ratio,
                "Tracking continuity-break ratio is too high.",
            ),
            (
                "person_tracking_reanchor_ratio",
                metric.person_tracking_reanchor_ratio,
                thresholds.max_person_tracking_reanchor_ratio,
                "Tracking reanchor ratio is too high.",
            ),
        )
        tier_issues = [
            _issue(tier, metric_name, observed, threshold, message)
            for metric_name, observed, threshold, message in checks
            if observed > threshold
        ]
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
                message="Tracking quality does not support any configured tier.",
                family=_FAMILY,
            )
        )
    return FamilyTierDecision(_FAMILY, status, tuple(supported), best, tuple(issues))


__all__ = ["evaluate_tracking_quality_tier"]
