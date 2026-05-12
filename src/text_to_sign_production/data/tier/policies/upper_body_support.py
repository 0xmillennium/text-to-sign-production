"""Upper-body-support tier threshold evaluation."""

from __future__ import annotations

from text_to_sign_production.data.tier.families.types import (
    BindingQualityFamily,
    UpperBodySupportMetrics,
)
from text_to_sign_production.data.tier.policies.filters import (
    UpperBodySupportTierFilters,
)
from text_to_sign_production.data.tier.policies.policies import TierPoliciesConfig
from text_to_sign_production.data.tier.policies.types import (
    TierFamilyDecision,
    TierIssue,
    TierIssueCode,
    TierName,
    TierStatus,
)

_FAMILY = BindingQualityFamily.UPPER_BODY_SUPPORT


def evaluate_upper_body_support_tier(
    metric: UpperBodySupportMetrics,
    filters: UpperBodySupportTierFilters,
    policies: TierPoliciesConfig,
) -> TierFamilyDecision:
    """Evaluate up to which tier upper-body support is sufficient."""
    supported: list[TierName] = []
    issues: list[TierIssue] = []
    for tier in policies.tier_order:
        level = policies.filter_level_for(tier, _FAMILY)
        thresholds = filters.thresholds_for(level)
        observed = metric.active_span_upper_body_landmark_coverage_ratio
        threshold = thresholds.min_active_span_upper_body_support_landmark_coverage_ratio
        if observed >= threshold:
            supported.append(tier)
        else:
            issues.append(
                _issue(
                    tier,
                    "active_span_upper_body_landmark_coverage_ratio",
                    observed,
                    threshold,
                )
            )
    return _decision(supported, issues)


def _issue(tier: TierName, metric_name: str, observed: float, threshold: float) -> TierIssue:
    return TierIssue(
        code=TierIssueCode.FAMILY_THRESHOLD_NOT_MET,
        message="Upper-body support is below the tier threshold.",
        family=_FAMILY,
        tier=tier,
        metric_name=metric_name,
        observed_value=observed,
        threshold_value=threshold,
    )


def _decision(supported: list[TierName], issues: list[TierIssue]) -> TierFamilyDecision:
    best = supported[-1] if supported else None
    status = TierStatus.PASSED if best is not None else TierStatus.FAILED
    if best is None:
        issues.append(
            TierIssue(
                code=TierIssueCode.FAMILY_UNSUPPORTED,
                message="Upper-body support does not support any configured tier.",
                family=_FAMILY,
            )
        )
    return TierFamilyDecision(_FAMILY, status, tuple(supported), best, tuple(issues))


__all__ = ["evaluate_upper_body_support_tier"]
