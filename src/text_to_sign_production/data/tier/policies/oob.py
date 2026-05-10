"""OOB tier threshold evaluation."""

from __future__ import annotations

from text_to_sign_production.data.tier.families import BindingQualityFamily, OobMetrics
from text_to_sign_production.data.tier.policies.filters import OobTierFilters
from text_to_sign_production.data.tier.policies.policies import TierPoliciesConfig
from text_to_sign_production.data.tier.policies.types import (
    FamilyTierDecision,
    TierIssue,
    TierIssueCode,
    TierName,
    TierStatus,
)

_FAMILY = BindingQualityFamily.OOB


def evaluate_oob_tier(
    metric: OobMetrics,
    filters: OobTierFilters,
    policies: TierPoliciesConfig,
) -> FamilyTierDecision:
    """Evaluate up to which tier the sample is supported on OOB constraints."""
    supported: list[TierName] = []
    issues: list[TierIssue] = []
    for tier in policies.tier_order:
        level = policies.filter_level_for(tier, _FAMILY)
        thresholds = filters.thresholds_for(level)
        observed = metric.max_channel_out_of_bounds_frame_ratio
        threshold = thresholds.max_out_of_bounds_ratio
        if observed <= threshold:
            supported.append(tier)
        else:
            issues.append(
                _issue(
                    tier,
                    "max_channel_out_of_bounds_frame_ratio",
                    observed,
                    threshold,
                )
            )
    return _decision(supported, issues)


def _issue(tier: TierName, metric_name: str, observed: float, threshold: float) -> TierIssue:
    return TierIssue(
        code=TierIssueCode.FAMILY_THRESHOLD_NOT_MET,
        message="OOB ratio exceeds the tier threshold.",
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
                message="OOB constraints do not support any configured tier.",
                family=_FAMILY,
            )
        )
    return FamilyTierDecision(
        family=_FAMILY,
        status=status,
        supported_tiers=tuple(supported),
        best_supported_tier=best,
        issues=tuple(issues),
    )


__all__ = ["evaluate_oob_tier"]
