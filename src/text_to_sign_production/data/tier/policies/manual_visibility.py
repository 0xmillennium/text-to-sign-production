"""Manual-visibility tier threshold evaluation."""

from __future__ import annotations

from text_to_sign_production.data.tier.families.types import (
    BindingQualityFamily,
    ManualVisibilityMetrics,
)
from text_to_sign_production.data.tier.policies.filters import (
    ManualVisibilityTierFilters,
)
from text_to_sign_production.data.tier.policies.policies import TierPoliciesConfig
from text_to_sign_production.data.tier.policies.types import (
    TierFamilyDecision,
    TierIssue,
    TierIssueCode,
    TierName,
    TierStatus,
)

_FAMILY = BindingQualityFamily.MANUAL_VISIBILITY


def evaluate_manual_visibility_tier(
    metric: ManualVisibilityMetrics,
    filters: ManualVisibilityTierFilters,
    policies: TierPoliciesConfig,
) -> TierFamilyDecision:
    """Evaluate up to which tier coarse manual visibility is sufficient."""
    supported: list[TierName] = []
    issues: list[TierIssue] = []
    for tier in policies.tier_order:
        level = policies.filter_level_for(tier, _FAMILY)
        thresholds = filters.thresholds_for(level)
        tier_issues: list[TierIssue] = []
        tier_issues.extend(
            _min_issue(
                tier,
                "active_span_any_hand_available_frame_ratio",
                metric.active_span_any_hand_available_frame_ratio,
                thresholds.min_active_span_any_hand_available_frame_ratio,
            )
        )
        tier_issues.extend(
            _max_issue(
                tier,
                "max_active_span_any_hand_dropout_run_ratio",
                metric.max_active_span_any_hand_dropout_run_ratio,
                thresholds.max_active_span_any_hand_unavailable_run_ratio,
            )
        )
        if tier_issues:
            issues.extend(tier_issues)
        else:
            supported.append(tier)
    return _decision(supported, issues)


def _min_issue(
    tier: TierName,
    metric_name: str,
    observed: float,
    threshold: float,
) -> tuple[TierIssue, ...]:
    if observed >= threshold:
        return ()
    return (_issue(tier, metric_name, observed, threshold, "Manual visibility is too low."),)


def _max_issue(
    tier: TierName,
    metric_name: str,
    observed: float,
    threshold: float,
) -> tuple[TierIssue, ...]:
    if observed <= threshold:
        return ()
    return (_issue(tier, metric_name, observed, threshold, "Manual dropout run is too high."),)


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


def _decision(supported: list[TierName], issues: list[TierIssue]) -> TierFamilyDecision:
    best = supported[-1] if supported else None
    status = TierStatus.PASSED if best is not None else TierStatus.FAILED
    if best is None:
        issues.append(
            TierIssue(
                code=TierIssueCode.FAMILY_UNSUPPORTED,
                message="Manual visibility does not support any configured tier.",
                family=_FAMILY,
            )
        )
    return TierFamilyDecision(_FAMILY, status, tuple(supported), best, tuple(issues))


__all__ = ["evaluate_manual_visibility_tier"]
