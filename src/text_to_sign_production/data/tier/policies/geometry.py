"""Geometry tier threshold evaluation."""

from __future__ import annotations

from text_to_sign_production.data.tier.families.types import BindingQualityFamily, GeometryMetrics
from text_to_sign_production.data.tier.policies.filters import GeometryTierFilters
from text_to_sign_production.data.tier.policies.policies import TierPoliciesConfig
from text_to_sign_production.data.tier.policies.types import (
    TierFamilyDecision,
    TierIssue,
    TierIssueCode,
    TierName,
    TierStatus,
)

_FAMILY = BindingQualityFamily.GEOMETRY


def evaluate_geometry_tier(
    metric: GeometryMetrics,
    filters: GeometryTierFilters,
    policies: TierPoliciesConfig,
) -> TierFamilyDecision:
    """Evaluate up to which tier geometry consistency is sufficient."""
    supported: list[TierName] = []
    issues: list[TierIssue] = []
    for tier in policies.tier_order:
        level = policies.filter_level_for(tier, _FAMILY)
        thresholds = filters.thresholds_for(level)
        checks = (
            (
                "active_span_upper_body_bone_length_outlier_frame_ratio",
                metric.active_span_upper_body_bone_length_outlier_frame_ratio,
                thresholds.max_active_span_upper_body_bone_length_outlier_frame_ratio,
                "Upper-body geometry outlier ratio is too high.",
            ),
            (
                "active_span_cross_channel_scale_outlier_frame_ratio",
                metric.active_span_cross_channel_scale_outlier_frame_ratio,
                thresholds.max_active_span_cross_channel_scale_outlier_frame_ratio,
                "Cross-channel scale outlier ratio is too high.",
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


def _decision(supported: list[TierName], issues: list[TierIssue]) -> TierFamilyDecision:
    best = supported[-1] if supported else None
    status = TierStatus.PASSED if best is not None else TierStatus.FAILED
    if best is None:
        issues.append(
            TierIssue(
                code=TierIssueCode.FAMILY_UNSUPPORTED,
                message="Geometry consistency does not support any configured tier.",
                family=_FAMILY,
            )
        )
    return TierFamilyDecision(_FAMILY, status, tuple(supported), best, tuple(issues))


__all__ = ["evaluate_geometry_tier"]
