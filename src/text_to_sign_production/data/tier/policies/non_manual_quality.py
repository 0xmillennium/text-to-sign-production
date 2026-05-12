"""Non-manual-quality tier threshold evaluation."""

from __future__ import annotations

from text_to_sign_production.data.tier.families.types import (
    BindingQualityFamily,
    NonManualQualityMetrics,
)
from text_to_sign_production.data.tier.policies.filters import (
    NonManualQualityTierFilters,
)
from text_to_sign_production.data.tier.policies.policies import TierPoliciesConfig
from text_to_sign_production.data.tier.policies.types import (
    TierFamilyDecision,
    TierIssue,
    TierIssueCode,
    TierName,
    TierStatus,
)

_FAMILY = BindingQualityFamily.NON_MANUAL_QUALITY


def evaluate_non_manual_quality_tier(
    metric: NonManualQualityMetrics,
    filters: NonManualQualityTierFilters,
    policies: TierPoliciesConfig,
) -> TierFamilyDecision:
    """Evaluate up to which tier face-detail quality is sufficient."""
    supported: list[TierName] = []
    issues: list[TierIssue] = []
    for tier in policies.tier_order:
        level = policies.filter_level_for(tier, _FAMILY)
        thresholds = filters.thresholds_for(level)
        min_checks = (
            (
                "active_span_face_landmark_coverage_ratio",
                metric.active_span_face_landmark_coverage_ratio,
                thresholds.min_active_span_face_landmark_coverage_ratio,
                "Face landmark coverage is too low.",
            ),
            (
                "active_span_upper_face_landmark_coverage_ratio",
                metric.active_span_upper_face_landmark_coverage_ratio,
                thresholds.min_active_span_upper_face_landmark_coverage_ratio,
                "Upper-face landmark coverage is too low.",
            ),
            (
                "active_span_lower_face_landmark_coverage_ratio",
                metric.active_span_lower_face_landmark_coverage_ratio,
                thresholds.min_active_span_lower_face_landmark_coverage_ratio,
                "Lower-face landmark coverage is too low.",
            ),
            (
                "active_span_face_available_given_manual_frame_ratio",
                metric.active_span_face_available_given_manual_frame_ratio,
                thresholds.min_active_span_face_available_given_manual_frame_ratio,
                "Face availability during manual-active frames is too low.",
            ),
        )
        tier_issues = [
            _issue(tier, metric_name, observed, threshold, message)
            for metric_name, observed, threshold, message in min_checks
            if observed < threshold
        ]
        if (
            metric.max_active_span_face_detail_dropout_run_ratio
            > thresholds.max_active_span_face_detail_dropout_run_ratio
        ):
            tier_issues.append(
                _issue(
                    tier,
                    "max_active_span_face_detail_dropout_run_ratio",
                    metric.max_active_span_face_detail_dropout_run_ratio,
                    thresholds.max_active_span_face_detail_dropout_run_ratio,
                    "Face-detail dropout run is too high.",
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


def _decision(supported: list[TierName], issues: list[TierIssue]) -> TierFamilyDecision:
    best = supported[-1] if supported else None
    status = TierStatus.PASSED if best is not None else TierStatus.FAILED
    if best is None:
        issues.append(
            TierIssue(
                code=TierIssueCode.FAMILY_UNSUPPORTED,
                message="Non-manual quality does not support any configured tier.",
                family=_FAMILY,
            )
        )
    return TierFamilyDecision(_FAMILY, status, tuple(supported), best, tuple(issues))


__all__ = ["evaluate_non_manual_quality_tier"]
