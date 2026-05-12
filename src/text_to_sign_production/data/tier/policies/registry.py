"""Typed family binding helpers for tier policy evaluation."""

from __future__ import annotations

from text_to_sign_production.data.tier.families.types import (
    BindingQualityFamily,
    QualityMetricBundle,
)
from text_to_sign_production.data.tier.policies.filters import TierFiltersConfig
from text_to_sign_production.data.tier.policies.policies import TierPoliciesConfig
from text_to_sign_production.data.tier.policies.roles import (
    TierFamilyRole,
    binding_family_roles,
)
from text_to_sign_production.data.tier.policies.types import TierFamilyDecision


def evaluate_binding_family_role(
    role: TierFamilyRole,
    *,
    metrics: QualityMetricBundle,
    filters: TierFiltersConfig,
    policies: TierPoliciesConfig,
) -> TierFamilyDecision:
    """Evaluate one binding family using typed bundle/filter access."""
    match role.family:
        case BindingQualityFamily.OOB:
            return role.evaluator(metrics.oob, filters.oob, policies)
        case BindingQualityFamily.UPPER_BODY_SUPPORT:
            return role.evaluator(metrics.upper_body_support, filters.upper_body_support, policies)
        case BindingQualityFamily.MANUAL_VISIBILITY:
            return role.evaluator(metrics.manual_visibility, filters.manual_visibility, policies)
        case BindingQualityFamily.NON_MANUAL_VISIBILITY:
            return role.evaluator(
                metrics.non_manual_visibility,
                filters.non_manual_visibility,
                policies,
            )
        case BindingQualityFamily.CONFIDENCE:
            return role.evaluator(metrics.confidence, filters.confidence, policies)
        case BindingQualityFamily.KINEMATIC_NATURALNESS:
            return role.evaluator(
                metrics.kinematic_naturalness,
                filters.kinematic_naturalness,
                policies,
            )
        case BindingQualityFamily.TRACKING_QUALITY:
            return role.evaluator(metrics.tracking_quality, filters.tracking_quality, policies)
        case BindingQualityFamily.MANUAL_DETAIL:
            return role.evaluator(metrics.manual_detail, filters.manual_detail, policies)
        case BindingQualityFamily.NON_MANUAL_QUALITY:
            return role.evaluator(metrics.non_manual_quality, filters.non_manual_quality, policies)
        case BindingQualityFamily.GEOMETRY:
            return role.evaluator(metrics.geometry, filters.geometry, policies)
    raise ValueError(f"Unsupported binding family role: {role.family}")


def evaluate_binding_family_roles(
    *,
    metrics: QualityMetricBundle,
    filters: TierFiltersConfig,
    policies: TierPoliciesConfig,
) -> tuple[TierFamilyDecision, ...]:
    """Evaluate all registered binding families in authoritative order."""
    return tuple(
        evaluate_binding_family_role(
            role,
            metrics=metrics,
            filters=filters,
            policies=policies,
        )
        for role in binding_family_roles()
    )


__all__ = [
    "evaluate_binding_family_role",
    "evaluate_binding_family_roles",
]
