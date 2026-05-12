"""Authoritative family-role registry for tier evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from text_to_sign_production.data.tier.families.types import (
    BindingQualityFamily,
    DiagnosticQualityFamily,
)
from text_to_sign_production.data.tier.policies.confidence import (
    evaluate_confidence_tier,
)
from text_to_sign_production.data.tier.policies.filters import (
    ConfidenceTierFilters,
    GeometryTierFilters,
    KinematicNaturalnessTierFilters,
    ManualDetailTierFilters,
    ManualVisibilityTierFilters,
    NonManualQualityTierFilters,
    NonManualVisibilityTierFilters,
    OobTierFilters,
    TrackingQualityTierFilters,
    UpperBodySupportTierFilters,
)
from text_to_sign_production.data.tier.policies.geometry import evaluate_geometry_tier
from text_to_sign_production.data.tier.policies.kinematic_naturalness import (
    evaluate_kinematic_naturalness_tier,
)
from text_to_sign_production.data.tier.policies.manual_detail import (
    evaluate_manual_detail_tier,
)
from text_to_sign_production.data.tier.policies.manual_visibility import (
    evaluate_manual_visibility_tier,
)
from text_to_sign_production.data.tier.policies.non_manual_quality import (
    evaluate_non_manual_quality_tier,
)
from text_to_sign_production.data.tier.policies.non_manual_visibility import (
    evaluate_non_manual_visibility_tier,
)
from text_to_sign_production.data.tier.policies.oob import evaluate_oob_tier
from text_to_sign_production.data.tier.policies.policies import TierPoliciesConfig
from text_to_sign_production.data.tier.policies.tracking_quality import (
    evaluate_tracking_quality_tier,
)
from text_to_sign_production.data.tier.policies.types import TierFamilyDecision
from text_to_sign_production.data.tier.policies.upper_body_support import (
    evaluate_upper_body_support_tier,
)

TierEvaluator = Callable[..., TierFamilyDecision]


@dataclass(frozen=True, slots=True)
class TierFamilyRole:
    """Registry entry for one binding family."""

    family: BindingQualityFamily
    filter_type: type[object]
    evaluator: TierEvaluator
    mandatory: bool = True


BINDING_TIER_FAMILIES: tuple[BindingQualityFamily, ...] = tuple(BindingQualityFamily)
DIAGNOSTIC_TIER_FAMILIES: tuple[DiagnosticQualityFamily, ...] = (
    DiagnosticQualityFamily.TEXT,
    DiagnosticQualityFamily.LENGTH,
)

TIER_FAMILY_ROLES: tuple[TierFamilyRole, ...] = (
    TierFamilyRole(
        family=BindingQualityFamily.OOB,
        filter_type=OobTierFilters,
        evaluator=evaluate_oob_tier,
    ),
    TierFamilyRole(
        family=BindingQualityFamily.UPPER_BODY_SUPPORT,
        filter_type=UpperBodySupportTierFilters,
        evaluator=evaluate_upper_body_support_tier,
    ),
    TierFamilyRole(
        family=BindingQualityFamily.MANUAL_VISIBILITY,
        filter_type=ManualVisibilityTierFilters,
        evaluator=evaluate_manual_visibility_tier,
    ),
    TierFamilyRole(
        family=BindingQualityFamily.NON_MANUAL_VISIBILITY,
        filter_type=NonManualVisibilityTierFilters,
        evaluator=evaluate_non_manual_visibility_tier,
    ),
    TierFamilyRole(
        family=BindingQualityFamily.CONFIDENCE,
        filter_type=ConfidenceTierFilters,
        evaluator=evaluate_confidence_tier,
    ),
    TierFamilyRole(
        family=BindingQualityFamily.KINEMATIC_NATURALNESS,
        filter_type=KinematicNaturalnessTierFilters,
        evaluator=evaluate_kinematic_naturalness_tier,
    ),
    TierFamilyRole(
        family=BindingQualityFamily.TRACKING_QUALITY,
        filter_type=TrackingQualityTierFilters,
        evaluator=evaluate_tracking_quality_tier,
    ),
    TierFamilyRole(
        family=BindingQualityFamily.MANUAL_DETAIL,
        filter_type=ManualDetailTierFilters,
        evaluator=evaluate_manual_detail_tier,
    ),
    TierFamilyRole(
        family=BindingQualityFamily.NON_MANUAL_QUALITY,
        filter_type=NonManualQualityTierFilters,
        evaluator=evaluate_non_manual_quality_tier,
    ),
    TierFamilyRole(
        family=BindingQualityFamily.GEOMETRY,
        filter_type=GeometryTierFilters,
        evaluator=evaluate_geometry_tier,
    ),
)

TIER_FAMILY_ROLES_BY_FAMILY: Mapping[BindingQualityFamily, TierFamilyRole] = MappingProxyType(
    {role.family: role for role in TIER_FAMILY_ROLES}
)


def binding_family_roles() -> tuple[TierFamilyRole, ...]:
    """Return binding family role entries in authoritative evaluation order."""
    return TIER_FAMILY_ROLES


def diagnostic_family_names() -> tuple[DiagnosticQualityFamily, ...]:
    """Return diagnostic families excluded from tier banding."""
    return DIAGNOSTIC_TIER_FAMILIES


def required_policy_families(_policies: TierPoliciesConfig) -> tuple[BindingQualityFamily, ...]:
    """Return mandatory binding families for selected-tier policy."""
    return tuple(role.family for role in TIER_FAMILY_ROLES if role.mandatory)


__all__ = [
    "BINDING_TIER_FAMILIES",
    "DIAGNOSTIC_TIER_FAMILIES",
    "TIER_FAMILY_ROLES",
    "TIER_FAMILY_ROLES_BY_FAMILY",
    "TierFamilyRole",
    "binding_family_roles",
    "diagnostic_family_names",
    "required_policy_families",
]
