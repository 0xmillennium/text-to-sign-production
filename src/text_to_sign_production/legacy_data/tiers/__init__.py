"""Curated public facade for deterministic tier decisions."""

from __future__ import annotations

from text_to_sign_production.legacy_data.tiers.analysis import (
    build_excluded_membership_records,
    build_included_membership_records,
    build_tier_inclusion_count_records,
)
from text_to_sign_production.legacy_data.tiers.compute import (
    TierDecisionProgressEvent,
    TierDecisionProgressSink,
    build_tier_bundle,
)
from text_to_sign_production.legacy_data.tiers.filters import load_filter_config
from text_to_sign_production.legacy_data.tiers.policies import load_tier_policies
from text_to_sign_production.legacy_data.tiers.types import (
    BindingTierFamily,
    FilterConfig,
    FilterLevel,
    TierBundle,
    TierDecisionDetail,
    TierMembership,
    TierMembershipRecord,
    TierName,
    TierPolicy,
    TierValidationIssue,
)
from text_to_sign_production.legacy_data.tiers.validate import validate_tier_bundle

__all__ = [
    "BindingTierFamily",
    "FilterConfig",
    "FilterLevel",
    "TierBundle",
    "TierDecisionDetail",
    "TierDecisionProgressEvent",
    "TierDecisionProgressSink",
    "TierMembership",
    "TierMembershipRecord",
    "TierName",
    "TierPolicy",
    "TierValidationIssue",
    "build_excluded_membership_records",
    "build_included_membership_records",
    "build_tier_inclusion_count_records",
    "build_tier_bundle",
    "load_filter_config",
    "load_tier_policies",
    "validate_tier_bundle",
]
