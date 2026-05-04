"""Strict, typed dataset tier decisions."""

from __future__ import annotations

from text_to_sign_production.data.tiers.decide import build_tier_bundle
from text_to_sign_production.data.tiers.filters import load_filter_config, parse_filter_config
from text_to_sign_production.data.tiers.policies import load_tier_policies, parse_tier_policies
from text_to_sign_production.data.tiers.surfaces import (
    build_excluded_surface_entries,
    build_included_surface_entries,
)
from text_to_sign_production.data.tiers.types import (
    ConfidenceThresholds,
    ExcludedTierSurfaceEntry,
    FaceThresholds,
    FilterConfig,
    FilterLevel,
    HandThresholds,
    IncludedTierSurfaceEntry,
    LengthThresholds,
    MetricFamily,
    OobThresholds,
    TextThresholds,
    TierBundle,
    TierDecision,
    TierLeakageFailure,
    TierMetricFailure,
    TierPolicy,
    TierValidationIssue,
    ValidThresholds,
)
from text_to_sign_production.data.tiers.validate import validate_tier_bundle

__all__ = [
    "ConfidenceThresholds",
    "ExcludedTierSurfaceEntry",
    "FaceThresholds",
    "FilterConfig",
    "FilterLevel",
    "HandThresholds",
    "IncludedTierSurfaceEntry",
    "LengthThresholds",
    "MetricFamily",
    "OobThresholds",
    "TextThresholds",
    "TierBundle",
    "TierDecision",
    "TierLeakageFailure",
    "TierMetricFailure",
    "TierPolicy",
    "TierValidationIssue",
    "ValidThresholds",
    "build_excluded_surface_entries",
    "build_included_surface_entries",
    "build_tier_bundle",
    "load_filter_config",
    "load_tier_policies",
    "parse_filter_config",
    "parse_tier_policies",
    "validate_tier_bundle",
]
