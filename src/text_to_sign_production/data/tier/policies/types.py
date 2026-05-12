"""Type system for tier policy parsing, binding, and evaluation."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from text_to_sign_production.core.ids import TierName
from text_to_sign_production.core.models import (
    CheckpointAdmission,
    PreparedSample,
    TierDecisionBundle,
    TierFamilyDecision,
    TierIssue,
    TierIssueCode,
    TierLeakageDecision,
    TierStatus,
)
from text_to_sign_production.data.tier.context import QualityContext
from text_to_sign_production.data.tier.facts import QualityFacts
from text_to_sign_production.data.tier.families.types import QualityMetricBundle
from text_to_sign_production.data.tier.leakages.types import LeakageSampleSummary


class TierValidationIssueCode(enum.StrEnum):
    """Stable tier-domain validation issue codes."""

    INVALID_THRESHOLD = "invalid_threshold"
    INVALID_FILTER_CONFIG = "invalid_filter_config"
    INVALID_POLICY_CONFIG = "invalid_policy_config"
    NON_MONOTONIC_THRESHOLD = "non_monotonic_threshold"
    MISSING_FAMILY_POLICY = "missing_family_policy"
    DIAGNOSTIC_FAMILY_BOUND = "diagnostic_family_bound"
    INVALID_DECISION_BUNDLE = "invalid_decision_bundle"
    SELECTED_TIER_CONTRADICTION = "selected_tier_contradiction"
    INVALID_CHECKPOINT_ADMISSION = "invalid_checkpoint_admission"


@dataclass(frozen=True, slots=True)
class TierValidationIssue:
    """Structured tier-domain validation issue."""

    code: TierValidationIssueCode
    message: str
    field_path: str | None = None


@dataclass(frozen=True, slots=True)
class TierEvaluationInput:
    """Single authoritative input contract for tier evaluation."""

    sample: PreparedSample
    quality_facts: QualityFacts
    quality_context: QualityContext
    quality_metrics: QualityMetricBundle
    leakage: LeakageSampleSummary
    checkpoint_admission: CheckpointAdmission


__all__ = [
    "CheckpointAdmission",
    "TierDecisionBundle",
    "TierEvaluationInput",
    "TierFamilyDecision",
    "TierIssue",
    "TierIssueCode",
    "TierLeakageDecision",
    "TierName",
    "TierStatus",
    "TierValidationIssue",
    "TierValidationIssueCode",
]
