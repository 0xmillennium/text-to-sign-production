"""Type system for tier policy parsing, binding, and evaluation."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from text_to_sign_production.core.ids import TierName
from text_to_sign_production.core.models import GateDecisionBundle, GateName, PreparedSample
from text_to_sign_production.data.tier.context import QualityContext
from text_to_sign_production.data.tier.facts import QualityFacts
from text_to_sign_production.data.tier.families import BindingQualityFamily, QualityMetricBundle
from text_to_sign_production.data.tier.leakages import LeakageBundle


class TierStatus(enum.StrEnum):
    """Controlled tier-decision status."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TierIssueCode(enum.StrEnum):
    """Stable machine-readable tier issue codes."""

    GATE_FAILED = "gate_failed"
    FAMILY_THRESHOLD_NOT_MET = "family_threshold_not_met"
    FAMILY_UNSUPPORTED = "family_unsupported"
    NO_SUPPORTED_TIER = "no_supported_tier"
    CONFIG_INVALID = "config_invalid"
    POLICY_INVALID = "policy_invalid"
    DECISION_INVALID = "decision_invalid"


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
    GATE_FAILED_TIER_SELECTED = "gate_failed_tier_selected"


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
    leakage: LeakageBundle
    gate_decisions: GateDecisionBundle


@dataclass(frozen=True, slots=True)
class TierIssue:
    """One structured tier policy issue."""

    code: TierIssueCode
    message: str
    family: BindingQualityFamily | None = None
    tier: TierName | None = None
    metric_name: str | None = None
    observed_value: int | float | str | bool | None = None
    threshold_value: int | float | str | bool | None = None
    gate_name: GateName | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class FamilyTierDecision:
    """Best supported tier for one binding metric family."""

    family: BindingQualityFamily
    status: TierStatus
    supported_tiers: tuple[TierName, ...]
    best_supported_tier: TierName | None
    issues: tuple[TierIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "supported_tiers", tuple(self.supported_tiers))
        object.__setattr__(self, "issues", tuple(self.issues))


@dataclass(frozen=True, slots=True)
class TierDecisionBundle:
    """Structured tier-band decision for one sample."""

    status: TierStatus
    selected_tier: TierName | None
    family_decisions: tuple[FamilyTierDecision, ...]
    issues: tuple[TierIssue, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_decisions", tuple(self.family_decisions))
        object.__setattr__(self, "issues", tuple(self.issues))


__all__ = [
    "FamilyTierDecision",
    "TierDecisionBundle",
    "TierEvaluationInput",
    "TierIssue",
    "TierIssueCode",
    "TierName",
    "TierStatus",
    "TierValidationIssue",
    "TierValidationIssueCode",
]
