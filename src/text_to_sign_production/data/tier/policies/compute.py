"""Thin orchestration for PreparedSample-based tier policy evaluation."""

from __future__ import annotations

from typing import cast

from text_to_sign_production.core.ids import TierName
from text_to_sign_production.core.models import CheckpointAdmission, PreparedSample
from text_to_sign_production.data.tier.context import QualityContext
from text_to_sign_production.data.tier.facts import QualityFacts
from text_to_sign_production.data.tier.families.types import (
    BindingQualityFamily,
    QualityMetricBundle,
)
from text_to_sign_production.data.tier.leakages import LeakageSampleSummary, LeakageSeverity
from text_to_sign_production.data.tier.policies.filters import TierFiltersConfig
from text_to_sign_production.data.tier.policies.policies import (
    TierLeakageSeverity,
    TierPoliciesConfig,
)
from text_to_sign_production.data.tier.policies.registry import evaluate_binding_family_roles
from text_to_sign_production.data.tier.policies.roles import (
    required_policy_families,
)
from text_to_sign_production.data.tier.policies.types import (
    TierDecisionBundle,
    TierEvaluationInput,
    TierFamilyDecision,
    TierIssue,
    TierIssueCode,
    TierLeakageDecision,
    TierStatus,
)

_LEAKAGE_SEVERITY_RANK = {str(severity): rank for rank, severity in enumerate(LeakageSeverity)}


def evaluate_tier_decision(
    sample: PreparedSample,
    facts: QualityFacts,
    context: QualityContext,
    metrics: QualityMetricBundle,
    leakage: LeakageSampleSummary,
    checkpoint_admission: CheckpointAdmission,
    filters: TierFiltersConfig,
    policies: TierPoliciesConfig,
) -> TierDecisionBundle:
    """Evaluate tier banding from PreparedSample-derived quality inputs."""
    if (
        sample.source.sample_id != checkpoint_admission.sample_id
        or sample.source.split is not checkpoint_admission.split
    ):
        raise ValueError("Checkpoint admission identity must match the PreparedSample.")
    if not checkpoint_admission.source_manifest_sha256 or checkpoint_admission.payload_ref is None:
        raise ValueError("Checkpoint admission must carry manifest digest and payload_ref.")
    input = TierEvaluationInput(
        sample=sample,
        quality_facts=facts,
        quality_context=context,
        quality_metrics=metrics,
        leakage=leakage,
        checkpoint_admission=checkpoint_admission,
    )
    family_decisions = evaluate_binding_family_roles(
        metrics=input.quality_metrics,
        filters=filters,
        policies=policies,
    )
    supported_by_family = {
        cast(BindingQualityFamily, decision.family): decision.supported_tiers
        for decision in family_decisions
    }
    binding_families = required_policy_families(policies)
    family_supported_tier = policies.select_best_supported_tier(
        supported_by_family,
        binding_families,
    )
    leakage_decision = _evaluate_leakage_admissibility(leakage, policies)
    selected_tier = _select_best_family_and_leakage_supported_tier(
        policies=policies,
        supported_by_family=supported_by_family,
        binding_families=binding_families,
        leakage_decision=leakage_decision,
    )
    issues = tuple(issue for decision in family_decisions for issue in decision.issues)
    if family_supported_tier is None:
        issues = (
            *issues,
            TierIssue(
                code=TierIssueCode.NO_SUPPORTED_TIER,
                message="No configured tier is supported by all binding families.",
            ),
        )
    elif selected_tier is None:
        strongest_rejected = _strongest_rejected_tier(policies, leakage_decision)
        issues = (
            *issues,
            TierIssue(
                code=TierIssueCode.LEAKAGE_SEVERITY_EXCEEDS_TIER_POLICY,
                message="Sample-local leakage severity exceeds every family-supported tier policy.",
                tier=strongest_rejected,
                observed_value=leakage_decision.observed_max_severity,
                threshold_value=(
                    None
                    if strongest_rejected is None
                    else policies.policy_for(strongest_rejected).max_allowed_leakage_severity.value
                ),
            ),
        )
    return TierDecisionBundle(
        status=TierStatus.PASSED if selected_tier is not None else TierStatus.FAILED,
        selected_tier=selected_tier,
        family_decisions=family_decisions,
        leakage_decision=leakage_decision,
        issues=issues,
        sample_id=sample.source.sample_id,
        split=sample.source.split,
    )


def _evaluate_leakage_admissibility(
    leakage: LeakageSampleSummary,
    policies: TierPoliciesConfig,
) -> TierLeakageDecision:
    admissible: list[TierName] = []
    rejected: list[TierName] = []
    observed = leakage.max_severity.value
    for tier in TierName:
        allowed = policies.policy_for(tier).max_allowed_leakage_severity
        if _leakage_allowed(observed, allowed):
            admissible.append(tier)
        else:
            rejected.append(tier)
    return TierLeakageDecision(
        observed_max_severity=observed,
        admissible_tiers=tuple(admissible),
        rejected_tiers=tuple(rejected),
    )


def _leakage_allowed(
    observed: str,
    allowed: TierLeakageSeverity,
) -> bool:
    return _LEAKAGE_SEVERITY_RANK[observed] <= _LEAKAGE_SEVERITY_RANK[allowed.value]


def _select_best_family_and_leakage_supported_tier(
    *,
    policies: TierPoliciesConfig,
    supported_by_family: dict[BindingQualityFamily, tuple[TierName, ...]],
    binding_families: tuple[BindingQualityFamily, ...],
    leakage_decision: TierLeakageDecision,
) -> TierName | None:
    for tier in policies.strongest_first:
        if tier not in leakage_decision.admissible_tiers:
            continue
        if all(tier in supported_by_family.get(family, ()) for family in binding_families):
            return tier
    return None


def _strongest_rejected_tier(
    policies: TierPoliciesConfig,
    leakage_decision: TierLeakageDecision,
) -> TierName | None:
    for tier in policies.strongest_first:
        if tier in leakage_decision.rejected_tiers:
            return tier
    return None


def family_decision_by_name(
    bundle: TierDecisionBundle,
) -> dict[BindingQualityFamily, TierFamilyDecision]:
    """Return per-family decisions keyed by binding family."""
    return {
        cast(BindingQualityFamily, decision.family): decision
        for decision in bundle.family_decisions
    }


__all__ = ["evaluate_tier_decision", "family_decision_by_name"]
