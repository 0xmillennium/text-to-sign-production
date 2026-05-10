"""Thin orchestration for PreparedSample-based tier policy evaluation."""

from __future__ import annotations

from text_to_sign_production.core.ids import SampleStatus
from text_to_sign_production.core.models import GateDecisionBundle, PreparedSample
from text_to_sign_production.data.tier.context import QualityContext
from text_to_sign_production.data.tier.facts import QualityFacts
from text_to_sign_production.data.tier.families import BindingQualityFamily, QualityMetricBundle
from text_to_sign_production.data.tier.leakages import LeakageBundle
from text_to_sign_production.data.tier.policies.filters import TierFiltersConfig
from text_to_sign_production.data.tier.policies.policies import TierPoliciesConfig
from text_to_sign_production.data.tier.policies.roles import (
    binding_family_roles,
    required_policy_families,
)
from text_to_sign_production.data.tier.policies.types import (
    FamilyTierDecision,
    TierDecisionBundle,
    TierEvaluationInput,
    TierIssue,
    TierIssueCode,
    TierStatus,
)


def evaluate_quality_tiers(
    sample: PreparedSample,
    facts: QualityFacts,
    context: QualityContext,
    metrics: QualityMetricBundle,
    leakage: LeakageBundle,
    gate_decisions: GateDecisionBundle,
    filters: TierFiltersConfig,
    policies: TierPoliciesConfig,
) -> TierDecisionBundle:
    """Evaluate quality tier banding from PreparedSample-derived inputs."""
    input = TierEvaluationInput(
        sample=sample,
        quality_facts=facts,
        quality_context=context,
        quality_metrics=metrics,
        leakage=leakage,
        gate_decisions=gate_decisions,
    )
    if gate_decisions.final_status is not SampleStatus.PASSED:
        return _skipped_for_gate_failure(input)

    family_decisions = tuple(
        role.evaluator(
            getattr(input.quality_metrics, role.metric_attr),
            getattr(filters, role.filter_attr),
            policies,
        )
        for role in binding_family_roles()
    )
    supported_by_family = {
        decision.family: decision.supported_tiers for decision in family_decisions
    }
    binding_families = required_policy_families(policies)
    selected_tier = policies.select_best_supported_tier(supported_by_family, binding_families)
    issues = tuple(issue for decision in family_decisions for issue in decision.issues)
    if selected_tier is None:
        issues = (
            *issues,
            TierIssue(
                code=TierIssueCode.NO_SUPPORTED_TIER,
                message="No configured tier is supported by all binding families.",
            ),
        )
    return TierDecisionBundle(
        status=TierStatus.PASSED if selected_tier is not None else TierStatus.FAILED,
        selected_tier=selected_tier,
        family_decisions=family_decisions,
        issues=issues,
    )


def _skipped_for_gate_failure(input: TierEvaluationInput) -> TierDecisionBundle:
    issues = tuple(
        TierIssue(
            code=TierIssueCode.GATE_FAILED,
            message="Tier banding skipped because a samples admission gate failed.",
            gate_name=gate_name,
        )
        for gate_name in input.gate_decisions.failed_gates
    )
    return TierDecisionBundle(
        status=TierStatus.SKIPPED,
        selected_tier=None,
        family_decisions=(),
        issues=issues,
    )


def family_decision_by_name(
    bundle: TierDecisionBundle,
) -> dict[BindingQualityFamily, FamilyTierDecision]:
    """Return per-family decisions keyed by binding family."""
    return {decision.family: decision for decision in bundle.family_decisions}


__all__ = ["evaluate_quality_tiers", "family_decision_by_name"]
