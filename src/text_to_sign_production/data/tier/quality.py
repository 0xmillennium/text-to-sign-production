"""Data-owned quality continuation for tier processing."""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sign_production.core.models import PreparedSample
from text_to_sign_production.data.tier.context import (
    QualityContext,
    build_quality_context,
    validate_quality_context,
)
from text_to_sign_production.data.tier.facts import (
    QualityFacts,
    build_quality_facts,
    validate_quality_facts_invariants,
)
from text_to_sign_production.data.tier.families.compute import build_quality_metric_bundle
from text_to_sign_production.data.tier.families.types import QualityMetricBundle
from text_to_sign_production.data.tier.families.validate import validate_quality_metric_bundle


class TierQualityComputationError(ValueError):
    """Raised when data-owned tier quality computation violates invariants."""


@dataclass(frozen=True, slots=True)
class TierQualityComputation:
    """Validated PreparedSample-derived quality truth for tier evaluation."""

    facts: QualityFacts
    context: QualityContext
    metrics: QualityMetricBundle


def compute_tier_quality(sample: PreparedSample) -> TierQualityComputation:
    """Compute and validate tier quality facts, context, and metric bundle."""
    facts = build_quality_facts(sample)
    fact_issues = validate_quality_facts_invariants(facts)
    if fact_issues:
        raise TierQualityComputationError(f"Quality facts validation failed: {fact_issues}")

    context = build_quality_context(sample, facts)
    context_issues = validate_quality_context(context)
    if context_issues:
        raise TierQualityComputationError(f"Quality context validation failed: {context_issues}")

    metrics = build_quality_metric_bundle(sample, facts, context)
    metric_issues = validate_quality_metric_bundle(metrics)
    if metric_issues:
        raise TierQualityComputationError(
            f"Quality metric bundle validation failed: {metric_issues}"
        )

    return TierQualityComputation(facts=facts, context=context, metrics=metrics)


__all__ = [
    "TierQualityComputation",
    "TierQualityComputationError",
    "compute_tier_quality",
]
