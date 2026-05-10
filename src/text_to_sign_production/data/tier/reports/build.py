"""Quality report bundle construction from PreparedSample-based inputs."""

from __future__ import annotations

from text_to_sign_production.core.models import GateDecisionBundle, PreparedSample
from text_to_sign_production.data.tier.context import QualityContext
from text_to_sign_production.data.tier.facts import QualityFacts
from text_to_sign_production.data.tier.families import QualityMetricBundle
from text_to_sign_production.data.tier.leakages import LeakageBundle
from text_to_sign_production.data.tier.policies import TierDecisionBundle
from text_to_sign_production.data.tier.reports.sections import (
    build_gates_section,
    build_leakage_section,
    build_metrics_section,
    build_sample_section,
    build_tiers_section,
)
from text_to_sign_production.data.tier.reports.summaries import (
    build_quality_report_summary,
)
from text_to_sign_production.data.tier.reports.tables import build_quality_report_tables
from text_to_sign_production.data.tier.reports.types import QualityReportBundle
from text_to_sign_production.data.tier.reports.validate import validate_quality_report


def build_quality_report(
    sample: PreparedSample,
    facts: QualityFacts,
    context: QualityContext,
    metrics: QualityMetricBundle,
    leakage: LeakageBundle,
    tier_decisions: TierDecisionBundle,
    gate_decisions: GateDecisionBundle | None = None,
) -> QualityReportBundle:
    """Build the quality-stage report from PreparedSample-based inputs."""
    report = QualityReportBundle(
        summary=build_quality_report_summary(
            sample,
            facts,
            leakage,
            tier_decisions,
            gate_decisions,
        ),
        sections=(
            build_sample_section(sample, facts),
            build_metrics_section(metrics, context),
            build_leakage_section(leakage),
            build_tiers_section(tier_decisions),
            build_gates_section(gate_decisions),
        ),
        tables=build_quality_report_tables(metrics, tier_decisions, gate_decisions),
    )
    issues = validate_quality_report(report)
    if issues:
        raise ValueError(f"Invalid quality report: {issues}")
    return report


__all__ = ["build_quality_report"]
