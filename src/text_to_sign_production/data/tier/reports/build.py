"""Tier report bundle construction from PreparedSample-based inputs."""

from __future__ import annotations

from text_to_sign_production.core.models import CheckpointAdmission, PreparedSample
from text_to_sign_production.data.tier.facts import QualityFacts
from text_to_sign_production.data.tier.families.types import QualityMetricBundle
from text_to_sign_production.data.tier.leakages import LeakageSampleSummary
from text_to_sign_production.data.tier.policies.types import TierDecisionBundle
from text_to_sign_production.data.tier.reports.sections import (
    build_gates_section,
    build_leakage_section,
    build_metrics_section,
    build_sample_section,
    build_tier_outcomes_section,
)
from text_to_sign_production.data.tier.reports.summaries import (
    build_tier_report_summary,
)
from text_to_sign_production.data.tier.reports.tables import build_tier_report_tables
from text_to_sign_production.data.tier.reports.types import TierReportBundle
from text_to_sign_production.data.tier.reports.validate import validate_tier_report


def build_tier_report(
    sample: PreparedSample,
    facts: QualityFacts,
    metrics: QualityMetricBundle,
    leakage: LeakageSampleSummary,
    tier_decisions: TierDecisionBundle,
    checkpoint_admission: CheckpointAdmission,
) -> TierReportBundle:
    """Build the tier-stage report from checkpoint-admitted PreparedSample inputs."""
    report = TierReportBundle(
        summary=build_tier_report_summary(
            sample,
            facts,
            leakage,
            tier_decisions,
            checkpoint_admission,
        ),
        sections=(
            build_sample_section(sample, facts),
            build_metrics_section(metrics),
            build_leakage_section(leakage),
            build_tier_outcomes_section(tier_decisions),
            build_gates_section(checkpoint_admission),
        ),
        tables=build_tier_report_tables(metrics, tier_decisions, checkpoint_admission),
    )
    issues = validate_tier_report(report)
    if issues:
        raise ValueError(f"Invalid tier report: {issues}")
    return report


__all__ = ["build_tier_report"]
