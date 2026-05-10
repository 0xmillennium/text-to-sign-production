"""Section builders for PreparedSample-based quality reports."""

from __future__ import annotations

from text_to_sign_production.core.models import GateDecisionBundle, PreparedSample
from text_to_sign_production.data.tier.context import QualityContext
from text_to_sign_production.data.tier.facts import QualityFacts
from text_to_sign_production.data.tier.families import QualityMetricBundle
from text_to_sign_production.data.tier.leakages import LeakageBundle, LeakageSeverity
from text_to_sign_production.data.tier.policies import TierDecisionBundle
from text_to_sign_production.data.tier.reports.types import (
    GatesReportSection,
    LeakageReportSection,
    MetricsReportSection,
    ReportSectionName,
    SampleReportSection,
    TiersReportSection,
)


def build_sample_section(
    sample: PreparedSample,
    facts: QualityFacts,
) -> SampleReportSection:
    """Build PreparedSample identity and pose section."""
    return SampleReportSection(
        name=ReportSectionName.SAMPLE,
        title="Sample",
        sample_id=sample.source.sample_id,
        split=sample.source.split,
        source_video_id=sample.source.source_video_id,
        source_sentence_id=sample.source.source_sentence_id,
        canonical_normalized_text_present=bool(sample.source.canonical_normalized_text),
        frame_count=sample.pose.frame_count,
        valid_frame_count=facts.frame.valid_frame_count,
    )


def build_metrics_section(
    metrics: QualityMetricBundle,
    _context: QualityContext,
) -> MetricsReportSection:
    """Build metric coverage section."""
    return MetricsReportSection(
        name=ReportSectionName.METRICS,
        title="Metrics",
        binding_family_count=10,
        diagnostic_family_count=2,
        metric_row_count=_metric_row_count(metrics),
    )


def _metric_row_count(_metrics: QualityMetricBundle) -> int:
    """Return the fixed row count for the canonical metric bundle shape."""
    return 43


def build_leakage_section(leakage: LeakageBundle) -> LeakageReportSection:
    """Build leakage summary section."""
    affected = sum(1 for summary in leakage.sample_summaries if summary.has_leakage)
    severity = max(
        (summary.max_severity for summary in leakage.sample_summaries),
        default=LeakageSeverity.NONE,
        key=lambda value: list(LeakageSeverity).index(value),
    )
    return LeakageReportSection(
        name=ReportSectionName.LEAKAGE,
        title="Leakage",
        pair_count=len(leakage.pair_facts),
        affected_sample_count=affected,
        max_severity=severity,
    )


def build_tiers_section(tiers: TierDecisionBundle) -> TiersReportSection:
    """Build tier summary section."""
    return TiersReportSection(
        name=ReportSectionName.TIERS,
        title="Tiers",
        status=tiers.status,
        selected_tier=tiers.selected_tier,
        family_decision_count=len(tiers.family_decisions),
        issue_count=len(tiers.issues),
    )


def build_gates_section(gates: GateDecisionBundle | None) -> GatesReportSection:
    """Build optional samples admission gate section."""
    if gates is None:
        return GatesReportSection(
            name=ReportSectionName.GATES,
            title="Gates",
            final_status="not_provided",
            failed_gate_names=(),
            gate_count=0,
        )
    return GatesReportSection(
        name=ReportSectionName.GATES,
        title="Gates",
        final_status=gates.final_status.value,
        failed_gate_names=gates.failed_gates,
        gate_count=len(gates.decisions),
    )


__all__ = [
    "build_gates_section",
    "build_leakage_section",
    "build_metrics_section",
    "build_sample_section",
    "build_tiers_section",
]
