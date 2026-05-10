"""Summary builders for PreparedSample-based quality reports."""

from __future__ import annotations

from text_to_sign_production.core.models import GateDecisionBundle, PreparedSample
from text_to_sign_production.data.tier.facts import QualityFacts
from text_to_sign_production.data.tier.leakages import LeakageBundle, LeakageSeverity
from text_to_sign_production.data.tier.policies import TierDecisionBundle
from text_to_sign_production.data.tier.reports.types import QualityReportSummary


def build_quality_report_summary(
    sample: PreparedSample,
    facts: QualityFacts,
    leakage: LeakageBundle,
    tiers: TierDecisionBundle,
    gates: GateDecisionBundle | None,
) -> QualityReportSummary:
    """Build top-level report summary."""
    max_severity = max(
        (summary.max_severity for summary in leakage.sample_summaries),
        default=LeakageSeverity.NONE,
        key=lambda value: list(LeakageSeverity).index(value),
    )
    return QualityReportSummary(
        sample_id=sample.source.sample_id,
        split=sample.source.split,
        frame_count=sample.pose.frame_count,
        duration_seconds=facts.text_length.duration_seconds,
        gate_status=None if gates is None else gates.final_status.value,
        tier_status=tiers.status,
        selected_tier=tiers.selected_tier,
        max_leakage_severity=max_severity,
        diagnostic_highlights=(),
    )


__all__ = ["build_quality_report_summary"]
