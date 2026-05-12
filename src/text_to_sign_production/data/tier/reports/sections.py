"""Section builders for PreparedSample-based tier reports."""

from __future__ import annotations

from text_to_sign_production.core.models import CheckpointAdmission, PreparedSample
from text_to_sign_production.data.dataset.analysis import summarize_prepared_sample_payload
from text_to_sign_production.data.tier.facts import QualityFacts
from text_to_sign_production.data.tier.families.analysis import summarize_quality_metric_bundle
from text_to_sign_production.data.tier.families.types import QualityMetricBundle
from text_to_sign_production.data.tier.leakages import LeakageSampleSummary
from text_to_sign_production.data.tier.policies.types import TierDecisionBundle
from text_to_sign_production.data.tier.reports.types import (
    GatesReportSection,
    LeakageReportSection,
    MetricsReportSection,
    ReportSectionName,
    SampleReportSection,
    TierOutcomesSection,
)


def build_sample_section(
    sample: PreparedSample,
    facts: QualityFacts,
) -> SampleReportSection:
    """Build PreparedSample identity and pose section."""
    sample_summary = summarize_prepared_sample_payload(sample)
    return SampleReportSection(
        name=ReportSectionName.SAMPLE,
        title="Sample",
        sample_id=sample.source.sample_id,
        split=sample.source.split,
        source_video_id=sample.source.source_video_id,
        source_sentence_id=sample.source.source_sentence_id,
        frame_count=sample.pose.frame_count,
        valid_frame_count=facts.frame.valid_frame_count,
        source_complete=sample_summary.source_complete,
        pose_complete=sample_summary.pose_complete,
        validation_issue_count=sample_summary.validation_issue_count,
    )


def build_metrics_section(
    metrics: QualityMetricBundle,
) -> MetricsReportSection:
    """Build metric coverage section."""
    summary = summarize_quality_metric_bundle(metrics)
    return MetricsReportSection(
        name=ReportSectionName.METRICS,
        title="Metrics",
        binding_family_count=summary.binding_family_count,
        diagnostic_family_count=summary.diagnostic_family_count,
        metric_row_count=summary.metric_count,
    )


def build_leakage_section(leakage: LeakageSampleSummary) -> LeakageReportSection:
    """Build sample-local leakage summary section."""
    return LeakageReportSection(
        name=ReportSectionName.LEAKAGE,
        title="Leakage",
        pair_count=len(leakage.matched_samples),
        affected_sample_count=1 if leakage.has_leakage else 0,
        max_severity=leakage.max_severity,
    )


def build_tier_outcomes_section(tier_decision: TierDecisionBundle) -> TierOutcomesSection:
    """Build tier summary section."""
    leakage = tier_decision.leakage_decision
    return TierOutcomesSection(
        name=ReportSectionName.TIER,
        title="Tier",
        status=tier_decision.status,
        selected_tier=tier_decision.selected_tier,
        family_decision_count=len(tier_decision.family_decisions),
        leakage_observed_max_severity=(
            None if leakage is None else leakage.observed_max_severity
        ),
        leakage_admissible_tiers=() if leakage is None else leakage.admissible_tiers,
        leakage_rejected_tiers=() if leakage is None else leakage.rejected_tiers,
        issue_count=len(tier_decision.issues),
    )


def build_gates_section(checkpoint_admission: CheckpointAdmission) -> GatesReportSection:
    """Build checkpoint admission section without fabricating gate predicate detail."""
    if checkpoint_admission.gate_detail_available:
        raise ValueError("Checkpoint-only tier flow cannot claim gate detail availability.")
    return GatesReportSection(
        name=ReportSectionName.GATES,
        title="Gates",
        sample_id=checkpoint_admission.sample_id,
        split=checkpoint_admission.split,
        checkpoint_admission_status=checkpoint_admission.checkpoint_surface,
        source_manifest_path=checkpoint_admission.source_manifest_path,
        source_manifest_sha256=checkpoint_admission.source_manifest_sha256,
        payload_ref=checkpoint_admission.payload_ref,
        gate_detail_available=checkpoint_admission.gate_detail_available,
        gate_detail_status=checkpoint_admission.gate_detail_status,
        failed_gate_names=(),
        gate_count=0,
    )


__all__ = [
    "build_gates_section",
    "build_leakage_section",
    "build_metrics_section",
    "build_sample_section",
    "build_tier_outcomes_section",
]
