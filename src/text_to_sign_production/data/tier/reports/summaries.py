"""Summary builders for PreparedSample-based tier reports."""

from __future__ import annotations

from text_to_sign_production.core.models import CheckpointAdmission, PreparedSample
from text_to_sign_production.data.tier.facts import QualityFacts
from text_to_sign_production.data.tier.leakages import LeakageSampleSummary
from text_to_sign_production.data.tier.policies.types import TierDecisionBundle
from text_to_sign_production.data.tier.reports.types import TierReportSummary


def build_tier_report_summary(
    sample: PreparedSample,
    facts: QualityFacts,
    leakage: LeakageSampleSummary,
    tier_decision: TierDecisionBundle,
    checkpoint_admission: CheckpointAdmission,
) -> TierReportSummary:
    """Build top-level report summary."""
    if (
        sample.source.sample_id != checkpoint_admission.sample_id
        or sample.source.split is not checkpoint_admission.split
    ):
        raise ValueError("Checkpoint admission identity must match the PreparedSample.")
    if not checkpoint_admission.source_manifest_sha256 or checkpoint_admission.payload_ref is None:
        raise ValueError("Checkpoint admission must carry manifest digest and payload_ref.")
    return TierReportSummary(
        sample_id=sample.source.sample_id,
        split=sample.source.split,
        frame_count=sample.pose.frame_count,
        duration_seconds=facts.text_length.duration_seconds,
        checkpoint_admission_status=checkpoint_admission.checkpoint_surface,
        source_manifest_path=checkpoint_admission.source_manifest_path,
        source_manifest_sha256=checkpoint_admission.source_manifest_sha256,
        payload_ref=checkpoint_admission.payload_ref,
        gate_detail_available=checkpoint_admission.gate_detail_available,
        gate_detail_status=checkpoint_admission.gate_detail_status,
        tier_status=tier_decision.status,
        selected_tier=tier_decision.selected_tier,
        max_leakage_severity=leakage.max_severity,
        diagnostic_highlights=(),
    )


__all__ = ["build_tier_report_summary"]
