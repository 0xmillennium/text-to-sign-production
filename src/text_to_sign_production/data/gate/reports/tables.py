"""Table projections for gate-stage reports."""

from __future__ import annotations

from text_to_sign_production.data.gate.reports.types import (
    GateCheckpointIntegrityTableRow,
    GateDroppedSamplePayloadTableRow,
    GateFailedGateCountTableRow,
    GateManifestOutcomeTableRow,
    GateOutcomeTableRow,
    GateReportBundle,
    GateReportTables,
    GateSourceCoverageTableRow,
    GateSplitCountTableRow,
)


def gate_report_tables(bundle: GateReportBundle) -> GateReportTables:
    """Project a gate report bundle into simple typed table rows."""
    return GateReportTables(
        source_coverage=(
            GateSourceCoverageTableRow(
                prepared_sample_count=bundle.source_coverage.prepared_sample_count,
                prepared_samples_with_source_issues=(
                    bundle.source_coverage.prepared_samples_with_source_issues
                ),
                source_complete_count=bundle.source_coverage.source_complete_count,
                validation_issue_count=bundle.source_coverage.validation_issue_count,
            ),
        ),
        split_counts=tuple(
            GateSplitCountTableRow(split=split, count=count)
            for split, count in sorted(bundle.source_coverage.split_counts.items())
        ),
        gate_outcomes=(
            GateOutcomeTableRow(
                evaluated_count=bundle.gate_outcomes.evaluated_count,
                passed_count=bundle.gate_outcomes.passed_count,
                dropped_count=bundle.gate_outcomes.dropped_count,
            ),
        ),
        manifest_outcomes=(
            GateManifestOutcomeTableRow(
                passed_count=bundle.manifest_outcomes.passed_count,
                dropped_count=bundle.manifest_outcomes.dropped_count,
                passed_validation_issue_count=(
                    bundle.manifest_outcomes.passed_validation_issue_count
                ),
                dropped_validation_issue_count=(
                    bundle.manifest_outcomes.dropped_validation_issue_count
                ),
            ),
        ),
        failed_gate_counts=tuple(
            GateFailedGateCountTableRow(gate=gate, count=count)
            for gate, count in sorted(bundle.gate_outcomes.failed_gate_counts.items())
        ),
        checkpoint_integrity=(
            GateCheckpointIntegrityTableRow(
                payload_count=bundle.checkpoint_integrity.payload_count,
                passed_manifest_count=bundle.checkpoint_integrity.passed_manifest_count,
                dropped_manifest_count=bundle.checkpoint_integrity.dropped_manifest_count,
                coherent_passed_count=bundle.checkpoint_integrity.coherent_passed_count,
                coherence_issue_count=bundle.checkpoint_integrity.coherence_issue_count,
            ),
        ),
        dropped_sample_payloads=(
            GateDroppedSamplePayloadTableRow(
                dropped_total_count=bundle.dropped_sample_payloads.dropped_total_count,
                source_dropped_sample_count=(
                    bundle.dropped_sample_payloads.source_dropped_sample_count
                ),
                pose_dropped_sample_count=(
                    bundle.dropped_sample_payloads.pose_dropped_sample_count
                ),
                gate_dropped_sample_count=(
                    bundle.dropped_sample_payloads.gate_dropped_sample_count
                ),
                dropped_sample_payload_written_count=(
                    bundle.dropped_sample_payloads.dropped_sample_payload_written_count
                ),
                dropped_manifest_entries_with_payload_ref_count=(
                    bundle.dropped_sample_payloads.dropped_manifest_entries_with_payload_ref_count
                ),
                dropped_manifest_entries_without_payload_ref_count=(
                    bundle.dropped_sample_payloads.dropped_manifest_entries_without_payload_ref_count
                ),
                dropped_manifest_payload_ref_count_coherent=(
                    bundle.dropped_sample_payloads.dropped_manifest_payload_ref_count_coherent
                ),
                dropped_manifest_payload_identity_coherent=(
                    bundle.dropped_sample_payloads.dropped_manifest_payload_identity_coherent
                ),
                dropped_manifest_payload_coherence_issue_count=(
                    bundle.dropped_sample_payloads.dropped_manifest_payload_coherence_issue_count
                ),
            ),
        ),
    )


__all__ = ["gate_report_tables"]
