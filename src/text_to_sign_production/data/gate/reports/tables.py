"""Table projections for gate-stage reports."""

from __future__ import annotations

from text_to_sign_production.data.gate.reports.types import (
    GateCheckpointIntegrityTableRow,
    GateDroppedDebugPayloadTableRow,
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
        dropped_debug_payloads=(
            GateDroppedDebugPayloadTableRow(
                materialize_dropped_debug_payloads=(
                    bundle.dropped_debug_payloads.materialize_dropped_debug_payloads
                ),
                dropped_total_count=bundle.dropped_debug_payloads.dropped_total_count,
                pose_or_source_dropped_without_prepared_payload_count=(
                    bundle.dropped_debug_payloads
                    .pose_or_source_dropped_without_prepared_payload_count
                ),
                gate_dropped_prepared_sample_count=(
                    bundle.dropped_debug_payloads.gate_dropped_prepared_sample_count
                ),
                dropped_debug_payload_written_count=(
                    bundle.dropped_debug_payloads.dropped_debug_payload_written_count
                ),
                dropped_manifest_entries_with_debug_ref_count=(
                    bundle.dropped_debug_payloads
                    .dropped_manifest_entries_with_debug_ref_count
                ),
                dropped_manifest_entries_without_debug_ref_count=(
                    bundle.dropped_debug_payloads
                    .dropped_manifest_entries_without_debug_ref_count
                ),
            ),
        ),
    )


__all__ = ["gate_report_tables"]
