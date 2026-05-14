"""Summary helpers for gate-stage report projections."""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sign_production.data.gate.reports.types import GateReportBundle


@dataclass(frozen=True, slots=True)
class GateReportSummary:
    """Compact summary of a gate report bundle."""

    prepared_sample_count: int
    passed_count: int
    dropped_count: int
    gate_passed_count: int
    dropped_sample_payload_written_count: int


def summarize_gate_report(bundle: GateReportBundle) -> GateReportSummary:
    """Build a compact summary of a gate report bundle."""
    return GateReportSummary(
        prepared_sample_count=bundle.source_coverage.prepared_sample_count,
        passed_count=bundle.checkpoint_integrity.passed_manifest_count,
        dropped_count=bundle.checkpoint_integrity.dropped_manifest_count,
        gate_passed_count=bundle.gate_outcomes.passed_count,
        dropped_sample_payload_written_count=(
            bundle.dropped_sample_payloads.dropped_sample_payload_written_count
        ),
    )


__all__ = ["GateReportSummary", "summarize_gate_report"]
