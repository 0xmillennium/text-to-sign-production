"""Summary helpers for samples-stage report projections."""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sign_production.data.gate.reports.types import SamplesReportBundle


@dataclass(frozen=True, slots=True)
class SamplesReportSummary:
    """Compact summary of a samples report bundle."""

    sample_count: int
    passed_count: int
    dropped_count: int
    gate_passed_count: int
    checkpoint_canonical_text_complete: bool


def summarize_samples_report(bundle: SamplesReportBundle) -> SamplesReportSummary:
    """Build a compact summary of a samples report bundle."""
    return SamplesReportSummary(
        sample_count=bundle.source_coverage.sample_count,
        passed_count=bundle.checkpoint_integrity.passed_manifest_count,
        dropped_count=bundle.checkpoint_integrity.dropped_manifest_count,
        gate_passed_count=bundle.gate_outcomes.passed_count,
        checkpoint_canonical_text_complete=(
            bundle.checkpoint_integrity.canonical_normalized_text_missing_count == 0
        ),
    )


__all__ = ["SamplesReportSummary", "summarize_samples_report"]
