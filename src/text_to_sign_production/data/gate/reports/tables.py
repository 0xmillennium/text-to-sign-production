"""Table projections for samples-stage reports."""

from __future__ import annotations

from text_to_sign_production.data.gate.reports.types import SamplesReportBundle


def samples_report_tables(bundle: SamplesReportBundle) -> dict[str, tuple[dict[str, object], ...]]:
    """Project a samples report bundle into simple typed table rows."""
    return {
        "source_coverage": (
            {
                "sample_count": bundle.source_coverage.sample_count,
                "samples_with_source_issues": (bundle.source_coverage.samples_with_source_issues),
            },
        ),
        "split_counts": tuple(
            {"split": split, "count": count}
            for split, count in sorted(bundle.source_coverage.split_counts.items())
        ),
        "gate_outcomes": (
            {
                "evaluated_count": bundle.gate_outcomes.evaluated_count,
                "passed_count": bundle.gate_outcomes.passed_count,
                "dropped_count": bundle.gate_outcomes.dropped_count,
            },
        ),
        "failed_gate_counts": tuple(
            {"gate": gate, "count": count}
            for gate, count in sorted(bundle.gate_outcomes.failed_gate_counts.items())
        ),
        "checkpoint_integrity": (
            {
                "passed_manifest_count": (bundle.checkpoint_integrity.passed_manifest_count),
                "dropped_manifest_count": (bundle.checkpoint_integrity.dropped_manifest_count),
                "coherent_passed_count": (bundle.checkpoint_integrity.coherent_passed_count),
                "canonical_normalized_text_missing_count": (
                    bundle.checkpoint_integrity.canonical_normalized_text_missing_count
                ),
            },
        ),
    }


__all__ = ["samples_report_tables"]
