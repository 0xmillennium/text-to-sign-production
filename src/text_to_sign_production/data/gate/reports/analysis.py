"""Read-only analysis helpers for samples-stage report projections."""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sign_production.data.gate.reports.types import SamplesReportBundle
from text_to_sign_production.data.gate.reports.validate import (
    validate_samples_report_bundle,
)


@dataclass(frozen=True, slots=True)
class SamplesReportAudit:
    """Compact audit of report projection integrity."""

    valid: bool
    validation_issue_count: int
    canonical_normalized_text_complete: bool


def audit_samples_report(bundle: SamplesReportBundle) -> SamplesReportAudit:
    """Audit report projection integrity."""
    issues = validate_samples_report_bundle(bundle)
    return SamplesReportAudit(
        valid=not issues,
        validation_issue_count=len(issues),
        canonical_normalized_text_complete=(
            bundle.checkpoint_integrity.canonical_normalized_text_missing_count == 0
        ),
    )


__all__ = ["SamplesReportAudit", "audit_samples_report"]
