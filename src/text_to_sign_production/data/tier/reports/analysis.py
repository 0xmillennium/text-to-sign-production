"""Read-only analysis helpers for quality reports."""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sign_production.data.tier.reports.types import QualityReportBundle
from text_to_sign_production.data.tier.reports.validate import validate_quality_report


@dataclass(frozen=True, slots=True)
class QualityReportAudit:
    """Compact quality report audit."""

    valid: bool
    validation_issue_count: int
    metric_row_count: int
    tier_row_count: int


def audit_quality_report(report: QualityReportBundle) -> QualityReportAudit:
    """Audit quality report projection integrity."""
    issues = validate_quality_report(report)
    return QualityReportAudit(
        valid=not issues,
        validation_issue_count=len(issues),
        metric_row_count=len(report.tables.metric_rows),
        tier_row_count=len(report.tables.tier_rows),
    )


__all__ = ["QualityReportAudit", "audit_quality_report"]
