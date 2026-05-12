"""Read-only analysis helpers for tier reports."""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sign_production.data.tier.reports.types import TierReportBundle
from text_to_sign_production.data.tier.reports.validate import validate_tier_report


@dataclass(frozen=True, slots=True)
class TierReportAudit:
    """Compact tier report audit."""

    valid: bool
    validation_issue_count: int
    metric_row_count: int
    tier_row_count: int


def audit_tier_report(report: TierReportBundle) -> TierReportAudit:
    """Audit tier report projection integrity."""
    issues = validate_tier_report(report)
    return TierReportAudit(
        valid=not issues,
        validation_issue_count=len(issues),
        metric_row_count=len(report.tables.metric_rows),
        tier_row_count=len(report.tables.tier_rows),
    )


__all__ = ["TierReportAudit", "audit_tier_report"]
