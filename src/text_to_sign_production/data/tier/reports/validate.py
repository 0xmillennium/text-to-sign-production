"""Validation for PreparedSample-based tier reports."""

from __future__ import annotations

from text_to_sign_production.data.tier.reports.types import (
    ReportSectionName,
    ReportValidationIssue,
    ReportValidationIssueCode,
    TierReportBundle,
)


def validate_tier_report(
    report: TierReportBundle,
) -> tuple[ReportValidationIssue, ...]:
    """Validate a tier report bundle."""
    issues: list[ReportValidationIssue] = []
    if not report.summary.sample_id:
        issues.append(
            ReportValidationIssue(
                ReportValidationIssueCode.BUILD_INPUT_INCOMPLETE,
                "Report summary sample_id is empty.",
                "summary.sample_id",
            )
        )
    section_names = tuple(section.name for section in report.sections)
    expected = tuple(ReportSectionName)
    if section_names != expected:
        issues.append(
            ReportValidationIssue(
                ReportValidationIssueCode.SECTION_MISSING,
                "Tier report sections must follow the canonical section order.",
                "sections",
            )
        )
    if report.summary.frame_count <= 0:
        issues.append(
            ReportValidationIssue(
                ReportValidationIssueCode.INVALID_REPORT,
                "Report frame_count must be positive.",
                "summary.frame_count",
            )
        )
    return tuple(issues)


__all__ = ["validate_tier_report"]
