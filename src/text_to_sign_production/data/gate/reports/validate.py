"""Validation for samples-stage report projections."""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sign_production.data.gate.reports.types import SamplesReportBundle


@dataclass(frozen=True, slots=True)
class SamplesReportValidationIssue:
    """Structured samples report validation issue."""

    message: str
    field_path: str


def validate_samples_report_bundle(
    bundle: SamplesReportBundle,
) -> tuple[SamplesReportValidationIssue, ...]:
    """Validate a samples report bundle projection."""
    issues: list[SamplesReportValidationIssue] = []
    if not bundle.schema_version.strip():
        issues.append(SamplesReportValidationIssue("schema_version is blank.", "schema_version"))
    for path, count in (
        ("source_coverage.sample_count", bundle.source_coverage.sample_count),
        ("matching_outcomes.matched_count", bundle.matching_outcomes.matched_count),
        ("matching_outcomes.dropped_count", bundle.matching_outcomes.dropped_count),
        ("pose_health.payload_count", bundle.pose_health.payload_count),
        ("pose_health.total_frame_count", bundle.pose_health.total_frame_count),
        ("pose_health.total_valid_frame_count", bundle.pose_health.total_valid_frame_count),
        ("gate_outcomes.evaluated_count", bundle.gate_outcomes.evaluated_count),
        ("gate_outcomes.passed_count", bundle.gate_outcomes.passed_count),
        ("gate_outcomes.dropped_count", bundle.gate_outcomes.dropped_count),
        (
            "checkpoint_integrity.canonical_normalized_text_missing_count",
            bundle.checkpoint_integrity.canonical_normalized_text_missing_count,
        ),
    ):
        if count < 0:
            issues.append(SamplesReportValidationIssue("Count cannot be negative.", path))
    if bundle.pose_health.total_valid_frame_count > bundle.pose_health.total_frame_count:
        issues.append(
            SamplesReportValidationIssue(
                "Valid frame count cannot exceed total frame count.",
                "pose_health.total_valid_frame_count",
            )
        )
    return tuple(issues)


__all__ = ["SamplesReportValidationIssue", "validate_samples_report_bundle"]
