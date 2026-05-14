"""Validation for gate-stage report projections."""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sign_production.data.gate.reports.types import GateReportBundle


@dataclass(frozen=True, slots=True)
class GateReportValidationIssue:
    """Structured gate report validation issue."""

    message: str
    field_path: str


def validate_gate_report_bundle(
    bundle: GateReportBundle,
) -> tuple[GateReportValidationIssue, ...]:
    """Validate a gate report bundle projection."""
    issues: list[GateReportValidationIssue] = []
    if not bundle.schema_version.strip():
        issues.append(GateReportValidationIssue("schema_version is blank.", "schema_version"))
    for path, count in (
        ("source_coverage.prepared_sample_count", bundle.source_coverage.prepared_sample_count),
        ("source_coverage.source_complete_count", bundle.source_coverage.source_complete_count),
        ("source_coverage.validation_issue_count", bundle.source_coverage.validation_issue_count),
        ("manifest_outcomes.passed_count", bundle.manifest_outcomes.passed_count),
        ("manifest_outcomes.dropped_count", bundle.manifest_outcomes.dropped_count),
        (
            "manifest_outcomes.passed_validation_issue_count",
            bundle.manifest_outcomes.passed_validation_issue_count,
        ),
        (
            "manifest_outcomes.dropped_validation_issue_count",
            bundle.manifest_outcomes.dropped_validation_issue_count,
        ),
        ("pose_health.prepared_payload_count", bundle.pose_health.prepared_payload_count),
        ("pose_health.total_frame_count", bundle.pose_health.total_frame_count),
        ("pose_health.total_valid_frame_count", bundle.pose_health.total_valid_frame_count),
        ("pose_health.pose_complete_count", bundle.pose_health.pose_complete_count),
        ("gate_outcomes.evaluated_count", bundle.gate_outcomes.evaluated_count),
        ("gate_outcomes.passed_count", bundle.gate_outcomes.passed_count),
        ("gate_outcomes.dropped_count", bundle.gate_outcomes.dropped_count),
        ("checkpoint_integrity.payload_count", bundle.checkpoint_integrity.payload_count),
        (
            "checkpoint_integrity.coherence_issue_count",
            bundle.checkpoint_integrity.coherence_issue_count,
        ),
        (
            "dropped_sample_payloads.dropped_total_count",
            bundle.dropped_sample_payloads.dropped_total_count,
        ),
        (
            "dropped_sample_payloads.source_dropped_sample_count",
            bundle.dropped_sample_payloads.source_dropped_sample_count,
        ),
        (
            "dropped_sample_payloads.pose_dropped_sample_count",
            bundle.dropped_sample_payloads.pose_dropped_sample_count,
        ),
        (
            "dropped_sample_payloads.gate_dropped_sample_count",
            bundle.dropped_sample_payloads.gate_dropped_sample_count,
        ),
        (
            "dropped_sample_payloads.dropped_sample_payload_written_count",
            bundle.dropped_sample_payloads.dropped_sample_payload_written_count,
        ),
        (
            "dropped_sample_payloads.dropped_manifest_entries_with_payload_ref_count",
            bundle.dropped_sample_payloads.dropped_manifest_entries_with_payload_ref_count,
        ),
        (
            "dropped_sample_payloads.dropped_manifest_entries_without_payload_ref_count",
            bundle.dropped_sample_payloads.dropped_manifest_entries_without_payload_ref_count,
        ),
        (
            "dropped_sample_payloads.dropped_manifest_payload_coherence_issue_count",
            bundle.dropped_sample_payloads.dropped_manifest_payload_coherence_issue_count,
        ),
    ):
        if count < 0:
            issues.append(GateReportValidationIssue("Count cannot be negative.", path))
    if (
        bundle.dropped_sample_payloads.dropped_manifest_entries_with_payload_ref_count
        + bundle.dropped_sample_payloads.dropped_manifest_entries_without_payload_ref_count
        != bundle.dropped_sample_payloads.dropped_total_count
    ):
        issues.append(
            GateReportValidationIssue(
                "Dropped manifest payload-ref counts must sum to total dropped count.",
                "dropped_sample_payloads",
            )
        )
    if not bundle.dropped_sample_payloads.dropped_manifest_payload_ref_count_coherent:
        issues.append(
            GateReportValidationIssue(
                "Dropped manifest payload refs must be count-coherent with written payload count.",
                "dropped_sample_payloads.dropped_manifest_payload_ref_count_coherent",
            )
        )
    if not bundle.dropped_sample_payloads.dropped_manifest_payload_identity_coherent:
        issues.append(
            GateReportValidationIssue(
                "Dropped manifest entries must be identity-coherent with DroppedSample payloads.",
                "dropped_sample_payloads.dropped_manifest_payload_identity_coherent",
            )
        )
    if bundle.pose_health.total_valid_frame_count > bundle.pose_health.total_frame_count:
        issues.append(
            GateReportValidationIssue(
                "Valid frame count cannot exceed total frame count.",
                "pose_health.total_valid_frame_count",
            )
        )
    return tuple(issues)


__all__ = ["GateReportValidationIssue", "validate_gate_report_bundle"]
