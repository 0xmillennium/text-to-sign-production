"""Samples-stage report projection package."""

from text_to_sign_production.data.gate.reports.analysis import (
    SamplesReportAudit,
    audit_samples_report,
)
from text_to_sign_production.data.gate.reports.build import build_samples_report_bundle
from text_to_sign_production.data.gate.reports.sections import (
    build_checkpoint_integrity_section,
    build_gate_outcomes_section,
    build_matching_outcomes_section,
    build_pose_health_section,
    build_source_coverage_section,
)
from text_to_sign_production.data.gate.reports.summaries import (
    SamplesReportSummary,
    summarize_samples_report,
)
from text_to_sign_production.data.gate.reports.tables import samples_report_tables
from text_to_sign_production.data.gate.reports.types import (
    CheckpointIntegritySection,
    GateOutcomesSection,
    MatchingOutcomesSection,
    PoseHealthSection,
    SamplesReportBundle,
    SourceCoverageSection,
)
from text_to_sign_production.data.gate.reports.validate import (
    SamplesReportValidationIssue,
    validate_samples_report_bundle,
)

__all__ = [
    "CheckpointIntegritySection",
    "GateOutcomesSection",
    "MatchingOutcomesSection",
    "PoseHealthSection",
    "SamplesReportAudit",
    "SamplesReportBundle",
    "SamplesReportSummary",
    "SamplesReportValidationIssue",
    "SourceCoverageSection",
    "audit_samples_report",
    "build_checkpoint_integrity_section",
    "build_gate_outcomes_section",
    "build_matching_outcomes_section",
    "build_pose_health_section",
    "build_samples_report_bundle",
    "build_source_coverage_section",
    "samples_report_tables",
    "summarize_samples_report",
    "validate_samples_report_bundle",
]
