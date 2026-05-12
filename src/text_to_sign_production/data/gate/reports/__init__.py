"""Gate-stage report projection package."""

from text_to_sign_production.data.gate.reports.analysis import (
    GateReportAudit,
    audit_gate_report,
)
from text_to_sign_production.data.gate.reports.build import build_gate_report_bundle
from text_to_sign_production.data.gate.reports.sections import (
    build_checkpoint_integrity_section,
    build_dropped_debug_payload_section,
    build_gate_outcomes_section,
    build_manifest_outcomes_section,
    build_pose_health_section,
    build_source_coverage_section,
)
from text_to_sign_production.data.gate.reports.summaries import (
    GateReportSummary,
    summarize_gate_report,
)
from text_to_sign_production.data.gate.reports.tables import gate_report_tables
from text_to_sign_production.data.gate.reports.types import (
    CheckpointIntegritySection,
    DroppedDebugPayloadSection,
    GateCheckpointIntegrityTableRow,
    GateDroppedDebugPayloadTableRow,
    GateFailedGateCountTableRow,
    GateManifestOutcomeTableRow,
    GateOutcomesSection,
    GateOutcomeTableRow,
    GateReportBundle,
    GateReportTables,
    GateSourceCoverageTableRow,
    GateSplitCountTableRow,
    ManifestOutcomesSection,
    PoseHealthSection,
    SourceCoverageSection,
)
from text_to_sign_production.data.gate.reports.validate import (
    GateReportValidationIssue,
    validate_gate_report_bundle,
)

__all__ = [
    "CheckpointIntegritySection",
    "DroppedDebugPayloadSection",
    "GateCheckpointIntegrityTableRow",
    "GateDroppedDebugPayloadTableRow",
    "GateFailedGateCountTableRow",
    "GateManifestOutcomeTableRow",
    "GateOutcomesSection",
    "GateOutcomeTableRow",
    "ManifestOutcomesSection",
    "PoseHealthSection",
    "GateReportAudit",
    "GateReportBundle",
    "GateReportTables",
    "GateReportSummary",
    "GateReportValidationIssue",
    "GateSourceCoverageTableRow",
    "GateSplitCountTableRow",
    "SourceCoverageSection",
    "audit_gate_report",
    "build_checkpoint_integrity_section",
    "build_dropped_debug_payload_section",
    "build_gate_outcomes_section",
    "build_manifest_outcomes_section",
    "build_pose_health_section",
    "build_gate_report_bundle",
    "build_source_coverage_section",
    "gate_report_tables",
    "summarize_gate_report",
    "validate_gate_report_bundle",
]
