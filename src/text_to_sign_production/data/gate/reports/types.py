"""Typed gate-stage report projection products."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceCoverageSection:
    """Source coverage projection for the gate stage."""

    prepared_sample_count: int
    split_counts: dict[str, int]
    prepared_samples_with_source_issues: int
    source_complete_count: int
    validation_issue_count: int


@dataclass(frozen=True, slots=True)
class ManifestOutcomesSection:
    """Final manifest row outcome projection for the gate stage."""

    passed_count: int
    dropped_count: int
    passed_validation_issue_count: int
    dropped_validation_issue_count: int


@dataclass(frozen=True, slots=True)
class PoseHealthSection:
    """Pose health projection for the gate stage."""

    prepared_payload_count: int
    total_frame_count: int
    total_valid_frame_count: int
    pose_complete_count: int


@dataclass(frozen=True, slots=True)
class GateOutcomesSection:
    """Gate outcome projection for the gate stage."""

    evaluated_count: int
    passed_count: int
    dropped_count: int
    failed_gate_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class CheckpointIntegritySection:
    """Checkpoint integrity projection for the gate stage."""

    payload_count: int
    passed_manifest_count: int
    dropped_manifest_count: int
    coherent_passed_count: int
    coherence_issue_count: int


@dataclass(frozen=True, slots=True)
class DroppedSamplePayloadSection:
    """Dropped sample payload projection for the gate stage."""

    dropped_total_count: int
    source_dropped_sample_count: int
    pose_dropped_sample_count: int
    gate_dropped_sample_count: int
    dropped_sample_payload_written_count: int
    dropped_manifest_entries_with_payload_ref_count: int
    dropped_manifest_entries_without_payload_ref_count: int
    dropped_manifest_payload_ref_count_coherent: bool
    dropped_manifest_payload_identity_coherent: bool
    dropped_manifest_payload_coherence_issue_count: int


@dataclass(frozen=True, slots=True)
class GateReportBundle:
    """Root gate-stage report bundle."""

    schema_version: str
    source_coverage: SourceCoverageSection
    manifest_outcomes: ManifestOutcomesSection
    pose_health: PoseHealthSection
    gate_outcomes: GateOutcomesSection
    checkpoint_integrity: CheckpointIntegritySection
    dropped_sample_payloads: DroppedSamplePayloadSection


@dataclass(frozen=True, slots=True)
class GateSourceCoverageTableRow:
    """Table row for source coverage report output."""

    prepared_sample_count: int
    prepared_samples_with_source_issues: int
    source_complete_count: int
    validation_issue_count: int


@dataclass(frozen=True, slots=True)
class GateSplitCountTableRow:
    """Table row for source split counts."""

    split: str
    count: int


@dataclass(frozen=True, slots=True)
class GateOutcomeTableRow:
    """Table row for aggregate gate outcomes."""

    evaluated_count: int
    passed_count: int
    dropped_count: int


@dataclass(frozen=True, slots=True)
class GateManifestOutcomeTableRow:
    """Table row for final manifest outcome counts."""

    passed_count: int
    dropped_count: int
    passed_validation_issue_count: int
    dropped_validation_issue_count: int


@dataclass(frozen=True, slots=True)
class GateFailedGateCountTableRow:
    """Table row for failed-gate counts."""

    gate: str
    count: int


@dataclass(frozen=True, slots=True)
class GateCheckpointIntegrityTableRow:
    """Table row for checkpoint integrity counts."""

    payload_count: int
    passed_manifest_count: int
    dropped_manifest_count: int
    coherent_passed_count: int
    coherence_issue_count: int


@dataclass(frozen=True, slots=True)
class GateDroppedSamplePayloadTableRow:
    """Table row for dropped sample payload clarity."""

    dropped_total_count: int
    source_dropped_sample_count: int
    pose_dropped_sample_count: int
    gate_dropped_sample_count: int
    dropped_sample_payload_written_count: int
    dropped_manifest_entries_with_payload_ref_count: int
    dropped_manifest_entries_without_payload_ref_count: int
    dropped_manifest_payload_ref_count_coherent: bool
    dropped_manifest_payload_identity_coherent: bool
    dropped_manifest_payload_coherence_issue_count: int


@dataclass(frozen=True, slots=True)
class GateReportTables:
    """Typed machine-readable table surfaces for a gate report."""

    source_coverage: tuple[GateSourceCoverageTableRow, ...]
    split_counts: tuple[GateSplitCountTableRow, ...]
    gate_outcomes: tuple[GateOutcomeTableRow, ...]
    manifest_outcomes: tuple[GateManifestOutcomeTableRow, ...]
    failed_gate_counts: tuple[GateFailedGateCountTableRow, ...]
    checkpoint_integrity: tuple[GateCheckpointIntegrityTableRow, ...]
    dropped_sample_payloads: tuple[GateDroppedSamplePayloadTableRow, ...]


__all__ = [
    "CheckpointIntegritySection",
    "DroppedSamplePayloadSection",
    "GateCheckpointIntegrityTableRow",
    "GateDroppedSamplePayloadTableRow",
    "GateFailedGateCountTableRow",
    "GateManifestOutcomeTableRow",
    "GateOutcomesSection",
    "GateOutcomeTableRow",
    "ManifestOutcomesSection",
    "PoseHealthSection",
    "GateReportBundle",
    "GateReportTables",
    "GateSourceCoverageTableRow",
    "GateSplitCountTableRow",
    "SourceCoverageSection",
]
