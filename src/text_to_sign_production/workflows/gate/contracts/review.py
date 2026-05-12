from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.workflows.gate.contracts.config import (
    GateWorkflowInputError,
)


@dataclass(frozen=True, slots=True)
class GateRuntimeAssetRow:
    label: str
    path: Path
    exists: bool
    valid: bool = True
    message: str | None = None
    scope: str = "asset"

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        object.__setattr__(self, "label", self.label.strip())


@dataclass(frozen=True, slots=True)
class GateSplitCountRow:
    split: str
    processed_count: int
    passed_count: int
    dropped_count: int

    def __post_init__(self) -> None:
        _validate_non_empty_text("split", self.split)
        _ensure_non_negative("processed_count", self.processed_count)
        _ensure_non_negative("passed_count", self.passed_count)
        _ensure_non_negative("dropped_count", self.dropped_count)
        object.__setattr__(self, "split", self.split.strip())


@dataclass(frozen=True, slots=True)
class GateIdentityInvariantRow:
    split: str
    sample_id: str
    sentence_id: str
    status: str
    issue_count: int

    def __post_init__(self) -> None:
        _validate_non_empty_text("split", self.split)
        _validate_non_empty_text("sample_id", self.sample_id)
        _validate_non_empty_text("sentence_id", self.sentence_id)
        _validate_non_empty_text("status", self.status)
        _ensure_non_negative("issue_count", self.issue_count)


@dataclass(frozen=True, slots=True)
class GateViabilityDropRow:
    split: str
    sample_id: str
    sentence_id: str
    viability_status: str
    issue_codes: tuple[str, ...]
    dropped: bool
    drop_stage: str | None

    def __post_init__(self) -> None:
        _validate_non_empty_text("split", self.split)
        _validate_non_empty_text("sample_id", self.sample_id)
        _validate_non_empty_text("sentence_id", self.sentence_id)
        _validate_non_empty_text("viability_status", self.viability_status)
        object.__setattr__(
            self,
            "issue_codes",
            tuple(code.strip() for code in self.issue_codes if code.strip()),
        )


@dataclass(frozen=True, slots=True)
class GateWrittenArtifactRow:
    label: str
    kind: str
    path: Path
    sha256: str
    execution_id: str
    sample_id: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        _validate_non_empty_text("kind", self.kind)
        _validate_non_empty_text("sha256", self.sha256)
        _validate_non_empty_text("execution_id", self.execution_id)


@dataclass(frozen=True, slots=True)
class GateProcessingSummaryRecord:
    split: str
    processed_count: int
    prepared_sample_count: int
    passed_count: int
    dropped_count: int
    materialize_dropped_debug_payloads: bool
    pose_or_source_dropped_without_prepared_payload_count: int
    gate_dropped_prepared_sample_count: int
    dropped_debug_payload_written_count: int
    dropped_manifest_entries_with_debug_ref_count: int
    dropped_manifest_entries_without_debug_ref_count: int


@dataclass(frozen=True, slots=True)
class GateProcessingDetailRecord:
    split: str
    sample_id: str
    match_status: str
    passed: bool
    drop_stage: str | None


@dataclass(frozen=True, slots=True)
class GateSummaryRecord:
    split: str
    evaluated_count: int
    passed_count: int
    dropped_count: int


@dataclass(frozen=True, slots=True)
class GateDecisionReviewRecord:
    gate: str
    status: str
    issue_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GateDetailRecord:
    split: str
    sample_id: str
    final_status: str
    terminal_gate: str | None
    failed_gates: tuple[str, ...]
    decisions: tuple[GateDecisionReviewRecord, ...]


@dataclass(frozen=True, slots=True)
class GateSourceIssueSummaryRecord:
    split: str
    source_issue_count: int
    unmatched_count: int


@dataclass(frozen=True, slots=True)
class GateSourceIssueDetailRecord:
    split: str
    sample_id: str
    issue_code: str
    detail: str | None


@dataclass(frozen=True, slots=True)
class GateReportArtifactRow:
    label: str
    path: Path

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        object.__setattr__(self, "label", self.label.strip())


@dataclass(frozen=True, slots=True)
class GatePublishTargetRow:
    label: str
    kind: str
    source_path: Path
    target_path: Path

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        object.__setattr__(self, "label", self.label.strip())


@dataclass(frozen=True, slots=True)
class GatePublishedArtifactRow:
    label: str
    kind: str
    target_path: Path
    target_sha256: str | None
    source_file_sha256: str | None = None
    file_digest_match: bool | None = None
    source_member_count: int | None = None
    source_member_listing_digest: str | None = None
    source_member_tree_digest: str | None = None
    target_member_count: int | None = None
    target_member_listing_digest: str | None = None
    members_preview: tuple[str, ...] = ()
    coherent: bool = True

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        _validate_non_empty_text("kind", self.kind)
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "kind", self.kind.strip())


@dataclass(frozen=True, slots=True)
class GateManifestRow:
    label: str
    path: Path
    partition: str
    split: str

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        _validate_non_empty_text("partition", self.partition)
        _validate_non_empty_text("split", self.split)
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "partition", self.partition.strip())
        object.__setattr__(self, "split", self.split.strip())


def _validate_non_empty_text(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise GateWorkflowInputError(f"{field_name} must be a non-empty string")


def _ensure_non_negative(field_name: str, value: int) -> None:
    if value < 0:
        raise GateWorkflowInputError(f"{field_name} must be >= 0")
