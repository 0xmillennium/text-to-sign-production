from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from text_to_sign_production.workflows.foundation.execution import (
    OperationBatchExecutionResult,
    WorkflowOperation,
)
from text_to_sign_production.workflows.foundation.provenance import (
    SourceMemberTreeReceipt,
)
from text_to_sign_production.workflows.gate.contracts.config import (
    GateWorkflowInputError,
)

GatePublishTargetKind: TypeAlias = Literal["report_file", "manifest_file", "archive_file"]


@dataclass(frozen=True, slots=True)
class GatePublishTarget:
    label: str
    kind: GatePublishTargetKind
    source_path: Path
    target_path: Path
    source_sha256: str | None = None
    source_execution_id: str | None = None
    source_member_tree: SourceMemberTreeReceipt | None = None
    expected_members: tuple[str, ...] = ()
    expected_member_count: int | None = None
    expected_member_listing_digest: str | None = None
    members_preview: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "expected_members", tuple(self.expected_members))
        if self.kind == "archive_file":
            if self.source_sha256 is not None:
                raise GateWorkflowInputError(
                    "archive_file targets must not use source_sha256 for source-tree truth"
                )
            if self.source_member_tree is None:
                raise GateWorkflowInputError("archive_file targets require source_member_tree")
            if not self.expected_members:
                raise GateWorkflowInputError("archive_file targets require expected_members")
            if len(set(self.expected_members)) != len(self.expected_members):
                raise GateWorkflowInputError("archive_file expected_members must be unique")
            expected_member_count = (
                self.source_member_tree.member_count
                if self.expected_member_count is None
                else self.expected_member_count
            )
            if expected_member_count != len(self.expected_members):
                raise GateWorkflowInputError(
                    "archive_file expected_member_count must equal expected_members length"
                )
            object.__setattr__(self, "expected_member_count", expected_member_count)
            expected_digest = (
                self.source_member_tree.member_listing_digest
                if self.expected_member_listing_digest is None
                else self.expected_member_listing_digest
            )
            if expected_digest != self.source_member_tree.member_listing_digest:
                raise GateWorkflowInputError(
                    "archive_file expected_member_listing_digest must match source member tree"
                )
            object.__setattr__(self, "expected_member_listing_digest", expected_digest)
            object.__setattr__(self, "members_preview", self.source_member_tree.members_preview)
        elif (
            self.source_member_tree is not None
            or self.expected_members
            or self.expected_member_count is not None
            or self.expected_member_listing_digest is not None
        ):
            raise GateWorkflowInputError(
                "non-archive publish targets must not carry expected archive members"
            )


@dataclass(frozen=True, slots=True)
class GatePublishPlan:
    targets: tuple[GatePublishTarget, ...]
    operations: tuple[WorkflowOperation, ...]
    split_archive_plans: tuple[GateSplitArchivePublishPlan, ...] = ()

    def __post_init__(self) -> None:
        targets = tuple(self.targets)
        _ensure_unique_target_paths(targets)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "split_archive_plans", tuple(self.split_archive_plans))


@dataclass(frozen=True, slots=True)
class GateSplitArchivePublishPlan:
    split: str
    passed_archive_path: Path
    passed_archive_planned: bool
    passed_archive_member_count: int
    passed_archive_member_type: str
    dropped_archive_path: Path
    dropped_archive_planned: bool
    dropped_archive_member_count: int
    dropped_archive_member_type: str
    dropped_archive_not_planned_reason: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_text("split", self.split)
        _validate_non_empty_text("passed_archive_member_type", self.passed_archive_member_type)
        _validate_non_empty_text("dropped_archive_member_type", self.dropped_archive_member_type)
        object.__setattr__(self, "split", self.split.strip())
        object.__setattr__(
            self,
            "passed_archive_member_type",
            self.passed_archive_member_type.strip(),
        )
        object.__setattr__(
            self,
            "dropped_archive_member_type",
            self.dropped_archive_member_type.strip(),
        )
        for field_name, value in (
            ("passed_archive_member_count", self.passed_archive_member_count),
            ("dropped_archive_member_count", self.dropped_archive_member_count),
        ):
            if value < 0:
                raise GateWorkflowInputError(f"{field_name} must be >= 0")
        if self.dropped_archive_planned and self.dropped_archive_not_planned_reason is not None:
            raise GateWorkflowInputError(
                "dropped_archive_not_planned_reason must be None when dropped archive is planned"
            )
        if not self.dropped_archive_planned and not self.dropped_archive_not_planned_reason:
            raise GateWorkflowInputError(
                "dropped_archive_not_planned_reason is required when dropped archive is not planned"
            )


@dataclass(frozen=True, slots=True)
class GatePublishExecution:
    plan: GatePublishPlan
    execution: OperationBatchExecutionResult


@dataclass(frozen=True, slots=True)
class GatePublishCheck:
    label: str
    kind: GatePublishTargetKind
    source_path: Path
    source_exists: bool
    target_path: Path
    target_exists: bool
    source_sha256: str | None = None
    target_sha256: str | None = None
    digest_match: bool | None = None
    source_execution_id: str | None = None
    source_member_count: int | None = None
    source_member_listing_digest: str | None = None
    source_member_tree_digest: str | None = None
    observed_source_member_count: int | None = None
    observed_source_member_listing_digest: str | None = None
    observed_source_member_tree_digest: str | None = None
    expected_member_count: int | None = None
    target_member_count: int | None = None
    expected_member_listing_digest: str | None = None
    target_member_listing_digest: str | None = None
    source_members_preview: tuple[str, ...] = ()
    target_members_preview: tuple[str, ...] = ()
    expected_absent: bool = False
    coherent: bool = True
    message: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        object.__setattr__(self, "label", self.label.strip())


@dataclass(frozen=True, slots=True)
class GatePublishVerification:
    checks: tuple[GatePublishCheck, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))

    @property
    def succeeded(self) -> bool:
        return all(
            (
                (check.expected_absent and not check.target_exists and check.coherent)
                or (
                    not check.expected_absent
                    and check.source_exists
                    and check.target_exists
                    and check.coherent
                    and check.digest_match is not False
                )
            )
            for check in self.checks
        )

    def missing_targets(self) -> tuple[Path, ...]:
        return tuple(
            check.target_path
            for check in self.checks
            if not check.target_exists and not check.expected_absent
        )


@dataclass(frozen=True, slots=True)
class GatePublishResult:
    plan: GatePublishPlan
    execution: GatePublishExecution
    verification: GatePublishVerification


def _validate_non_empty_text(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise GateWorkflowInputError(f"{field_name} must be a non-empty string")


def _ensure_unique_target_paths(targets: tuple[GatePublishTarget, ...]) -> None:
    keys = [(target.kind, target.target_path) for target in targets]
    if len(set(keys)) != len(keys):
        raise GateWorkflowInputError(
            "publish targets must not contain duplicate kind/target_path pairs"
        )
