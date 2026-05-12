from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from text_to_sign_production.workflows.foundation.execution import (
    OperationBatchExecutionResult,
    WorkflowOperation,
)
from text_to_sign_production.workflows.tier.contracts.config import (
    TierWorkflowInputError,
)

TierPublishTargetKind: TypeAlias = Literal["tiered_manifest_file", "report_file"]


@dataclass(frozen=True, slots=True)
class TierPublishTarget:
    label: str
    kind: TierPublishTargetKind
    source_path: Path
    target_path: Path
    source_sha256: str | None = None
    source_execution_id: str | None = None
    tier: str | None = None
    membership: str | None = None
    split: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        object.__setattr__(self, "label", self.label.strip())
        if self.kind == "tiered_manifest_file":
            _validate_required_target_field("tier", self.tier)
            _validate_required_target_field("membership", self.membership)
            _validate_required_target_field("split", self.split)
            object.__setattr__(self, "tier", self.tier.strip() if self.tier else None)
            object.__setattr__(
                self,
                "membership",
                self.membership.strip() if self.membership else None,
            )
            object.__setattr__(self, "split", self.split.strip() if self.split else None)
        elif self.kind == "report_file":
            if self.tier is not None or self.membership is not None or self.split is not None:
                raise TierWorkflowInputError(
                    "report_file publish targets must not carry tier, membership, or split"
                )
        else:
            raise TierWorkflowInputError(f"Unsupported publish target kind: {self.kind}")


@dataclass(frozen=True, slots=True)
class TierPublishPlan:
    targets: tuple[TierPublishTarget, ...]
    operations: tuple[WorkflowOperation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "operations", tuple(self.operations))


@dataclass(frozen=True, slots=True)
class TierPublishExecution:
    plan: TierPublishPlan
    execution: OperationBatchExecutionResult


@dataclass(frozen=True, slots=True)
class TierPublishCheck:
    label: str
    kind: TierPublishTargetKind
    source_path: Path
    source_exists: bool
    target_path: Path
    target_exists: bool
    source_sha256: str | None = None
    target_sha256: str | None = None
    digest_match: bool | None = None
    source_execution_id: str | None = None
    coherent: bool = True
    message: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        object.__setattr__(self, "label", self.label.strip())


@dataclass(frozen=True, slots=True)
class TierPublishVerification:
    checks: tuple[TierPublishCheck, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))

    @property
    def succeeded(self) -> bool:
        return all(
            check.source_exists
            and check.target_exists
            and check.coherent
            and check.digest_match is not False
            for check in self.checks
        )

    def missing_targets(self) -> tuple[Path, ...]:
        return tuple(check.target_path for check in self.checks if not check.target_exists)


@dataclass(frozen=True, slots=True)
class TierPublishResult:
    plan: TierPublishPlan
    execution: TierPublishExecution
    verification: TierPublishVerification


def _validate_non_empty_text(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TierWorkflowInputError(f"{field_name} must be a non-empty string")


def _validate_required_target_field(field_name: str, value: str | None) -> None:
    if value is None or not isinstance(value, str) or not value.strip():
        raise TierWorkflowInputError(
            f"{field_name} is required for tiered_manifest_file publish targets"
        )
