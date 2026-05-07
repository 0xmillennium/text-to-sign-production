from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from text_to_sign_production.workflows.foundation.execution import (
    OperationBatchExecutionResult,
    WorkflowOperation,
)
from text_to_sign_production.workflows.samples.contracts.config import (
    SamplesWorkflowInputError,
)

SamplesPublishTargetKind: TypeAlias = Literal["report_file", "manifest_file", "archive_file"]


@dataclass(frozen=True, slots=True)
class SamplesPublishTarget:
    label: str
    kind: SamplesPublishTargetKind
    source_path: Path
    target_path: Path

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        object.__setattr__(self, "label", self.label.strip())


@dataclass(frozen=True, slots=True)
class SamplesPublishPlan:
    targets: tuple[SamplesPublishTarget, ...]
    operations: tuple[WorkflowOperation, ...]

    def __post_init__(self) -> None:
        targets = tuple(self.targets)
        _ensure_unique_target_paths(targets)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "operations", tuple(self.operations))


@dataclass(frozen=True, slots=True)
class SamplesPublishExecution:
    plan: SamplesPublishPlan
    execution: OperationBatchExecutionResult


@dataclass(frozen=True, slots=True)
class SamplesPublishCheck:
    label: str
    target_path: Path
    exists: bool

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        object.__setattr__(self, "label", self.label.strip())


@dataclass(frozen=True, slots=True)
class SamplesPublishVerification:
    checks: tuple[SamplesPublishCheck, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))

    @property
    def succeeded(self) -> bool:
        return all(check.exists for check in self.checks)

    def missing_targets(self) -> tuple[Path, ...]:
        return tuple(check.target_path for check in self.checks if not check.exists)


@dataclass(frozen=True, slots=True)
class SamplesPublishResult:
    plan: SamplesPublishPlan
    execution: SamplesPublishExecution
    verification: SamplesPublishVerification


def _validate_non_empty_text(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SamplesWorkflowInputError(f"{field_name} must be a non-empty string")


def _ensure_unique_target_paths(targets: tuple[SamplesPublishTarget, ...]) -> None:
    keys = [(target.kind, target.target_path) for target in targets]
    if len(set(keys)) != len(keys):
        raise SamplesWorkflowInputError(
            "publish targets must not contain duplicate kind/target_path pairs"
        )
