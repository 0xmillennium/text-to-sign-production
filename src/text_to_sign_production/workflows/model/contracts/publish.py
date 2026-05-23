"""Publish contracts for model workflow artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.workflows.foundation.execution import (
    OperationBatchExecutionResult,
    WorkflowOperation,
)
from text_to_sign_production.workflows.model.contracts.config import ModelWorkflowInputError
from text_to_sign_production.workflows.model.contracts.results import (
    ModelReportArtifacts,
    ModelObjectiveArtifactResult,
    ModelRunMetadataArtifacts,
    ModelValidationArtifactResult,
    ModelStageArtifactReceiptResult,
    ModelStageExecutionWorkflowResult,
)


@dataclass(frozen=True, slots=True)
class ModelPublishTarget:
    label: str
    kind: str
    source_path: Path
    target_path: Path
    source_sha256: str
    source_execution_id: str | None = None

    def __post_init__(self) -> None:
        _require_text("label", self.label)
        _require_text("kind", self.kind)
        _require_text("source_sha256", self.source_sha256)
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "target_path", Path(self.target_path))
        if self.source_path == self.target_path:
            raise ModelWorkflowInputError("publish source and target paths must differ")


@dataclass(frozen=True, slots=True)
class ModelPublishSkippedSource:
    label: str
    path: Path
    reason: str

    def __post_init__(self) -> None:
        _require_text("label", self.label)
        _require_text("reason", self.reason)
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True, slots=True)
class ModelPublishSourceBundle:
    metadata_artifacts: ModelRunMetadataArtifacts | None = None
    validation_artifacts: ModelValidationArtifactResult | None = None
    report_artifacts: ModelReportArtifacts | None = None
    objective_artifacts: tuple[ModelObjectiveArtifactResult, ...] = ()
    stage_artifact_receipts: ModelStageArtifactReceiptResult | None = None
    stage_execution: ModelStageExecutionWorkflowResult | None = None


@dataclass(frozen=True, slots=True)
class ModelPublishPlan:
    targets: tuple[ModelPublishTarget, ...]
    skipped_sources: tuple[ModelPublishSkippedSource, ...]
    operations: tuple[WorkflowOperation, ...]

    def __post_init__(self) -> None:
        targets = tuple(self.targets)
        paths = tuple(target.target_path for target in targets)
        if len(set(paths)) != len(paths):
            raise ModelWorkflowInputError("publish targets must not contain duplicate paths")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "skipped_sources", tuple(self.skipped_sources))
        object.__setattr__(self, "operations", tuple(self.operations))


@dataclass(frozen=True, slots=True)
class ModelPublishExecution:
    plan: ModelPublishPlan
    execution: OperationBatchExecutionResult


@dataclass(frozen=True, slots=True)
class ModelPublishCheck:
    target: ModelPublishTarget
    exists: bool
    sha256_matches: bool
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ModelPublishVerification:
    checks: tuple[ModelPublishCheck, ...]

    @property
    def succeeded(self) -> bool:
        return all(check.exists and check.sha256_matches for check in self.checks)

    @property
    def missing_targets(self) -> tuple[Path, ...]:
        return tuple(check.target.target_path for check in self.checks if not check.exists)


@dataclass(frozen=True, slots=True)
class ModelPublishResult:
    plan: ModelPublishPlan
    execution: ModelPublishExecution
    verification: ModelPublishVerification


def _require_text(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ModelWorkflowInputError(f"{field_name} must be non-empty")


__all__ = [
    "ModelPublishCheck",
    "ModelPublishExecution",
    "ModelPublishPlan",
    "ModelPublishResult",
    "ModelPublishSkippedSource",
    "ModelPublishSourceBundle",
    "ModelPublishTarget",
    "ModelPublishVerification",
]
