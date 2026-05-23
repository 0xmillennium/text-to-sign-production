"""Runtime restore and readiness contracts for model workflow execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.artifacts.store import ArtifactTopology
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.candidates import ModelRunRequest
from text_to_sign_production.workflows.foundation.execution import (
    OperationBatchExecutionResult,
    ReadinessLevel,
    RuntimeCheckScope,
    RuntimeReadinessScope,
    WorkflowOperation,
)
from text_to_sign_production.workflows.foundation.provenance import FileProvenance
from text_to_sign_production.workflows.model.contracts.config import ModelWorkflowInputError


@dataclass(frozen=True, slots=True)
class ModelRuntimeSplitInputs:
    split: SampleSplit
    manifest_path: Path
    passed_samples_split_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", SampleSplit(self.split))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(
            self,
            "passed_samples_split_root",
            Path(self.passed_samples_split_root),
        )


@dataclass(frozen=True, slots=True)
class ModelWorkflowExecutionInputs:
    request: ModelRunRequest
    model_config_path: Path | None
    model_config_provenance: FileProvenance | None
    semantic_objective_config_path: Path | None
    semantic_objective_config_provenance: FileProvenance | None
    runtime_topology: ArtifactTopology
    model_run_root: Path
    report_root: Path
    split_inputs: tuple[ModelRuntimeSplitInputs, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, ModelRunRequest):
            raise ModelWorkflowInputError("request must be a ModelRunRequest")
        if self.model_config_path is not None:
            object.__setattr__(self, "model_config_path", Path(self.model_config_path))
        if self.semantic_objective_config_path is not None:
            object.__setattr__(
                self,
                "semantic_objective_config_path",
                Path(self.semantic_objective_config_path),
            )
        if not isinstance(self.runtime_topology, ArtifactTopology):
            raise ModelWorkflowInputError("runtime_topology must be an ArtifactTopology")
        object.__setattr__(self, "model_run_root", Path(self.model_run_root))
        object.__setattr__(self, "report_root", Path(self.report_root))
        split_inputs = tuple(self.split_inputs)
        if any(not isinstance(split_input, ModelRuntimeSplitInputs) for split_input in split_inputs):
            raise ModelWorkflowInputError(
                "split_inputs must contain ModelRuntimeSplitInputs values"
            )
        splits = tuple(split_input.split for split_input in split_inputs)
        if len(set(splits)) != len(splits):
            raise ModelWorkflowInputError("split_inputs must not contain duplicate splits")
        object.__setattr__(self, "split_inputs", split_inputs)


@dataclass(frozen=True, slots=True)
class ModelRuntimePlan:
    restore_operations: tuple[WorkflowOperation, ...]
    execution_inputs: ModelWorkflowExecutionInputs

    def __post_init__(self) -> None:
        object.__setattr__(self, "restore_operations", tuple(self.restore_operations))


@dataclass(frozen=True, slots=True)
class ModelRuntimeRestoreResult:
    plan: ModelRuntimePlan
    execution: OperationBatchExecutionResult


@dataclass(frozen=True, slots=True)
class ModelRuntimeAssetCheck:
    label: str
    path: Path
    exists: bool
    valid: bool = True
    message: str | None = None
    scope: RuntimeCheckScope = "asset"

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ModelWorkflowInputError("runtime check label must be non-empty")
        if self.scope not in {"asset", "domain"}:
            raise ModelWorkflowInputError("runtime check scope must be asset or domain")
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True, slots=True)
class ModelRuntimeVerification:
    checks: tuple[ModelRuntimeAssetCheck, ...]
    readiness_level: ReadinessLevel | str
    checked_semantics: tuple[str, ...]
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "readiness_level", ReadinessLevel(self.readiness_level))
        object.__setattr__(
            self,
            "checked_semantics",
            tuple(item.strip() for item in self.checked_semantics if item.strip()),
        )
        object.__setattr__(
            self,
            "limitations",
            tuple(item.strip() for item in self.limitations if item.strip()),
        )

    @property
    def readiness(self) -> RuntimeReadinessScope:
        return RuntimeReadinessScope(
            readiness_level=self.readiness_level,
            checked_semantics=self.checked_semantics,
            limitations=self.limitations,
        )

    @property
    def succeeded(self) -> bool:
        return all(check.exists and check.valid for check in self.checks)

    @property
    def missing_paths(self) -> tuple[Path, ...]:
        return tuple(check.path for check in self.checks if not check.exists)

    @property
    def failed_checks(self) -> tuple[ModelRuntimeAssetCheck, ...]:
        return tuple(check for check in self.checks if not check.exists or not check.valid)


__all__ = [
    "ModelRuntimeAssetCheck",
    "ModelRuntimePlan",
    "ModelRuntimeRestoreResult",
    "ModelRuntimeSplitInputs",
    "ModelRuntimeVerification",
    "ModelWorkflowExecutionInputs",
]
