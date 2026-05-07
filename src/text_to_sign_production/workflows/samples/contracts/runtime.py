from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.workflows.foundation.execution import (
    OperationBatchExecutionResult,
    WorkflowOperation,
)
from text_to_sign_production.workflows.samples.contracts.config import (
    SamplesWorkflowInputError,
)


@dataclass(frozen=True, slots=True)
class SamplesSplitRuntimeInputs:
    split: str
    translation_csv_path: Path
    keypoint_root: Path
    keypoint_json_root: Path
    keypoint_video_root: Path

    def __post_init__(self) -> None:
        _validate_non_empty_text("split", self.split)
        object.__setattr__(self, "split", self.split.strip())


@dataclass(frozen=True, slots=True)
class SamplesWorkflowExecutionInputs:
    gates_config_path: Path
    split_inputs: tuple[SamplesSplitRuntimeInputs, ...]

    def __post_init__(self) -> None:
        split_inputs = tuple(self.split_inputs)
        _ensure_unique_splits(split_inputs)
        object.__setattr__(self, "split_inputs", split_inputs)


@dataclass(frozen=True, slots=True)
class SamplesRuntimePlan:
    restore_operations: tuple[WorkflowOperation, ...]
    execution_inputs: SamplesWorkflowExecutionInputs

    def __post_init__(self) -> None:
        object.__setattr__(self, "restore_operations", tuple(self.restore_operations))


@dataclass(frozen=True, slots=True)
class SamplesRuntimeRestoreResult:
    plan: SamplesRuntimePlan
    execution: OperationBatchExecutionResult


@dataclass(frozen=True, slots=True)
class SamplesRuntimeAssetCheck:
    label: str
    path: Path
    exists: bool

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        object.__setattr__(self, "label", self.label.strip())


@dataclass(frozen=True, slots=True)
class SamplesRuntimeVerification:
    checks: tuple[SamplesRuntimeAssetCheck, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))

    @property
    def succeeded(self) -> bool:
        return all(check.exists for check in self.checks)

    def missing_paths(self) -> tuple[Path, ...]:
        return tuple(check.path for check in self.checks if not check.exists)


def _validate_non_empty_text(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SamplesWorkflowInputError(f"{field_name} must be a non-empty string")


def _ensure_unique_splits(split_inputs: tuple[SamplesSplitRuntimeInputs, ...]) -> None:
    splits = [split_input.split for split_input in split_inputs]
    if len(set(splits)) != len(splits):
        raise SamplesWorkflowInputError("split_inputs must not contain duplicate split names")
