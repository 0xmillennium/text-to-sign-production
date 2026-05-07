from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.workflows.foundation.execution import (
    OperationBatchExecutionResult,
    WorkflowOperation,
)
from text_to_sign_production.workflows.tiers.contracts.config import (
    TiersWorkflowInputError,
)


@dataclass(frozen=True, slots=True)
class TiersSplitRuntimeInputs:
    split: str
    passed_manifest_path: Path
    passed_samples_split_root: Path

    def __post_init__(self) -> None:
        _validate_non_empty_text("split", self.split)
        object.__setattr__(self, "split", self.split.strip())


@dataclass(frozen=True, slots=True)
class TiersWorkflowExecutionInputs:
    filters_config_path: Path
    tiers_config_path: Path
    passed_samples_root: Path
    split_inputs: tuple[TiersSplitRuntimeInputs, ...]

    def __post_init__(self) -> None:
        split_inputs = tuple(self.split_inputs)
        _ensure_unique_split_inputs(split_inputs)
        object.__setattr__(self, "split_inputs", split_inputs)


@dataclass(frozen=True, slots=True)
class TiersRuntimePlan:
    restore_operations: tuple[WorkflowOperation, ...]
    execution_inputs: TiersWorkflowExecutionInputs

    def __post_init__(self) -> None:
        object.__setattr__(self, "restore_operations", tuple(self.restore_operations))


@dataclass(frozen=True, slots=True)
class TiersRuntimeRestoreResult:
    plan: TiersRuntimePlan
    execution: OperationBatchExecutionResult


@dataclass(frozen=True, slots=True)
class TiersRuntimeAssetCheck:
    label: str
    path: Path
    exists: bool

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        object.__setattr__(self, "label", self.label.strip())


@dataclass(frozen=True, slots=True)
class TiersRuntimeVerification:
    checks: tuple[TiersRuntimeAssetCheck, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))

    @property
    def succeeded(self) -> bool:
        return all(check.exists for check in self.checks)

    def missing_paths(self) -> tuple[Path, ...]:
        return tuple(check.path for check in self.checks if not check.exists)


def _validate_non_empty_text(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TiersWorkflowInputError(f"{field_name} must be a non-empty string")


def _ensure_unique_split_inputs(split_inputs: tuple[TiersSplitRuntimeInputs, ...]) -> None:
    splits = tuple(split_input.split for split_input in split_inputs)
    if len(set(splits)) != len(splits):
        raise TiersWorkflowInputError("split_inputs must not contain duplicate splits")
