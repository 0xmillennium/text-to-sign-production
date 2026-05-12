from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.workflows.foundation.execution import (
    OperationBatchExecutionResult,
    ReadinessLevel,
    RuntimeCheckScope,
    RuntimeReadinessScope,
    WorkflowOperation,
)
from text_to_sign_production.workflows.foundation.provenance import FileProvenance
from text_to_sign_production.workflows.tier.contracts.config import (
    TierWorkflowInputError,
)


@dataclass(frozen=True, slots=True)
class TierSplitRuntimeInputs:
    split: str
    passed_manifest_path: Path
    passed_samples_split_root: Path

    def __post_init__(self) -> None:
        _validate_non_empty_text("split", self.split)
        object.__setattr__(self, "split", self.split.strip())


@dataclass(frozen=True, slots=True)
class TierWorkflowExecutionInputs:
    filters_config_path: Path
    filters_config_provenance: FileProvenance
    tier_config_path: Path
    tier_config_provenance: FileProvenance
    passed_samples_root: Path
    split_inputs: tuple[TierSplitRuntimeInputs, ...]

    def __post_init__(self) -> None:
        split_inputs = tuple(self.split_inputs)
        _ensure_unique_split_inputs(split_inputs)
        object.__setattr__(self, "split_inputs", split_inputs)


@dataclass(frozen=True, slots=True)
class TierRuntimePlan:
    restore_operations: tuple[WorkflowOperation, ...]
    execution_inputs: TierWorkflowExecutionInputs

    def __post_init__(self) -> None:
        object.__setattr__(self, "restore_operations", tuple(self.restore_operations))


@dataclass(frozen=True, slots=True)
class TierRuntimeRestoreResult:
    plan: TierRuntimePlan
    execution: OperationBatchExecutionResult


@dataclass(frozen=True, slots=True)
class TierRuntimeAssetCheck:
    label: str
    path: Path
    exists: bool
    valid: bool = True
    message: str | None = None
    scope: RuntimeCheckScope = "asset"

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        if self.scope not in {"asset", "domain"}:
            raise TierWorkflowInputError("runtime check scope must be asset or domain")
        object.__setattr__(self, "label", self.label.strip())


@dataclass(frozen=True, slots=True)
class TierRuntimeVerification:
    checks: tuple[TierRuntimeAssetCheck, ...]
    readiness_level: ReadinessLevel | str
    checked_semantics: tuple[str, ...]
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "readiness_level", ReadinessLevel(self.readiness_level))
        object.__setattr__(
            self,
            "checked_semantics",
            tuple(
                semantic.strip()
                for semantic in self.checked_semantics
                if isinstance(semantic, str) and semantic.strip()
            ),
        )
        object.__setattr__(
            self,
            "limitations",
            tuple(
                limitation.strip()
                for limitation in self.limitations
                if isinstance(limitation, str) and limitation.strip()
            ),
        )

    @property
    def readiness(self) -> RuntimeReadinessScope:
        return RuntimeReadinessScope(
            readiness_level=ReadinessLevel(self.readiness_level),
            checked_semantics=self.checked_semantics,
            limitations=self.limitations,
        )

    @property
    def asset_checks(self) -> tuple[TierRuntimeAssetCheck, ...]:
        return tuple(check for check in self.checks if check.scope == "asset")

    @property
    def domain_checks(self) -> tuple[TierRuntimeAssetCheck, ...]:
        return tuple(check for check in self.checks if check.scope == "domain")

    @property
    def succeeded(self) -> bool:
        return all(check.exists and check.valid for check in self.checks)

    def missing_paths(self) -> tuple[Path, ...]:
        return tuple(check.path for check in self.checks if not check.exists)


def _validate_non_empty_text(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TierWorkflowInputError(f"{field_name} must be a non-empty string")


def _ensure_unique_split_inputs(split_inputs: tuple[TierSplitRuntimeInputs, ...]) -> None:
    splits = tuple(split_input.split for split_input in split_inputs)
    if len(set(splits)) != len(splits):
        raise TierWorkflowInputError("split_inputs must not contain duplicate splits")
