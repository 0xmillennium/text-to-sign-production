"""Typed preflight contracts for the single-sample test_model workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from text_to_sign_production.modeling.data import ModelingManifestFamily
from text_to_sign_production.modeling.research import ModelKey, ObjectiveKey
from text_to_sign_production.workflows.test_model.contracts.config import (
    TestModelWorkflowInputError,
)
from text_to_sign_production.workflows.test_model.contracts.request import CheckpointPolicy

_STATUSES = frozenset({"pass", "warn", "fail"})


@dataclass(frozen=True, slots=True)
class TestModelPreflightCheck:
    name: str
    scope: str
    status: str
    message: str
    path: Path | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text("name", self.name)
        _require_text("scope", self.scope)
        _require_text("message", self.message)
        if self.status not in _STATUSES:
            raise TestModelWorkflowInputError(
                f"test_model preflight status must be one of {sorted(_STATUSES)}"
            )
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "details", MappingProxyType(_json_safe_mapping(self.details)))


@dataclass(frozen=True, slots=True)
class TestModelExpectedInput:
    label: str
    source_path: Path
    target_path: Path | None
    artifact_family: str
    required_for_restore: bool
    provider_key: str | None = None

    def __post_init__(self) -> None:
        _require_text("label", self.label)
        _require_text("artifact_family", self.artifact_family)
        object.__setattr__(self, "source_path", Path(self.source_path))
        if self.target_path is not None:
            object.__setattr__(self, "target_path", Path(self.target_path))
        if not isinstance(self.required_for_restore, bool):
            raise TestModelWorkflowInputError("required_for_restore must be a boolean")
        if self.provider_key is not None:
            _require_text("provider_key", self.provider_key)


@dataclass(frozen=True, slots=True)
class TestModelExpectedOutput:
    label: str
    path: Path
    artifact_family: str
    required_for_success: bool

    def __post_init__(self) -> None:
        _require_text("label", self.label)
        _require_text("artifact_family", self.artifact_family)
        object.__setattr__(self, "path", Path(self.path))
        if not isinstance(self.required_for_success, bool):
            raise TestModelWorkflowInputError("required_for_success must be a boolean")


@dataclass(frozen=True, slots=True)
class TestModelPreflightResult:
    model_run_name: str
    checkpoint_policy: CheckpointPolicy
    target_sentence_name: str
    ready: bool
    resolved_model_key: ModelKey | None
    resolved_manifest_family: ModelingManifestFamily | None
    resolved_sample_id: str | None
    auxiliary_objectives: tuple[ObjectiveKey, ...]
    semantic_objective_attached: bool
    semantic_ready_for_comparison: bool | None
    semantic_required_baseline_missing: bool | None
    semantic_ablation_status: str
    checks: tuple[TestModelPreflightCheck, ...]
    expected_inputs: tuple[TestModelExpectedInput, ...]
    expected_outputs: tuple[TestModelExpectedOutput, ...]
    blocking_issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text("model_run_name", self.model_run_name)
        _require_text("target_sentence_name", self.target_sentence_name)
        object.__setattr__(self, "checkpoint_policy", CheckpointPolicy(self.checkpoint_policy))
        if self.resolved_model_key is not None:
            object.__setattr__(self, "resolved_model_key", ModelKey(self.resolved_model_key))
        if (
            self.resolved_manifest_family is not None
            and not isinstance(self.resolved_manifest_family, ModelingManifestFamily)
        ):
            raise TestModelWorkflowInputError(
                "resolved_manifest_family must be a ModelingManifestFamily or None"
            )
        objectives = tuple(ObjectiveKey(value) for value in self.auxiliary_objectives)
        object.__setattr__(self, "auxiliary_objectives", objectives)
        if not isinstance(self.semantic_objective_attached, bool):
            raise TestModelWorkflowInputError(
                "semantic_objective_attached must be a boolean"
            )
        if self.semantic_objective_attached != (
            ObjectiveKey.SEMANTIC_CONSISTENCY in objectives
        ):
            raise TestModelWorkflowInputError(
                "semantic objective attachment must match auxiliary_objectives"
            )
        for field_name in (
            "semantic_ready_for_comparison",
            "semantic_required_baseline_missing",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise TestModelWorkflowInputError(f"{field_name} must be a boolean or None")
        _require_text("semantic_ablation_status", self.semantic_ablation_status)
        if not isinstance(self.ready, bool):
            raise TestModelWorkflowInputError("ready must be a boolean")
        checks = tuple(self.checks)
        if any(not isinstance(check, TestModelPreflightCheck) for check in checks):
            raise TestModelWorkflowInputError(
                "checks must contain TestModelPreflightCheck values"
            )
        inputs = tuple(self.expected_inputs)
        if any(not isinstance(item, TestModelExpectedInput) for item in inputs):
            raise TestModelWorkflowInputError(
                "expected_inputs must contain TestModelExpectedInput values"
            )
        outputs = tuple(self.expected_outputs)
        if any(not isinstance(output, TestModelExpectedOutput) for output in outputs):
            raise TestModelWorkflowInputError(
                "expected_outputs must contain TestModelExpectedOutput values"
            )
        blocking = tuple(check.message for check in checks if check.status == "fail")
        warnings = tuple(check.message for check in checks if check.status == "warn")
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "expected_inputs", inputs)
        object.__setattr__(self, "expected_outputs", outputs)
        object.__setattr__(self, "blocking_issues", blocking)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "ready", not blocking)


def _require_text(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TestModelWorkflowInputError(f"{field_name} must be non-empty")


def _json_safe_mapping(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TestModelWorkflowInputError("details must be a mapping")
    return {str(key): _json_safe_value(item) for key, item in value.items()}


def _json_safe_value(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, (tuple, list)):
        return tuple(_json_safe_value(item) for item in value)
    return str(value)


__all__ = [
    "TestModelExpectedInput",
    "TestModelExpectedOutput",
    "TestModelPreflightCheck",
    "TestModelPreflightResult",
]
