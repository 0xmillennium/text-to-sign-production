"""Typed preflight contracts for the model workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from text_to_sign_production.modeling.candidates import ModelRunMode
from text_to_sign_production.modeling.data import ModelingManifestFamily
from text_to_sign_production.modeling.research import ModelKey, ObjectiveKey
from text_to_sign_production.workflows.model.contracts.config import ModelWorkflowInputError

_STATUSES = frozenset({"pass", "warn", "fail"})


@dataclass(frozen=True, slots=True)
class ModelPreflightCheck:
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
            raise ModelWorkflowInputError(
                f"model preflight status must be one of {sorted(_STATUSES)}"
            )
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "details", MappingProxyType(_json_safe_mapping(self.details)))


@dataclass(frozen=True, slots=True)
class ModelExpectedInput:
    label: str
    source_path: Path
    target_path: Path
    artifact_family: str
    required_for_restore: bool

    def __post_init__(self) -> None:
        _require_text("label", self.label)
        _require_text("artifact_family", self.artifact_family)
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "target_path", Path(self.target_path))
        if not isinstance(self.required_for_restore, bool):
            raise ModelWorkflowInputError("required_for_restore must be a boolean")


@dataclass(frozen=True, slots=True)
class ModelExpectedOutput:
    label: str
    path: Path
    artifact_family: str
    required_for_success: bool
    materialized_before_execution: bool

    def __post_init__(self) -> None:
        _require_text("label", self.label)
        _require_text("artifact_family", self.artifact_family)
        object.__setattr__(self, "path", Path(self.path))
        for name in ("required_for_success", "materialized_before_execution"):
            if not isinstance(getattr(self, name), bool):
                raise ModelWorkflowInputError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class ModelPreflightResult:
    model_key: ModelKey
    run_name: str
    manifest_family: ModelingManifestFamily
    run_mode: ModelRunMode
    auxiliary_objectives: tuple[ObjectiveKey, ...]
    ready: bool
    checks: tuple[ModelPreflightCheck, ...]
    expected_inputs: tuple[ModelExpectedInput, ...]
    expected_outputs: tuple[ModelExpectedOutput, ...]
    blocking_issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_key", ModelKey(self.model_key))
        _require_text("run_name", self.run_name)
        if not isinstance(self.manifest_family, ModelingManifestFamily):
            raise ModelWorkflowInputError("manifest_family must be a ModelingManifestFamily")
        object.__setattr__(self, "run_mode", ModelRunMode(self.run_mode))
        objectives = tuple(ObjectiveKey(value) for value in self.auxiliary_objectives)
        object.__setattr__(self, "auxiliary_objectives", objectives)
        if not isinstance(self.ready, bool):
            raise ModelWorkflowInputError("ready must be a boolean")
        checks = tuple(self.checks)
        if any(not isinstance(check, ModelPreflightCheck) for check in checks):
            raise ModelWorkflowInputError("checks must contain ModelPreflightCheck values")
        expected_inputs = tuple(self.expected_inputs)
        if any(not isinstance(item, ModelExpectedInput) for item in expected_inputs):
            raise ModelWorkflowInputError(
                "expected_inputs must contain ModelExpectedInput values"
            )
        expected_outputs = tuple(self.expected_outputs)
        if any(not isinstance(output, ModelExpectedOutput) for output in expected_outputs):
            raise ModelWorkflowInputError(
                "expected_outputs must contain ModelExpectedOutput values"
            )
        blocking = tuple(check.message for check in checks if check.status == "fail")
        warnings = tuple(check.message for check in checks if check.status == "warn")
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "expected_inputs", expected_inputs)
        object.__setattr__(self, "expected_outputs", expected_outputs)
        object.__setattr__(self, "blocking_issues", blocking)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "ready", not blocking)


def _require_text(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ModelWorkflowInputError(f"{field_name} must be non-empty")


def _json_safe_mapping(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ModelWorkflowInputError("details must be a mapping")
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
    "ModelExpectedInput",
    "ModelExpectedOutput",
    "ModelPreflightCheck",
    "ModelPreflightResult",
]
