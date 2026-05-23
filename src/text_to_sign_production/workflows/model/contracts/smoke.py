"""Typed smoke execution protocol contracts for the model workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from text_to_sign_production.modeling.candidates import ModelRunMode
from text_to_sign_production.modeling.data import ModelingManifestFamily
from text_to_sign_production.modeling.research import ModelKey, ObjectiveKey
from text_to_sign_production.workflows.model.contracts.config import ModelWorkflowInputError


@dataclass(frozen=True, slots=True)
class ModelSmokeProtocolStep:
    order: int
    name: str
    purpose: str
    starts_after: str
    stop_condition: str
    success_signal: str

    def __post_init__(self) -> None:
        if not isinstance(self.order, int) or self.order <= 0:
            raise ModelWorkflowInputError("smoke protocol step order must be positive")
        for field_name in (
            "name",
            "purpose",
            "starts_after",
            "stop_condition",
            "success_signal",
        ):
            _require_text(field_name, getattr(self, field_name))


@dataclass(frozen=True, slots=True)
class ModelSmokeHandoff:
    model_run_name: str
    model_key: ModelKey
    manifest_family: ModelingManifestFamily
    run_mode: ModelRunMode
    auxiliary_objectives: tuple[ObjectiveKey, ...]
    recommended_checkpoint_policy: str
    test_manifest_path: Path | None
    target_sentence_guidance: str
    paste_into_test_model: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text("model_run_name", self.model_run_name)
        object.__setattr__(self, "model_key", ModelKey(self.model_key))
        if not isinstance(self.manifest_family, ModelingManifestFamily):
            raise ModelWorkflowInputError(
                "manifest_family must be a ModelingManifestFamily"
            )
        object.__setattr__(self, "run_mode", ModelRunMode(self.run_mode))
        object.__setattr__(
            self,
            "auxiliary_objectives",
            tuple(ObjectiveKey(objective) for objective in self.auxiliary_objectives),
        )
        if self.recommended_checkpoint_policy != "best":
            raise ModelWorkflowInputError(
                "recommended checkpoint policy must be 'best'"
            )
        if self.test_manifest_path is not None:
            object.__setattr__(self, "test_manifest_path", Path(self.test_manifest_path))
        _require_text("target_sentence_guidance", self.target_sentence_guidance)
        paste = _string_mapping(self.paste_into_test_model, "paste_into_test_model")
        required = {"MODEL_RUN_NAME", "CHECKPOINT_POLICY", "TARGET_SENTENCE_NAME"}
        missing = sorted(required.difference(paste))
        if missing:
            raise ModelWorkflowInputError(
                f"paste_into_test_model is missing keys: {missing}"
            )
        object.__setattr__(self, "paste_into_test_model", MappingProxyType(paste))


@dataclass(frozen=True, slots=True)
class ModelSmokeExecutionProtocol:
    run_name: str
    smoke_mode: bool
    ready_for_smoke_execution: bool
    steps: tuple[ModelSmokeProtocolStep, ...]
    handoff: ModelSmokeHandoff
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text("run_name", self.run_name)
        if not isinstance(self.smoke_mode, bool):
            raise ModelWorkflowInputError("smoke_mode must be a boolean")
        if not isinstance(self.ready_for_smoke_execution, bool):
            raise ModelWorkflowInputError(
                "ready_for_smoke_execution must be a boolean"
            )
        steps = tuple(self.steps)
        if any(not isinstance(step, ModelSmokeProtocolStep) for step in steps):
            raise ModelWorkflowInputError(
                "steps must contain ModelSmokeProtocolStep values"
            )
        orders = tuple(step.order for step in steps)
        if not orders:
            raise ModelWorkflowInputError("smoke protocol requires at least one step")
        if orders != tuple(sorted(orders)) or len(set(orders)) != len(orders):
            raise ModelWorkflowInputError(
                "smoke protocol step orders must be unique and sorted"
            )
        if not isinstance(self.handoff, ModelSmokeHandoff):
            raise ModelWorkflowInputError("handoff must be a ModelSmokeHandoff")
        warnings = _text_tuple("warnings", self.warnings)
        limitations = _text_tuple("limitations", self.limitations)
        if not self.smoke_mode and not warnings:
            raise ModelWorkflowInputError(
                "non-smoke protocols must include an explanatory warning"
            )
        if not self.ready_for_smoke_execution and not warnings:
            raise ModelWorkflowInputError(
                "not-ready smoke protocols must include an explanatory warning"
            )
        if not limitations:
            raise ModelWorkflowInputError("smoke protocol limitations must be non-empty")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "limitations", limitations)


def _require_text(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ModelWorkflowInputError(f"{field_name} must be non-empty")


def _text_tuple(field_name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    result = tuple(values)
    for value in result:
        _require_text(field_name, value)
    return result


def _string_mapping(value: Mapping[str, str], field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ModelWorkflowInputError(f"{field_name} must be a mapping")
    result: dict[str, str] = {}
    for key, item in value.items():
        _require_text(field_name, key)
        _require_text(field_name, item)
        result[str(key)] = str(item)
    return result


__all__ = [
    "ModelSmokeExecutionProtocol",
    "ModelSmokeHandoff",
    "ModelSmokeProtocolStep",
]
