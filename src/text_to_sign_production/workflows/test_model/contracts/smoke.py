"""Typed smoke execution protocol contracts for test_model."""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sign_production.modeling.research import ObjectiveKey
from text_to_sign_production.workflows.test_model.contracts.config import (
    TestModelWorkflowInputError,
)
from text_to_sign_production.workflows.test_model.contracts.request import CheckpointPolicy


@dataclass(frozen=True, slots=True)
class TestModelSmokeProtocolStep:
    order: int
    name: str
    purpose: str
    starts_after: str
    stop_condition: str
    success_signal: str

    def __post_init__(self) -> None:
        if not isinstance(self.order, int) or self.order <= 0:
            raise TestModelWorkflowInputError("smoke protocol step order must be positive")
        for field_name in (
            "name",
            "purpose",
            "starts_after",
            "stop_condition",
            "success_signal",
        ):
            _require_text(field_name, getattr(self, field_name))


@dataclass(frozen=True, slots=True)
class TestModelSmokeExecutionProtocol:
    model_run_name: str
    checkpoint_policy: CheckpointPolicy
    target_sentence_name: str
    auxiliary_objectives: tuple[ObjectiveKey, ...]
    semantic_objective_attached: bool
    semantic_ready_for_comparison: bool | None
    semantic_required_baseline_missing: bool | None
    semantic_ablation_status: str
    ready_for_smoke_execution: bool
    steps: tuple[TestModelSmokeProtocolStep, ...]
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text("model_run_name", self.model_run_name)
        _require_text("target_sentence_name", self.target_sentence_name)
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
        object.__setattr__(
            self,
            "checkpoint_policy",
            CheckpointPolicy(self.checkpoint_policy),
        )
        if not isinstance(self.ready_for_smoke_execution, bool):
            raise TestModelWorkflowInputError(
                "ready_for_smoke_execution must be a boolean"
            )
        steps = tuple(self.steps)
        if any(not isinstance(step, TestModelSmokeProtocolStep) for step in steps):
            raise TestModelWorkflowInputError(
                "steps must contain TestModelSmokeProtocolStep values"
            )
        orders = tuple(step.order for step in steps)
        if not orders:
            raise TestModelWorkflowInputError(
                "test_model smoke protocol requires at least one step"
            )
        if orders != tuple(sorted(orders)) or len(set(orders)) != len(orders):
            raise TestModelWorkflowInputError(
                "test_model smoke protocol step orders must be unique and sorted"
            )
        warnings = _text_tuple("warnings", self.warnings)
        limitations = _text_tuple("limitations", self.limitations)
        if not self.ready_for_smoke_execution and not warnings:
            raise TestModelWorkflowInputError(
                "not-ready test_model smoke protocols must include an explanatory warning"
            )
        if not limitations:
            raise TestModelWorkflowInputError(
                "test_model smoke protocol limitations must be non-empty"
            )
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "limitations", limitations)


def _require_text(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TestModelWorkflowInputError(f"{field_name} must be non-empty")


def _text_tuple(field_name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    result = tuple(values)
    for value in result:
        _require_text(field_name, value)
    return result


__all__ = [
    "TestModelSmokeExecutionProtocol",
    "TestModelSmokeProtocolStep",
]
