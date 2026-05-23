"""Centralized construction of model workflow run requests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from text_to_sign_production.modeling.candidates import ModelRunRequest
from text_to_sign_production.modeling.research import ObjectiveKey
from text_to_sign_production.workflows.model.contracts import (
    ModelWorkflowConfig,
    ModelWorkflowInvariantError,
)

if TYPE_CHECKING:
    from text_to_sign_production.workflows.model.contracts import (
        ModelWorkflowExecutionInputs,
    )
    from text_to_sign_production.workflows.model.layout import ModelLayout


def build_model_run_request_from_layout(
    config: ModelWorkflowConfig,
    layout: ModelLayout,
) -> ModelRunRequest:
    """Build the workflow request from a validated runtime layout."""

    if layout.config != config:
        raise ModelWorkflowInvariantError("layout.config must match the provided config")
    return build_model_run_request(
        config,
        config_path=layout.runtime.model_config_path,
        semantic_objective_config_path=layout.runtime.semantic_objective_config_path,
    )


def build_model_run_request_from_execution_inputs(
    config: ModelWorkflowConfig,
    execution_inputs: ModelWorkflowExecutionInputs,
) -> ModelRunRequest:
    """Build the workflow request from restored execution inputs."""

    return build_model_run_request(
        config,
        config_path=execution_inputs.model_config_path,
        semantic_objective_config_path=execution_inputs.semantic_objective_config_path,
    )


def build_model_run_request(
    config: ModelWorkflowConfig,
    *,
    config_path: Path | None,
    semantic_objective_config_path: Path | None,
) -> ModelRunRequest:
    """Build the provider-facing request with objective paths in one place."""

    objective_config_paths = _objective_config_paths_for_request(
        config,
        semantic_objective_config_path=semantic_objective_config_path,
    )
    return config.to_model_run_request(
        config_path=config_path,
        objective_config_paths=objective_config_paths,
    )


def _objective_config_paths_for_request(
    config: ModelWorkflowConfig,
    *,
    semantic_objective_config_path: Path | None,
) -> dict[ObjectiveKey, Path]:
    semantic_requested = (
        ObjectiveKey.SEMANTIC_CONSISTENCY in config.auxiliary_objectives
    )
    if semantic_requested:
        if semantic_objective_config_path is None:
            raise ModelWorkflowInvariantError(
                "semantic_consistency request requires a semantic objective config path"
            )
        return {
            ObjectiveKey.SEMANTIC_CONSISTENCY: Path(
                semantic_objective_config_path
            )
        }
    if semantic_objective_config_path is not None:
        raise ModelWorkflowInvariantError(
            "semantic objective config path is present without semantic_consistency"
        )
    return {}


__all__ = [
    "build_model_run_request",
    "build_model_run_request_from_execution_inputs",
    "build_model_run_request_from_layout",
]
