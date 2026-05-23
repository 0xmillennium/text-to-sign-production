"""Resolve research registry declarations for a model workflow request."""

from __future__ import annotations

from text_to_sign_production.modeling.candidates import validate_objective_attachments
from text_to_sign_production.modeling.registry import require_model_spec
from text_to_sign_production.modeling.research import validate_trace_paths
from text_to_sign_production.workflows.model.contracts import (
    ModelResearchResolution,
    ModelWorkflowConfig,
    ModelWorkflowExecutionInputs,
    ModelWorkflowInvariantError,
)
from text_to_sign_production.workflows.model.request import (
    build_model_run_request_from_execution_inputs,
)


def resolve_model_research(
    config: ModelWorkflowConfig,
    execution_inputs: ModelWorkflowExecutionInputs,
) -> ModelResearchResolution:
    """Resolve registered model/objective specifications and trace issues."""

    expected = build_model_run_request_from_execution_inputs(
        config,
        execution_inputs,
    )
    if execution_inputs.request != expected:
        raise ModelWorkflowInvariantError("research request does not match workflow config")
    model_spec = require_model_spec(execution_inputs.request.model_key)
    objectives = validate_objective_attachments(
        model_spec,
        execution_inputs.request.auxiliary_objectives,
    )
    trace_issues = list(
        validate_trace_paths(model_spec.trace, project_root=config.project_root)
    )
    for objective in objectives:
        trace_issues.extend(
            validate_trace_paths(objective.trace, project_root=config.project_root)
        )
    return ModelResearchResolution(
        request=execution_inputs.request,
        model_spec=model_spec,
        objective_specs=objectives,
        trace_issues=tuple(trace_issues),
    )


__all__ = ["resolve_model_research"]
