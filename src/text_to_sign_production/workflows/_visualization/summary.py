"""Notebook-facing Visualization workflow summary projections."""

from __future__ import annotations

from text_to_sign_production.workflows._visualization.publish import (
    _verify_publish_plan_outputs,
)
from text_to_sign_production.workflows._visualization.types import (
    VisualizationPublishPlan,
    VisualizationPublishSummary,
    VisualizationWorkflowOutputSummary,
    VisualizationWorkflowResult,
)
from text_to_sign_production.workflows._visualization.verify import _verify_visualization_result


def summarize_visualization_workflow_outputs(
    result: VisualizationWorkflowResult,
) -> VisualizationWorkflowOutputSummary:
    """Return a notebook-safe summary of verified Visualization workflow outputs."""

    return _verify_visualization_result(result)


def summarize_visualization_publish_plan(
    plan: VisualizationPublishPlan,
) -> VisualizationPublishSummary:
    """Verify and summarize a notebook-executed Visualization publish plan."""

    return _verify_publish_plan_outputs(plan)


__all__ = [
    "summarize_visualization_publish_plan",
    "summarize_visualization_workflow_outputs",
]
