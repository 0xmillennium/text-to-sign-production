"""Workflow-owned progress helpers for model orchestration."""

from __future__ import annotations

from text_to_sign_production.core.progress import (
    ProgressSession,
    ProgressStageSpec,
    TqdmProgressSink,
)
from text_to_sign_production.workflows.model.constants import MODEL_WORKFLOW_NAME


def visible_model_progress_session(
    progress_session: ProgressSession | None,
) -> ProgressSession:
    """Return an operator-visible model workflow progress session."""

    if progress_session is not None:
        return progress_session
    return ProgressSession(workflow_id=MODEL_WORKFLOW_NAME, sink=TqdmProgressSink())


def model_progress_stage(
    *,
    stage_id: str,
    label: str,
    unit: str,
    owner_module: str,
    split_behavior: str = "global",
    operation_kind: str,
    total_semantics: str,
    bar_eligible: bool = True,
    allowed_counters: tuple[str, ...] = (),
) -> ProgressStageSpec:
    """Build one workflow-owned progress stage specification."""

    return ProgressStageSpec(
        workflow_id=MODEL_WORKFLOW_NAME,
        stage_id=stage_id,
        label=label,
        unit=unit,
        owner_module=owner_module,
        split_behavior=split_behavior,
        operation_kind=operation_kind,
        total_semantics=total_semantics,
        bar_eligible=bar_eligible,
        allowed_counters=allowed_counters,
    )


__all__ = ["model_progress_stage", "visible_model_progress_session"]
