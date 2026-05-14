from __future__ import annotations

from text_to_sign_production.core.progress import (
    ProgressSession,
    ProgressStageSpec,
    TqdmProgressSink,
)
from text_to_sign_production.workflows.debug.constants import DEBUG_WORKFLOW_NAME


def visible_debug_progress_session(
    progress_session: ProgressSession | None,
) -> ProgressSession:
    if progress_session is not None:
        return progress_session
    return ProgressSession(
        workflow_id=DEBUG_WORKFLOW_NAME,
        sink=TqdmProgressSink(),
    )


def debug_progress_stage(
    *,
    stage_id: str,
    label: str,
    unit: str,
    owner_module: str,
    split_behavior: str,
    operation_kind: str,
    total_semantics: str,
    allowed_counters: tuple[str, ...] = (),
) -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=DEBUG_WORKFLOW_NAME,
        stage_id=stage_id,
        label=label,
        unit=unit,
        owner_module=owner_module,
        split_behavior=split_behavior,
        operation_kind=operation_kind,
        total_semantics=total_semantics,
        bar_eligible=True,
        allowed_counters=allowed_counters,
    )


__all__ = ["debug_progress_stage", "visible_debug_progress_session"]
