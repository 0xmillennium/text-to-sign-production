"""Progress helpers for the test_model workflow."""

from __future__ import annotations

from text_to_sign_production.core.progress import (
    ProgressSession,
    ProgressStageSpec,
    TqdmProgressSink,
)
from text_to_sign_production.workflows.test_model.constants import TEST_MODEL_WORKFLOW_NAME


def test_model_progress_stage(
    *,
    stage_id: str,
    label: str,
    unit: str,
    owner_module: str,
    operation_kind: str,
    total_semantics: str,
    split_behavior: str = "global",
    bar_eligible: bool = True,
    allowed_counters: tuple[str, ...] = (),
) -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TEST_MODEL_WORKFLOW_NAME,
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


def visible_test_model_progress_session(
    progress_session: ProgressSession | None,
) -> ProgressSession:
    if progress_session is not None:
        return progress_session
    return ProgressSession(
        workflow_id=TEST_MODEL_WORKFLOW_NAME,
        sink=TqdmProgressSink(),
    )


__all__ = ["test_model_progress_stage", "visible_test_model_progress_session"]
