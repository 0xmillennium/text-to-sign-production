"""Execute model workflow runtime restoration operations."""

from __future__ import annotations

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.workflows.foundation.execution import WorkflowExecutor
from text_to_sign_production.workflows.model.contracts import (
    ModelRuntimePlan,
    ModelRuntimeRestoreResult,
)


def restore_model_runtime(
    plan: ModelRuntimePlan,
    *,
    executor: WorkflowExecutor,
    progress_session: ProgressSession | None = None,
) -> ModelRuntimeRestoreResult:
    """Execute planned input restoration using the shared executor."""

    execution = executor.execute_many(
        plan.restore_operations,
        progress_session=progress_session,
    )
    return ModelRuntimeRestoreResult(plan=plan, execution=execution)


__all__ = ["restore_model_runtime"]
