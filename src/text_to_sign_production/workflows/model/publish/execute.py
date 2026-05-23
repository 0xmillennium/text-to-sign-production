"""Execute model workflow publication operations."""

from __future__ import annotations

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.workflows.foundation.execution import WorkflowExecutor
from text_to_sign_production.workflows.model.contracts import (
    ModelPublishExecution,
    ModelPublishPlan,
)


def execute_model_publish(
    plan: ModelPublishPlan,
    *,
    executor: WorkflowExecutor,
    progress_session: ProgressSession | None = None,
) -> ModelPublishExecution:
    """Publish planned files using shared execution operations."""

    return ModelPublishExecution(
        plan=plan,
        execution=executor.execute_many(
            plan.operations,
            progress_session=progress_session,
        ),
    )


__all__ = ["execute_model_publish"]
