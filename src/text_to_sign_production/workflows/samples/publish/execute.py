from __future__ import annotations

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.workflows.foundation.execution import WorkflowExecutor
from text_to_sign_production.workflows.samples.contracts import (
    SamplesPublishExecution,
    SamplesPublishPlan,
)


def execute_samples_publish(
    plan: SamplesPublishPlan,
    *,
    executor: WorkflowExecutor,
    progress_session: ProgressSession | None = None,
) -> SamplesPublishExecution:
    execution_result = executor.execute_many(
        plan.operations,
        progress_session=progress_session,
    )
    return SamplesPublishExecution(plan=plan, execution=execution_result)
