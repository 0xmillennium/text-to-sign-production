from __future__ import annotations

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.workflows.foundation.execution import WorkflowExecutor
from text_to_sign_production.workflows.tiers.contracts import (
    TiersPublishExecution,
    TiersPublishPlan,
)


def execute_tiers_publish(
    plan: TiersPublishPlan,
    *,
    executor: WorkflowExecutor,
    progress_session: ProgressSession | None = None,
) -> TiersPublishExecution:
    execution_result = executor.execute_many(
        plan.operations,
        progress_session=progress_session,
    )
    return TiersPublishExecution(plan=plan, execution=execution_result)
