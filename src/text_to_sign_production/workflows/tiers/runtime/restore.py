from __future__ import annotations

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.workflows.foundation.execution import WorkflowExecutor
from text_to_sign_production.workflows.tiers.contracts import (
    TiersRuntimePlan,
    TiersRuntimeRestoreResult,
)


def restore_tiers_runtime(
    plan: TiersRuntimePlan,
    *,
    executor: WorkflowExecutor,
    progress_session: ProgressSession | None = None,
) -> TiersRuntimeRestoreResult:
    execution_result = executor.execute_many(
        plan.restore_operations,
        progress_session=progress_session,
    )
    return TiersRuntimeRestoreResult(plan=plan, execution=execution_result)
