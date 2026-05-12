from __future__ import annotations

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.workflows.foundation.execution import WorkflowExecutor
from text_to_sign_production.workflows.tier.contracts import (
    TierRuntimePlan,
    TierRuntimeRestoreResult,
)


def restore_tier_runtime(
    plan: TierRuntimePlan,
    *,
    executor: WorkflowExecutor,
    progress_session: ProgressSession | None = None,
) -> TierRuntimeRestoreResult:
    execution_result = executor.execute_many(
        plan.restore_operations,
        progress_session=progress_session,
    )
    return TierRuntimeRestoreResult(plan=plan, execution=execution_result)
