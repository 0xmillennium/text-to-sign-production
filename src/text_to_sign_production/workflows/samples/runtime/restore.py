from __future__ import annotations

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.workflows.foundation.execution import WorkflowExecutor
from text_to_sign_production.workflows.samples.contracts import (
    SamplesRuntimePlan,
    SamplesRuntimeRestoreResult,
)


def restore_samples_runtime(
    plan: SamplesRuntimePlan,
    *,
    executor: WorkflowExecutor,
    progress_session: ProgressSession | None = None,
) -> SamplesRuntimeRestoreResult:
    execution_result = executor.execute_many(
        plan.restore_operations,
        progress_session=progress_session,
    )
    return SamplesRuntimeRestoreResult(plan=plan, execution=execution_result)
