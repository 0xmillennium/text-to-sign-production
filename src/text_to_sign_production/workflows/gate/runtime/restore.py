from __future__ import annotations

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.workflows.foundation.execution import WorkflowExecutor
from text_to_sign_production.workflows.gate.contracts import (
    GateRuntimePlan,
    GateRuntimeRestoreResult,
)


def restore_gate_runtime(
    plan: GateRuntimePlan,
    *,
    executor: WorkflowExecutor,
    progress_session: ProgressSession | None = None,
) -> GateRuntimeRestoreResult:
    execution_result = executor.execute_many(
        plan.restore_operations,
        progress_session=progress_session,
    )
    return GateRuntimeRestoreResult(plan=plan, execution=execution_result)
