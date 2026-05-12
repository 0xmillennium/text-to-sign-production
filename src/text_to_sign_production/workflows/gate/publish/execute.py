from __future__ import annotations

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.workflows.foundation.execution import WorkflowExecutor
from text_to_sign_production.workflows.gate.contracts import (
    GatePublishExecution,
    GatePublishPlan,
)


def execute_gate_publish(
    plan: GatePublishPlan,
    *,
    executor: WorkflowExecutor,
    progress_session: ProgressSession | None = None,
) -> GatePublishExecution:
    execution_result = executor.execute_many(
        plan.operations,
        progress_session=progress_session,
    )
    return GatePublishExecution(plan=plan, execution=execution_result)
