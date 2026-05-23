"""Execute test_model publish plans."""

from __future__ import annotations

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.workflows.foundation.execution import WorkflowExecutor
from text_to_sign_production.workflows.test_model.contracts import (
    TestModelIssue,
    TestModelPublishExecution,
    TestModelPublishPlan,
)


def execute_test_model_publish(
    plan: TestModelPublishPlan,
    *,
    executor: WorkflowExecutor,
    progress_session: ProgressSession | None = None,
) -> TestModelPublishExecution:
    if plan.blocking_errors:
        return TestModelPublishExecution(plan, None, 0, plan.blocking_errors, plan.warnings)
    execution = executor.execute_many(plan.operations, progress_session=progress_session)
    errors = tuple(
        TestModelIssue(
            "publish_operation_failed",
            result.failure.failure_message if result.failure else "publish failed",
            label=result.label,
        )
        for result in execution.results
        if not result.succeeded
    )
    return TestModelPublishExecution(
        plan=plan,
        execution=execution,
        published_file_count=sum(1 for result in execution.results if result.succeeded),
        errors=errors,
        warnings=plan.warnings,
    )


__all__ = ["execute_test_model_publish"]
