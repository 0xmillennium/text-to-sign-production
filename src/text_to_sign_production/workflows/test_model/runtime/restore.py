"""Execute test_model runtime restoration."""

from __future__ import annotations

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.workflows.foundation.execution import WorkflowExecutor
from text_to_sign_production.workflows.test_model.contracts import (
    TestModelRestoreOperationResult,
    TestModelRestorePlan,
    TestModelRestoreResult,
)


def restore_test_model_runtime(
    plan: TestModelRestorePlan,
    *,
    executor: WorkflowExecutor,
    progress_session: ProgressSession | None = None,
) -> TestModelRestoreResult:
    if plan.blocking_errors:
        return TestModelRestoreResult(plan=plan, operation_results=(), execution=None)
    executable = tuple(
        operation for operation in plan.operations if operation.required or operation.source.exists()
    )
    execution = executor.execute_many(
        tuple(operation.workflow_operation for operation in executable),
        progress_session=progress_session,
    )
    results_by_label = {result.label: result for result in execution.results}
    operation_results: list[TestModelRestoreOperationResult] = []
    for operation in plan.operations:
        if not operation.source.exists() and not operation.required:
            operation_results.append(
                TestModelRestoreOperationResult(
                    operation=operation,
                    status="skipped_optional",
                    message=f"source does not exist: {operation.source}",
                )
            )
            continue
        result = results_by_label.get(operation.workflow_operation.label)
        if result is None:
            operation_results.append(
                TestModelRestoreOperationResult(
                    operation=operation,
                    status="failed_required" if operation.required else "failed_optional",
                    message="operation was not executed by the batch executor",
                )
            )
            continue
        operation_results.append(
            TestModelRestoreOperationResult(
                operation=operation,
                status=(
                    "succeeded"
                    if result.succeeded
                    else "failed_required"
                    if operation.required
                    else "failed_optional"
                ),
                message=None
                if result.succeeded
                else result.failure.failure_message
                if result.failure
                else "restore operation failed",
            )
        )
    return TestModelRestoreResult(
        plan=plan,
        operation_results=tuple(operation_results),
        execution=execution,
    )


__all__ = ["restore_test_model_runtime"]
