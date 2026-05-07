from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.workflows.foundation.execution.contracts import (
    WorkflowOperation,
    expected_operation_outputs,
    operation_kind,
)
from text_to_sign_production.workflows.foundation.execution.notebook_shell import (
    NotebookShellRunner,
)
from text_to_sign_production.workflows.foundation.execution.renderer import (
    render_operation_to_shell,
)
from text_to_sign_production.workflows.foundation.execution.results import (
    ExecutionFailureDetail,
    OperationBatchExecutionResult,
    OperationExecutionResult,
    RenderedShellCommand,
    ShellExecutionResult,
)


class WorkflowExecutor(Protocol):
    def execute(
        self,
        operation: WorkflowOperation,
        *,
        progress_session: ProgressSession | None = None,
    ) -> OperationExecutionResult: ...

    def execute_many(
        self,
        operations: tuple[WorkflowOperation, ...],
        *,
        progress_session: ProgressSession | None = None,
    ) -> OperationBatchExecutionResult: ...


@dataclass(slots=True)
class NotebookShellExecutor:
    runner: NotebookShellRunner = field(default_factory=NotebookShellRunner)
    stop_on_failure: bool = False

    def execute(
        self,
        operation: WorkflowOperation,
        *,
        progress_session: ProgressSession | None = None,
    ) -> OperationExecutionResult:
        self._emit_shell_progress_status(operation, progress_session)
        rendered = render_operation_to_shell(operation)
        shell_result = self.runner.run(rendered)
        observed_outputs = self._collect_observed_outputs(rendered.expected_outputs)
        failure = self._build_failure_detail(rendered, shell_result)
        return OperationExecutionResult(
            label=rendered.label,
            operation_kind=rendered.operation_kind,
            succeeded=shell_result.returncode == 0,
            returncode=shell_result.returncode,
            failure=failure,
            execution_mode=shell_result.execution_mode,
            progress_stage_id=(
                rendered.progress.stage.stage_id if rendered.progress is not None else None
            ),
            expected_outputs=rendered.expected_outputs,
            observed_outputs=observed_outputs,
        )

    def execute_many(
        self,
        operations: tuple[WorkflowOperation, ...],
        *,
        progress_session: ProgressSession | None = None,
    ) -> OperationBatchExecutionResult:
        results: list[OperationExecutionResult] = []
        for operation in operations:
            result = self.execute(operation, progress_session=progress_session)
            results.append(result)
            if self.stop_on_failure and not result.succeeded:
                break
        return OperationBatchExecutionResult(results=tuple(results))

    def _emit_shell_progress_status(
        self,
        operation: WorkflowOperation,
        progress_session: ProgressSession | None,
    ) -> None:
        if progress_session is None or operation.progress is None:
            return
        if operation.progress.live_owner != "shell":
            return
        progress_session.status(
            "Running shell-owned workflow operation",
            stage_id=operation.progress.stage.stage_id,
            label=operation.label,
            expected_total=operation.progress.expected_total,
        )

    def _collect_observed_outputs(self, expected_outputs: tuple[Path, ...]) -> tuple[Path, ...]:
        return tuple(path for path in expected_outputs if path.exists())

    def _build_failure_detail(
        self,
        rendered: RenderedShellCommand,
        shell_result: ShellExecutionResult,
    ) -> ExecutionFailureDetail | None:
        if shell_result.returncode == 0:
            return None
        return ExecutionFailureDetail(
            label=rendered.label,
            operation_kind=rendered.operation_kind,
            failure_message=rendered.failure_message,
            returncode=shell_result.returncode,
            execution_mode=shell_result.execution_mode,
        )


@dataclass(slots=True)
class RecordingExecutor:
    recorded_operations: list[WorkflowOperation] = field(default_factory=list)
    forced_results_by_label: dict[str, OperationExecutionResult] = field(default_factory=dict)

    def execute(
        self,
        operation: WorkflowOperation,
        *,
        progress_session: ProgressSession | None = None,
    ) -> OperationExecutionResult:
        del progress_session
        self.recorded_operations.append(operation)
        forced_result = self.forced_results_by_label.get(operation.label)
        if forced_result is not None:
            return forced_result
        return _deterministic_operation_result(
            operation,
            succeeded=True,
            returncode=0,
            execution_mode="recording_executor",
        )

    def execute_many(
        self,
        operations: tuple[WorkflowOperation, ...],
        *,
        progress_session: ProgressSession | None = None,
    ) -> OperationBatchExecutionResult:
        return OperationBatchExecutionResult(
            results=tuple(
                self.execute(operation, progress_session=progress_session)
                for operation in operations
            )
        )


@dataclass(slots=True)
class FailingExecutor:
    fail_labels: set[str] = field(default_factory=set)
    returncode: int = 1

    def execute(
        self,
        operation: WorkflowOperation,
        *,
        progress_session: ProgressSession | None = None,
    ) -> OperationExecutionResult:
        del progress_session
        if operation.label in self.fail_labels:
            return _deterministic_operation_result(
                operation,
                succeeded=False,
                returncode=self.returncode,
                execution_mode="failing_executor",
            )
        return _deterministic_operation_result(
            operation,
            succeeded=True,
            returncode=0,
            execution_mode="failing_executor",
        )

    def execute_many(
        self,
        operations: tuple[WorkflowOperation, ...],
        *,
        progress_session: ProgressSession | None = None,
    ) -> OperationBatchExecutionResult:
        return OperationBatchExecutionResult(
            results=tuple(
                self.execute(operation, progress_session=progress_session)
                for operation in operations
            )
        )


def _deterministic_operation_result(
    operation: WorkflowOperation,
    *,
    succeeded: bool,
    returncode: int,
    execution_mode: str,
) -> OperationExecutionResult:
    kind = operation_kind(operation)
    failure = None
    if not succeeded:
        failure = ExecutionFailureDetail(
            label=operation.label,
            operation_kind=kind,
            failure_message=operation.failure_message,
            returncode=returncode,
            execution_mode=execution_mode,
        )
    expected_outputs = expected_operation_outputs(operation)
    return OperationExecutionResult(
        label=operation.label,
        operation_kind=kind,
        succeeded=succeeded,
        returncode=returncode,
        failure=failure,
        execution_mode=execution_mode,
        progress_stage_id=(
            operation.progress.stage.stage_id if operation.progress is not None else None
        ),
        expected_outputs=expected_outputs,
        observed_outputs=expected_outputs if succeeded else (),
    )
