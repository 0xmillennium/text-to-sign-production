from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.progress import ProgressStageSpec
from text_to_sign_production.workflows.debug.constants import (
    DEBUG_STAGE_OUTPUT_PUBLISH,
    DEBUG_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.debug.contracts import (
    DebugIssue,
    DebugPublishOperation,
    DebugPublishOperationResult,
    DebugPublishPlan,
    DebugPublishResult,
    DebugPublishVerification,
    DebugReportResult,
    DebugVisualizationResult,
)
from text_to_sign_production.workflows.debug.contracts.verdicts import (
    DebugPublishOperationStatus,
    DebugPublishVerificationStatus,
)
from text_to_sign_production.workflows.debug.layout import (
    DebugLayout,
    drive_debug_samples_root,
    runtime_debug_samples_root,
)
from text_to_sign_production.workflows.foundation.execution import (
    FileCopyOperation,
    OperationProgressSpec,
    WorkflowExecutor,
)


def build_debug_publish_plan(
    *,
    layout: DebugLayout,
    report_result: DebugReportResult,
    visual_result: DebugVisualizationResult,
) -> DebugPublishPlan:
    runtime_output_root = report_result.output_root
    errors: list[DebugIssue] = []
    warnings: list[DebugIssue] = []
    if visual_result.output_root != runtime_output_root:
        warnings.append(
            _issue(
                "visual_output_root_mismatch",
                "visualization output root differs from report output root; publishing report root",
                path=visual_result.output_root,
            )
        )
    if not report_result.succeeded:
        warnings.append(
            _issue(
                "report_write_warning",
                "report writing did not fully succeed; publishing available files only",
                path=runtime_output_root,
            )
        )
    drive_output_root = _drive_output_root_for_runtime_root(layout, runtime_output_root, errors)
    if not runtime_output_root.exists():
        errors.append(
            _issue(
                "runtime_output_root_missing",
                f"debug runtime output root is missing: {runtime_output_root}",
                path=runtime_output_root,
            )
        )
        files: tuple[Path, ...] = ()
    elif not runtime_output_root.is_dir():
        errors.append(
            _issue(
                "runtime_output_root_not_directory",
                f"debug runtime output root is not a directory: {runtime_output_root}",
                path=runtime_output_root,
            )
        )
        files = ()
    else:
        files = tuple(sorted(path for path in runtime_output_root.rglob("*") if path.is_file()))
        if not files:
            warnings.append(
                _issue(
                    "runtime_output_root_empty",
                    f"debug runtime output root contains no files: {runtime_output_root}",
                    path=runtime_output_root,
                )
            )
    operations = tuple(
        _publish_operation(
            source=source,
            target=drive_output_root / source.relative_to(runtime_output_root),
        )
        for source in files
        if source.resolve(strict=False)
        != (drive_output_root / source.relative_to(runtime_output_root)).resolve(strict=False)
    )
    if len(operations) != len(files):
        errors.append(
            _issue(
                "publish_source_equals_target",
                "one or more debug publish source files resolve to their target path",
                path=runtime_output_root,
            )
        )
    return DebugPublishPlan(
        runtime_output_root=runtime_output_root,
        drive_output_root=drive_output_root,
        operations=operations,
        expected_file_count=len(files),
        blocking_errors=tuple(sorted(errors, key=_issue_sort_key)),
        warnings=tuple(sorted(warnings, key=_issue_sort_key)),
    )


def publish_debug_outputs(
    plan: DebugPublishPlan,
    *,
    executor: WorkflowExecutor,
    progress_session=None,
) -> DebugPublishResult:
    if plan.blocking_errors:
        return DebugPublishResult(
            plan=plan,
            operation_results=(),
            execution=None,
            published_file_count=0,
            blocking_errors=plan.blocking_errors,
            warnings=plan.warnings,
        )
    warnings = list(plan.warnings)
    executable_operations: list[DebugPublishOperation] = []
    missing_source_issues: dict[str, DebugIssue] = {}
    for operation in plan.operations:
        if not operation.source.is_file():
            issue = _issue(
                "publish_source_missing",
                f"debug publish source file is missing: {operation.source}",
                path=operation.source,
                label=operation.workflow_operation.label,
            )
            warnings.append(issue)
            missing_source_issues[operation.workflow_operation.label] = issue
            continue
        executable_operations.append(operation)
    execution = None
    results_by_label = {}
    if executable_operations:
        execution = executor.execute_many(
            tuple(operation.workflow_operation for operation in executable_operations),
            progress_session=progress_session,
        )
        results_by_label = {result.label: result for result in execution.results}
    errors: list[DebugIssue] = []
    operation_results: list[DebugPublishOperationResult] = []
    for operation in plan.operations:
        missing_source_issue = missing_source_issues.get(operation.workflow_operation.label)
        if missing_source_issue is not None:
            operation_results.append(
                DebugPublishOperationResult(
                    source=operation.source,
                    target=operation.target,
                    status=DebugPublishOperationStatus.SKIPPED_MISSING_SOURCE,
                    message=missing_source_issue.message,
                )
            )
            continue
        result = results_by_label.get(operation.workflow_operation.label)
        if result is None:
            errors.append(
                _issue(
                    "publish_operation_not_executed",
                    f"debug publish operation was not executed: {operation.description}",
                    path=operation.target,
                    label=operation.workflow_operation.label,
                )
            )
            operation_results.append(
                DebugPublishOperationResult(
                    source=operation.source,
                    target=operation.target,
                    status=DebugPublishOperationStatus.FAILED,
                    message="operation was not executed by the batch executor",
                )
            )
            continue
        if result.succeeded:
            operation_results.append(
                DebugPublishOperationResult(
                    source=operation.source,
                    target=operation.target,
                    status=DebugPublishOperationStatus.PUBLISHED,
                    message=None,
                )
            )
            continue
        message = result.failure.failure_message if result.failure else "publish failed"
        errors.append(
            _issue(
                "publish_operation_failed",
                f"{operation.description}: {message}",
                path=operation.target,
                label=operation.workflow_operation.label,
            )
        )
        operation_results.append(
            DebugPublishOperationResult(
                source=operation.source,
                target=operation.target,
                status=DebugPublishOperationStatus.FAILED,
                message=message,
            )
        )
    return DebugPublishResult(
        plan=plan,
        operation_results=tuple(operation_results),
        execution=execution,
        published_file_count=sum(
            1
            for result in operation_results
            if result.status is DebugPublishOperationStatus.PUBLISHED
        ),
        blocking_errors=tuple(sorted(errors, key=_issue_sort_key)),
        warnings=tuple(sorted(warnings, key=_issue_sort_key)),
    )


def verify_debug_publish(result: DebugPublishResult) -> DebugPublishVerification:
    missing_targets = tuple(
        operation_result.target
        for operation_result in result.operation_results
        if operation_result.status is DebugPublishOperationStatus.PUBLISHED
        and not operation_result.target.is_file()
    )
    warnings = result.warnings
    if result.blocking_errors or missing_targets:
        status = DebugPublishVerificationStatus.FAILED
    elif warnings or result.published_file_count != result.plan.expected_file_count:
        status = DebugPublishVerificationStatus.VERIFIED_WITH_WARNINGS
    else:
        status = DebugPublishVerificationStatus.VERIFIED
    return DebugPublishVerification(
        drive_output_root=result.plan.drive_output_root,
        verified_file_count=sum(
            1
            for operation_result in result.operation_results
            if operation_result.status is DebugPublishOperationStatus.PUBLISHED
            and operation_result.target.is_file()
        ),
        missing_targets=tuple(sorted(missing_targets)),
        status=status,
        warnings=warnings,
    )


def _drive_output_root_for_runtime_root(
    layout: DebugLayout,
    runtime_output_root: Path,
    errors: list[DebugIssue],
) -> Path:
    runtime_root = runtime_debug_samples_root(layout)
    try:
        relative = runtime_output_root.relative_to(runtime_root)
    except ValueError:
        errors.append(
            _issue(
                "runtime_output_root_outside_debug_reports",
                "debug runtime output root is outside the configured debug report root",
                path=runtime_output_root,
            )
        )
        return drive_debug_samples_root(layout) / runtime_output_root.name
    return drive_debug_samples_root(layout) / relative


def _publish_operation(*, source: Path, target: Path) -> DebugPublishOperation:
    description = f"publish debug output {source.name}"
    input_bytes = _maybe_input_bytes(source)
    return DebugPublishOperation(
        source=source,
        target=target,
        required=True,
        description=description,
        workflow_operation=FileCopyOperation(
            label=description,
            source_path=source,
            target_path=target,
            failure_message=f"Failed to {description}",
            overwrite_policy="atomic_replace",
            expected_input_bytes=input_bytes,
            progress=_publish_progress_spec(input_bytes),
        ),
    )


def _issue(
    code: str,
    message: str,
    *,
    path: Path | None = None,
    label: str | None = None,
) -> DebugIssue:
    return DebugIssue(code=code, message=message, path=path, label=label)


def _issue_sort_key(issue: DebugIssue) -> tuple[str, str, str]:
    return (
        issue.code,
        "" if issue.label is None else issue.label,
        "" if issue.path is None else issue.path.as_posix(),
    )


def _maybe_input_bytes(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _publish_progress_spec(expected_total: int | None) -> OperationProgressSpec:
    return OperationProgressSpec(
        stage=ProgressStageSpec(
            workflow_id=DEBUG_WORKFLOW_NAME,
            stage_id=DEBUG_STAGE_OUTPUT_PUBLISH,
            label="Publish debug outputs",
            unit="bytes",
            owner_module="text_to_sign_production.workflows.debug.runtime.publish",
            split_behavior="global",
            operation_kind="publish",
            total_semantics="debug output file bytes copied to Drive",
            bar_eligible=False,
        ),
        expected_total=expected_total,
        live_owner="shell",
    )


__all__ = [
    "build_debug_publish_plan",
    "publish_debug_outputs",
    "verify_debug_publish",
]
