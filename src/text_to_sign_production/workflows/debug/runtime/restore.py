from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.progress import ProgressStageSpec
from text_to_sign_production.core.ids import SampleSplit, TierMembership, TierName
from text_to_sign_production.workflows.debug.contracts import (
    DebugIssue,
    DebugRestoreOperation,
    DebugRestoreOperationResult,
    DebugRestorePlan,
    DebugRestorePlanValidation,
    DebugRestoreResult,
    DebugWorkflowConfig,
)
from text_to_sign_production.workflows.debug.contracts.config import (
    CANONICAL_FILTERS_CONFIG_RELPATH,
    CANONICAL_GATES_CONFIG_RELPATH,
    CANONICAL_TIERS_CONFIG_RELPATH,
)
from text_to_sign_production.workflows.debug.constants import (
    DEBUG_STAGE_RUNTIME_RESTORE,
    DEBUG_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.debug.contracts.verdicts import (
    DebugRestoreOperationStatus,
)
from text_to_sign_production.workflows.debug.layout import (
    DebugLayout,
    project_config_path,
)
from text_to_sign_production.workflows.foundation.execution import (
    ArchiveExtractOperation,
    FileCopyOperation,
    OperationProgressSpec,
    WorkflowExecutor,
)

SPLIT_SCOPED_ARCHIVE_GROUPS = frozenset(("raw_sources", "gate_outputs"))


def build_debug_restore_plan(
    config: DebugWorkflowConfig,
    debug_splits: tuple[SampleSplit, ...],
    *,
    layout: DebugLayout,
) -> DebugRestorePlan:
    runtime = layout.runtime
    drive = layout.drive
    operations: list[DebugRestoreOperation] = []
    for name in ("gates.yaml", "filters.yaml", "tiers.yaml"):
        source = project_config_path(config, name)
        target = config.configs_runtime_root / name
        operations.append(
            _copy_op(
                group="configs",
                split=None,
                source=source,
                target=target,
                required=True,
                description=f"snapshot {name}",
            )
        )

    for split in debug_splits:
        operations.append(
            _copy_op(
                group="translations",
                split=split,
                source=drive.assets.translation_csv(split).path,
                target=runtime.assets.translation_csv(split).path,
                required=True,
                description=f"restore How2Sign translation file [{split.value}]",
            )
        )
        operations.append(
            _extract_op(
                group="raw_sources",
                split=split,
                source=drive.assets.keypoint_archive(split).path,
                target=runtime.assets.keypoint_extract_root().path,
                required=False,
                description=f"restore BFH/OpenPose keypoint archive [{split.value}]",
            )
        )
        operations.extend(
            (
                _copy_op(
                    group="gate_outputs",
                    split=split,
                    source=drive.manifests.untiered_passed_manifest(split).path,
                    target=runtime.manifests.untiered_passed_manifest(split).path,
                    required=False,
                    description=f"restore untiered passed manifest [{split.value}]",
                ),
                _copy_op(
                    group="gate_outputs",
                    split=split,
                    source=drive.manifests.untiered_dropped_manifest(split).path,
                    target=runtime.manifests.untiered_dropped_manifest(split).path,
                    required=False,
                    description=f"restore untiered dropped manifest [{split.value}]",
                ),
                _extract_op(
                    group="gate_outputs",
                    split=split,
                    source=drive.samples.split_archive("passed", split).path,
                    target=runtime.samples.split_extract_root("passed").path,
                    required=False,
                    description=f"restore passed sample payload archive [{split.value}]",
                ),
                _extract_op(
                    group="gate_outputs",
                    split=split,
                    source=drive.samples.split_archive("dropped", split).path,
                    target=runtime.samples.split_extract_root("dropped").path,
                    required=False,
                    description=f"restore dropped sample record archive [{split.value}]",
                ),
            )
        )
        for tier in TierName:
            for membership in TierMembership:
                operations.append(
                    _copy_op(
                        group="tier_outputs",
                        split=split,
                        source=drive.manifests.tiered_manifest(tier, membership, split).path,
                        target=runtime.manifests.tiered_manifest(tier, membership, split).path,
                        required=False,
                        description=(
                            f"restore {tier.value} {membership.value} manifest [{split.value}]"
                        ),
                    )
                )
    return DebugRestorePlan(
        debug_splits=debug_splits,
        operations=tuple(operations),
        required_groups=("configs", "translations"),
        optional_groups=("raw_sources", "gate_outputs", "tier_outputs"),
    )


def validate_debug_restore_plan(plan: DebugRestorePlan) -> DebugRestorePlanValidation:
    errors: list[DebugIssue] = []
    warnings: list[DebugIssue] = []
    seen_labels: dict[str, DebugRestoreOperation] = {}
    seen_file_targets: dict[Path, DebugRestoreOperation] = {}
    seen_archive_operations: dict[
        tuple[Path, Path, str],
        DebugRestoreOperation,
    ] = {}
    seen_archive_roots: dict[tuple[Path, str], DebugRestoreOperation] = {}
    if not plan.operations:
        errors.append(_issue("empty_plan", "restore plan is empty"))
    if not plan.debug_splits:
        errors.append(_issue("empty_debug_splits", "DEBUG_SPLITS is empty"))
    for operation in plan.operations:
        label = operation.workflow_operation.label
        if label in seen_labels:
            errors.append(
                _issue(
                    "duplicate_operation_label",
                    f"duplicate restore operation label: {label}",
                    label=label,
                )
            )
        seen_labels[label] = operation
        resolved_target = operation.target.resolve(strict=False)
        workflow_operation = operation.workflow_operation
        if isinstance(workflow_operation, FileCopyOperation):
            previous_target = seen_file_targets.get(resolved_target)
            if previous_target is not None:
                errors.append(
                    _issue(
                        "duplicate_target_path",
                        f"duplicate file-copy restore target path: {operation.target}",
                        path=operation.target,
                        label=label,
                    )
                )
            seen_file_targets[resolved_target] = operation
        elif isinstance(workflow_operation, ArchiveExtractOperation):
            resolved_source = operation.source.resolve(strict=False)
            archive_key = (resolved_source, resolved_target, operation.group)
            if archive_key in seen_archive_operations:
                errors.append(
                    _issue(
                        "duplicate_archive_extract_operation",
                        "duplicate archive extraction restore operation: "
                        f"{operation.source} -> {operation.target} [{operation.group}]",
                        path=operation.target,
                        label=label,
                    )
                )
            seen_archive_operations[archive_key] = operation
            root_key = (resolved_target, operation.group)
            previous_root = seen_archive_roots.get(root_key)
            if previous_root is not None and operation.group in SPLIT_SCOPED_ARCHIVE_GROUPS:
                warnings.append(
                    _issue(
                        "shared_archive_extraction_root",
                        "multiple split-scoped archives extract into the same restore root: "
                        f"{operation.target}",
                        path=operation.target,
                        label=label,
                    )
                )
            seen_archive_roots[root_key] = operation
        if operation.source.resolve(strict=False) == resolved_target:
            errors.append(
                _issue(
                    "source_equals_target",
                    f"restore source and target are the same path: {operation.source}",
                    path=operation.source,
                    label=label,
                )
            )
        if operation.workflow_operation is None:
            errors.append(
                _issue(
                    "missing_foundation_operation",
                    f"restore operation has no foundation operation: {operation.description}",
                    label=label,
                )
            )
        if not operation.source.exists():
            message = f"{operation.description}: source does not exist: {operation.source}"
            if operation.required:
                errors.append(
                    _issue(
                        "required_source_missing",
                        message,
                        path=operation.source,
                        label=label,
                    )
                )
            else:
                warnings.append(
                    _issue(
                        "optional_source_missing",
                        message,
                        path=operation.source,
                        label=label,
                    )
                )
        parent_issue = _target_parent_issue(operation)
        if parent_issue is not None:
            errors.append(parent_issue)
    errors.extend(_split_coverage_errors(plan))
    errors.extend(_canonical_config_errors(plan))
    errors.extend(_required_group_errors(plan))
    return DebugRestorePlanValidation(
        plan=plan,
        valid=not errors,
        blocking_errors=tuple(sorted(errors, key=_issue_sort_key)),
        warnings=tuple(sorted(warnings, key=_issue_sort_key)),
    )


def execute_debug_restore(
    plan: DebugRestorePlan,
    *,
    executor: WorkflowExecutor,
    progress_session=None,
) -> DebugRestoreResult:
    executable_operations = tuple(
        operation for operation in plan.operations if operation.required or operation.source.exists()
    )
    execution = executor.execute_many(
        tuple(operation.workflow_operation for operation in executable_operations),
        progress_session=progress_session,
    )
    results_by_label = {result.label: result for result in execution.results}
    operation_results: list[DebugRestoreOperationResult] = []
    for operation in plan.operations:
        if not operation.source.exists() and not operation.required:
            message = f"source does not exist: {operation.source}"
            operation_results.append(
                DebugRestoreOperationResult(
                    operation,
                    DebugRestoreOperationStatus.SKIPPED_OPTIONAL,
                    message,
                )
            )
            continue
        result = results_by_label.get(operation.workflow_operation.label)
        if result is None:
            operation_results.append(
                DebugRestoreOperationResult(
                    operation,
                    DebugRestoreOperationStatus.FAILED_REQUIRED
                    if operation.required
                    else DebugRestoreOperationStatus.FAILED_OPTIONAL,
                    "operation was not executed by the batch executor",
                )
            )
            continue
        status = (
            DebugRestoreOperationStatus.SUCCEEDED
            if result.succeeded
            else DebugRestoreOperationStatus.FAILED_REQUIRED
        )
        if not result.succeeded and not operation.required:
            status = DebugRestoreOperationStatus.FAILED_OPTIONAL
        operation_results.append(
            DebugRestoreOperationResult(
                operation,
                status,
                None
                if result.succeeded
                else result.failure.failure_message
                if result.failure
                else None,
            )
        )
    return DebugRestoreResult(
        plan=plan,
        operation_results=tuple(operation_results),
        execution=execution,
    )


def _copy_op(
    *,
    group: str,
    split: SampleSplit | None,
    source: Path,
    target: Path,
    required: bool,
    description: str,
) -> DebugRestoreOperation:
    return DebugRestoreOperation(
        group=group,
        split=split,
        source=source,
        target=target,
        required=required,
        description=description,
        workflow_operation=FileCopyOperation(
            label=description,
            source_path=source,
            target_path=target,
            failure_message=f"Failed to {description}",
            overwrite_policy="atomic_replace",
            expected_input_bytes=_maybe_input_bytes(source),
            progress=_restore_progress_spec(_maybe_input_bytes(source)),
        ),
    )


def _extract_op(
    *,
    group: str,
    split: SampleSplit | None,
    source: Path,
    target: Path,
    required: bool,
    description: str,
) -> DebugRestoreOperation:
    return DebugRestoreOperation(
        group=group,
        split=split,
        source=source,
        target=target,
        required=required,
        description=description,
        workflow_operation=ArchiveExtractOperation(
            label=description,
            archive_path=source,
            extraction_root=target,
            failure_message=f"Failed to {description}",
            compression_kind="tar_zst",
            strip_components=None,
            members=(),
            expected_input_bytes=_maybe_input_bytes(source),
            progress=_restore_progress_spec(_maybe_input_bytes(source)),
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


def _target_parent_issue(operation: DebugRestoreOperation) -> DebugIssue | None:
    parent = operation.target.parent
    if parent.exists():
        if not parent.is_dir():
            return _issue(
                "target_parent_not_directory",
                f"restore target parent is not a directory: {parent}",
                path=parent,
                label=operation.workflow_operation.label,
            )
        return None
    ancestor = parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.exists() or not ancestor.is_dir():
        return _issue(
            "target_parent_not_creatable",
            f"restore target parent has no creatable directory ancestor: {parent}",
            path=parent,
            label=operation.workflow_operation.label,
        )
    return None


def _split_coverage_errors(plan: DebugRestorePlan) -> list[DebugIssue]:
    errors: list[DebugIssue] = []
    for split in plan.debug_splits:
        has_translation = any(
            operation.group == "translations"
            and operation.split == split
            and operation.required
            for operation in plan.operations
        )
        if not has_translation:
            errors.append(
                _issue(
                    "missing_translation_operation",
                    f"selected split has no required translation restore operation: {split.value}",
                    label=split.value,
                )
            )
    return errors


def _canonical_config_errors(plan: DebugRestorePlan) -> list[DebugIssue]:
    required = (
        CANONICAL_GATES_CONFIG_RELPATH,
        CANONICAL_FILTERS_CONFIG_RELPATH,
        CANONICAL_TIERS_CONFIG_RELPATH,
    )
    config_part_count = len(CANONICAL_GATES_CONFIG_RELPATH.parts)
    config_source_suffixes = {
        _path_suffix(operation.source, config_part_count)
        for operation in plan.operations
        if operation.group == "configs" and operation.required
    }
    return [
        _issue(
            "missing_canonical_config_operation",
            f"required canonical config restore operation is missing: {relpath.as_posix()}",
            path=relpath,
            label=relpath.as_posix(),
        )
        for relpath in required
        if relpath not in config_source_suffixes
    ]


def _path_suffix(path: Path, parts: int) -> Path:
    return Path(*path.parts[-parts:])


def _required_group_errors(plan: DebugRestorePlan) -> list[DebugIssue]:
    errors: list[DebugIssue] = []
    for group in plan.required_groups:
        if not any(operation.group == group and operation.required for operation in plan.operations):
            errors.append(
                _issue(
                    "missing_required_group",
                    f"required restore group has no required operations: {group}",
                    label=group,
                )
            )
    return errors


def _maybe_input_bytes(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _restore_progress_spec(expected_total: int | None) -> OperationProgressSpec:
    return OperationProgressSpec(
        stage=ProgressStageSpec(
            workflow_id=DEBUG_WORKFLOW_NAME,
            stage_id=DEBUG_STAGE_RUNTIME_RESTORE,
            label="Restore debug runtime artifacts",
            unit="bytes",
            owner_module="text_to_sign_production.workflows.debug.runtime",
            split_behavior="per_split",
            operation_kind="restore",
            total_semantics="runtime restore input bytes",
            bar_eligible=False,
        ),
        expected_total=expected_total,
        live_owner="shell",
    )


__all__ = [
    "build_debug_restore_plan",
    "execute_debug_restore",
    "validate_debug_restore_plan",
]
