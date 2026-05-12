from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.progress import ProgressStageSpec
from text_to_sign_production.workflows.foundation.execution import (
    ArchiveExtractOperation,
    FileCopyOperation,
    OperationProgressSpec,
    WorkflowOperation,
)
from text_to_sign_production.workflows.foundation.provenance import file_provenance
from text_to_sign_production.workflows.tier.constants import (
    TIER_STAGE_RUNTIME_RESTORE,
    TIER_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tier.contracts import (
    TierRuntimePlan,
    TierSplitRuntimeInputs,
    TierWorkflowConfig,
    TierWorkflowExecutionInputs,
    TierWorkflowInvariantError,
)
from text_to_sign_production.workflows.tier.layout import (
    TierDriveSplitLayout,
    TierLayout,
    TierRuntimeSplitLayout,
)


def build_tier_runtime_plan(
    config: TierWorkflowConfig,
    layout: TierLayout,
) -> TierRuntimePlan:
    if layout.config != config:
        raise TierWorkflowInvariantError("runtime layout config must match workflow config")
    return TierRuntimePlan(
        restore_operations=_build_restore_operations(config, layout),
        execution_inputs=_build_execution_inputs(layout),
    )


def _build_execution_inputs(layout: TierLayout) -> TierWorkflowExecutionInputs:
    return TierWorkflowExecutionInputs(
        filters_config_path=layout.runtime.filters_config_path,
        filters_config_provenance=file_provenance(
            "filters config",
            layout.runtime.filters_config_original_path,
            execution_path=layout.runtime.filters_config_path,
        ),
        tier_config_path=layout.runtime.tier_config_path,
        tier_config_provenance=file_provenance(
            "tier config",
            layout.runtime.tier_config_original_path,
            execution_path=layout.runtime.tier_config_path,
        ),
        passed_samples_root=layout.runtime.passed_samples_root,
        split_inputs=tuple(
            TierSplitRuntimeInputs(
                split=runtime_split.split,
                passed_manifest_path=runtime_split.passed_manifest_path,
                passed_samples_split_root=runtime_split.passed_samples_split_root,
            )
            for runtime_split in layout.runtime.splits
        ),
    )


def _build_restore_operations(
    config: TierWorkflowConfig,
    layout: TierLayout,
) -> tuple[WorkflowOperation, ...]:
    operations: list[WorkflowOperation] = [
        _build_config_copy_operation(
            label="snapshot filters config",
            source_path=layout.runtime.filters_config_original_path,
            target_path=layout.runtime.filters_config_path,
            failure_message="Failed to snapshot filters config for tier execution",
        ),
        _build_config_copy_operation(
            label="snapshot tier config",
            source_path=layout.runtime.tier_config_original_path,
            target_path=layout.runtime.tier_config_path,
            failure_message="Failed to snapshot tier config for tier execution",
        ),
    ]
    drive_splits = {drive_split.split: drive_split for drive_split in layout.drive_splits}
    runtime_splits = {runtime_split.split: runtime_split for runtime_split in layout.runtime.splits}
    for split in config.splits:
        drive_split = drive_splits.get(split)
        runtime_split = runtime_splits.get(split)
        if drive_split is None or runtime_split is None:
            raise TierWorkflowInvariantError(f"Missing runtime restore layout for split: {split}")
        operations.append(
            _build_passed_manifest_copy_operation(
                split=split,
                drive_split=drive_split,
                runtime_split=runtime_split,
            )
        )
        operations.append(
            _build_passed_archive_extract_operation(
                split=split,
                drive_split=drive_split,
                extraction_root=layout.stores.runtime.samples.split_extract_root("passed").path,
            )
        )
    return tuple(operations)


def _build_config_copy_operation(
    *,
    label: str,
    source_path: Path,
    target_path: Path,
    failure_message: str,
) -> FileCopyOperation:
    expected_input_bytes = _maybe_input_bytes(source_path)
    return FileCopyOperation(
        label=label,
        source_path=source_path,
        target_path=target_path,
        failure_message=failure_message,
        overwrite_policy="atomic_replace",
        expected_input_bytes=expected_input_bytes,
        progress=_restore_progress_spec(expected_input_bytes),
    )


def _build_passed_manifest_copy_operation(
    *,
    split: str,
    drive_split: TierDriveSplitLayout,
    runtime_split: TierRuntimeSplitLayout,
) -> FileCopyOperation:
    expected_input_bytes = _maybe_input_bytes(drive_split.passed_manifest_path)
    return FileCopyOperation(
        label=f"restore passed manifest [{split}]",
        source_path=drive_split.passed_manifest_path,
        target_path=runtime_split.passed_manifest_path,
        failure_message=f"Failed to restore passed manifest for split '{split}'",
        overwrite_policy="atomic_replace",
        expected_input_bytes=expected_input_bytes,
        progress=_restore_progress_spec(expected_input_bytes),
    )


def _build_passed_archive_extract_operation(
    *,
    split: str,
    drive_split: TierDriveSplitLayout,
    extraction_root: Path,
) -> ArchiveExtractOperation:
    expected_input_bytes = _maybe_input_bytes(drive_split.passed_archive_path)
    return ArchiveExtractOperation(
        label=f"restore passed samples [{split}]",
        archive_path=drive_split.passed_archive_path,
        extraction_root=extraction_root,
        failure_message=f"Failed to restore passed archive for split '{split}'",
        compression_kind="tar_zst",
        strip_components=None,
        members=(),
        expected_input_bytes=expected_input_bytes,
        progress=_restore_progress_spec(expected_input_bytes),
    )


def _maybe_input_bytes(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _restore_progress_spec(expected_total: int | None) -> OperationProgressSpec:
    return OperationProgressSpec(
        stage=ProgressStageSpec(
            workflow_id=TIER_WORKFLOW_NAME,
            stage_id=TIER_STAGE_RUNTIME_RESTORE,
            label="Restore tier workflow runtime artifacts",
            unit="bytes",
            owner_module="text_to_sign_production.workflows.tier.runtime",
            split_behavior="per_split",
            operation_kind="restore",
            total_semantics="runtime restore input bytes",
            bar_eligible=False,
        ),
        expected_total=expected_total,
        live_owner="shell",
    )
