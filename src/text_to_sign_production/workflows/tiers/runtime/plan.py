from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.progress import ProgressStageSpec
from text_to_sign_production.workflows.foundation.execution import (
    ArchiveExtractOperation,
    FileCopyOperation,
    OperationProgressSpec,
    WorkflowOperation,
)
from text_to_sign_production.workflows.tiers.constants import (
    TIERS_STAGE_RUNTIME_RESTORE,
    TIERS_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tiers.contracts import (
    TiersRuntimePlan,
    TiersSplitRuntimeInputs,
    TiersWorkflowConfig,
    TiersWorkflowExecutionInputs,
    TiersWorkflowInvariantError,
)
from text_to_sign_production.workflows.tiers.layout import (
    TiersDriveSplitLayout,
    TiersLayout,
    TiersRuntimeSplitLayout,
)


def build_tiers_runtime_plan(
    config: TiersWorkflowConfig,
    layout: TiersLayout,
) -> TiersRuntimePlan:
    if layout.config != config:
        raise TiersWorkflowInvariantError("runtime layout config must match workflow config")
    return TiersRuntimePlan(
        restore_operations=_build_restore_operations(config, layout),
        execution_inputs=_build_execution_inputs(layout),
    )


def _build_execution_inputs(layout: TiersLayout) -> TiersWorkflowExecutionInputs:
    return TiersWorkflowExecutionInputs(
        filters_config_path=layout.runtime.filters_config_path,
        tiers_config_path=layout.runtime.tiers_config_path,
        passed_samples_root=layout.runtime.passed_samples_root,
        split_inputs=tuple(
            TiersSplitRuntimeInputs(
                split=runtime_split.split,
                passed_manifest_path=runtime_split.passed_manifest_path,
                passed_samples_split_root=runtime_split.passed_samples_split_root,
            )
            for runtime_split in layout.runtime.splits
        ),
    )


def _build_restore_operations(
    config: TiersWorkflowConfig,
    layout: TiersLayout,
) -> tuple[WorkflowOperation, ...]:
    operations: list[WorkflowOperation] = []
    drive_splits = {drive_split.split: drive_split for drive_split in layout.drive_splits}
    runtime_splits = {runtime_split.split: runtime_split for runtime_split in layout.runtime.splits}
    for split in config.splits:
        drive_split = drive_splits.get(split)
        runtime_split = runtime_splits.get(split)
        if drive_split is None or runtime_split is None:
            raise TiersWorkflowInvariantError(f"Missing runtime restore layout for split: {split}")
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


def _build_passed_manifest_copy_operation(
    *,
    split: str,
    drive_split: TiersDriveSplitLayout,
    runtime_split: TiersRuntimeSplitLayout,
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
    drive_split: TiersDriveSplitLayout,
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
            workflow_id=TIERS_WORKFLOW_NAME,
            stage_id=TIERS_STAGE_RUNTIME_RESTORE,
            label="Restore tiers workflow runtime artifacts",
            unit="bytes",
            owner_module="text_to_sign_production.workflows.tiers.runtime",
            split_behavior="per_split",
            operation_kind="restore",
            total_semantics="runtime restore input bytes",
            bar_eligible=False,
        ),
        expected_total=expected_total,
        live_owner="shell",
    )
