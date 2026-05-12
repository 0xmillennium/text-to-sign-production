from __future__ import annotations

from pathlib import Path

from text_to_sign_production.workflows.foundation.execution import (
    ArchiveExtractOperation,
    FileCopyOperation,
)
from text_to_sign_production.workflows.tier.contracts import (
    TierRuntimePlan,
    TierWorkflowConfig,
    TierWorkflowInputError,
    TierWorkflowInvariantError,
)
from text_to_sign_production.workflows.tier.layout import (
    TierDriveSplitLayout,
    TierLayout,
    TierRuntimeSplitLayout,
)


def validate_tier_runtime_plan(
    config: TierWorkflowConfig,
    layout: TierLayout,
    plan: TierRuntimePlan,
) -> None:
    _validate_input_surfaces(config, layout, plan)
    _validate_config_layout_alignment(config, layout)
    _validate_split_alignment(config, layout, plan)
    _validate_execution_inputs_alignment(layout, plan)
    _validate_restore_operation_alignment(config, layout, plan)
    _validate_passed_root_alignment(layout)


def _validate_input_surfaces(
    config: TierWorkflowConfig,
    layout: TierLayout,
    plan: TierRuntimePlan,
) -> None:
    if not isinstance(config, TierWorkflowConfig):
        raise TierWorkflowInputError("config must be a TierWorkflowConfig")
    if not isinstance(layout, TierLayout):
        raise TierWorkflowInputError("layout must be a TierLayout")
    if not isinstance(plan, TierRuntimePlan):
        raise TierWorkflowInputError("plan must be a TierRuntimePlan")


def _validate_config_layout_alignment(
    config: TierWorkflowConfig,
    layout: TierLayout,
) -> None:
    if layout.config != config:
        raise TierWorkflowInvariantError("runtime layout config must match workflow config")
    if (
        layout.runtime.filters_config_original_path
        != config.project_root / config.filters_config_relpath
    ):
        raise TierWorkflowInvariantError("filters config original path is misaligned")
    if layout.runtime.tier_config_original_path != config.project_root / config.tier_config_relpath:
        raise TierWorkflowInvariantError("tier config original path is misaligned")


def _validate_split_alignment(
    config: TierWorkflowConfig,
    layout: TierLayout,
    plan: TierRuntimePlan,
) -> None:
    expected_splits = _expected_split_sequence(config)
    observed_sequences = (
        tuple(drive_split.split for drive_split in layout.drive_splits),
        tuple(runtime_split.split for runtime_split in layout.runtime.splits),
        tuple(split_input.split for split_input in plan.execution_inputs.split_inputs),
    )
    for observed_splits in observed_sequences:
        if observed_splits != expected_splits:
            raise TierWorkflowInvariantError("tier runtime split sequence is misaligned")


def _validate_execution_inputs_alignment(
    layout: TierLayout,
    plan: TierRuntimePlan,
) -> None:
    execution_inputs = plan.execution_inputs
    if execution_inputs.filters_config_path != layout.runtime.filters_config_path:
        raise TierWorkflowInvariantError("filters config execution input is misaligned")
    if execution_inputs.tier_config_path != layout.runtime.tier_config_path:
        raise TierWorkflowInvariantError("tier config execution input is misaligned")
    if execution_inputs.passed_samples_root != layout.runtime.passed_samples_root:
        raise TierWorkflowInvariantError("passed samples root execution input is misaligned")

    runtime_splits = {runtime_split.split: runtime_split for runtime_split in layout.runtime.splits}
    for split_input in execution_inputs.split_inputs:
        runtime_split = runtime_splits.get(split_input.split)
        if runtime_split is None:
            raise TierWorkflowInvariantError(
                f"Missing runtime split layout for execution input: {split_input.split}"
            )
        if split_input.passed_manifest_path != runtime_split.passed_manifest_path:
            raise TierWorkflowInvariantError("passed manifest execution input is misaligned")
        if split_input.passed_samples_split_root != runtime_split.passed_samples_split_root:
            raise TierWorkflowInvariantError(
                "passed samples split root execution input is misaligned"
            )


def _validate_restore_operation_alignment(
    config: TierWorkflowConfig,
    layout: TierLayout,
    plan: TierRuntimePlan,
) -> None:
    expected_operation_count = 2 + (2 * len(config.splits))
    if len(plan.restore_operations) != expected_operation_count:
        raise TierWorkflowInvariantError("tier runtime restore operation count is misaligned")

    filters_config_copy = plan.restore_operations[0]
    tier_config_copy = plan.restore_operations[1]
    if not isinstance(filters_config_copy, FileCopyOperation):
        raise TierWorkflowInvariantError("first restore operation must snapshot filters config")
    if not isinstance(tier_config_copy, FileCopyOperation):
        raise TierWorkflowInvariantError("second restore operation must snapshot tier config")
    if filters_config_copy.source_path != layout.runtime.filters_config_original_path:
        raise TierWorkflowInvariantError("filters config snapshot source path is misaligned")
    if filters_config_copy.target_path != layout.runtime.filters_config_path:
        raise TierWorkflowInvariantError("filters config snapshot target path is misaligned")
    if tier_config_copy.source_path != layout.runtime.tier_config_original_path:
        raise TierWorkflowInvariantError("tier config snapshot source path is misaligned")
    if tier_config_copy.target_path != layout.runtime.tier_config_path:
        raise TierWorkflowInvariantError("tier config snapshot target path is misaligned")

    drive_splits = {drive_split.split: drive_split for drive_split in layout.drive_splits}
    runtime_splits = {runtime_split.split: runtime_split for runtime_split in layout.runtime.splits}
    passed_extract_root = layout.stores.runtime.samples.split_extract_root("passed").path
    for index, split in enumerate(config.splits):
        drive_split = _required_drive_split(drive_splits, split)
        runtime_split = _required_runtime_split(runtime_splits, split)
        operation_index = 2 + (2 * index)
        copy_operation = plan.restore_operations[operation_index]
        extract_operation = plan.restore_operations[operation_index + 1]

        if not isinstance(copy_operation, FileCopyOperation):
            raise TierWorkflowInvariantError("passed manifest restore operation must be file copy")
        if not isinstance(extract_operation, ArchiveExtractOperation):
            raise TierWorkflowInvariantError(
                "passed archive restore operation must be archive extract"
            )

        if copy_operation.source_path != drive_split.passed_manifest_path:
            raise TierWorkflowInvariantError("passed manifest restore source path is misaligned")
        if copy_operation.target_path != runtime_split.passed_manifest_path:
            raise TierWorkflowInvariantError("passed manifest restore target path is misaligned")
        if extract_operation.archive_path != drive_split.passed_archive_path:
            raise TierWorkflowInvariantError("passed archive restore source path is misaligned")
        if extract_operation.extraction_root != passed_extract_root:
            raise TierWorkflowInvariantError("passed archive restore extraction root is misaligned")


def _validate_passed_root_alignment(layout: TierLayout) -> None:
    passed_root = layout.runtime.passed_samples_root
    for runtime_split in layout.runtime.splits:
        if runtime_split.passed_samples_split_root == passed_root:
            raise TierWorkflowInvariantError(
                "passed samples split root must be below passed samples root"
            )
        if not _is_relative_to(runtime_split.passed_samples_split_root, passed_root):
            raise TierWorkflowInvariantError(
                "passed samples split root is outside passed samples root"
            )


def _expected_split_sequence(config: TierWorkflowConfig) -> tuple[str, ...]:
    return tuple(config.splits)


def _required_drive_split(
    drive_splits: dict[str, TierDriveSplitLayout],
    split: str,
) -> TierDriveSplitLayout:
    drive_split = drive_splits.get(split)
    if drive_split is None:
        raise TierWorkflowInvariantError(f"Missing drive split layout: {split}")
    return drive_split


def _required_runtime_split(
    runtime_splits: dict[str, TierRuntimeSplitLayout],
    split: str,
) -> TierRuntimeSplitLayout:
    runtime_split = runtime_splits.get(split)
    if runtime_split is None:
        raise TierWorkflowInvariantError(f"Missing runtime split layout: {split}")
    return runtime_split


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
