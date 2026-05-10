from __future__ import annotations

from text_to_sign_production.workflows.foundation.execution import (
    ArchiveExtractOperation,
    FileCopyOperation,
)
from text_to_sign_production.workflows.samples.contracts import (
    SamplesRuntimePlan,
    SamplesWorkflowConfig,
    SamplesWorkflowInvariantError,
)
from text_to_sign_production.workflows.samples.layout import SamplesLayout


def validate_samples_runtime_plan(
    config: SamplesWorkflowConfig,
    layout: SamplesLayout,
    plan: SamplesRuntimePlan,
) -> None:
    _validate_config_layout_alignment(config, layout, plan)
    _validate_split_alignment(config, layout, plan)
    _validate_execution_inputs_alignment(layout, plan)
    _validate_restore_operation_alignment(config, layout, plan)


def _validate_config_layout_alignment(
    config: SamplesWorkflowConfig,
    layout: SamplesLayout,
    plan: SamplesRuntimePlan,
) -> None:
    if layout.config != config:
        raise SamplesWorkflowInvariantError("layout.config must equal config")
    expected_gates_config_path = config.project_root / config.gates_config_relpath
    if layout.runtime.gates_config_path != expected_gates_config_path:
        raise SamplesWorkflowInvariantError("layout runtime gates config path is misaligned")
    if plan.execution_inputs.gates_config_path != layout.runtime.gates_config_path:
        raise SamplesWorkflowInvariantError("plan gates config path is misaligned")
    if (
        plan.execution_inputs.translation_canonical_text_column
        != config.translation_canonical_text_column
    ):
        raise SamplesWorkflowInvariantError("translation canonical text column is misaligned")


def _validate_split_alignment(
    config: SamplesWorkflowConfig,
    layout: SamplesLayout,
    plan: SamplesRuntimePlan,
) -> None:
    expected = _expected_split_sequence(config)
    drive_splits = tuple(split_layout.split for split_layout in layout.drive_splits)
    runtime_splits = tuple(split_layout.split for split_layout in layout.runtime.splits)
    execution_splits = tuple(
        split_input.split for split_input in plan.execution_inputs.split_inputs
    )
    if drive_splits != expected:
        raise SamplesWorkflowInvariantError("drive split sequence is misaligned")
    if runtime_splits != expected:
        raise SamplesWorkflowInvariantError("runtime split sequence is misaligned")
    if execution_splits != expected:
        raise SamplesWorkflowInvariantError("execution split sequence is misaligned")


def _validate_execution_inputs_alignment(
    layout: SamplesLayout,
    plan: SamplesRuntimePlan,
) -> None:
    for runtime_split, execution_input in zip(
        layout.runtime.splits,
        plan.execution_inputs.split_inputs,
        strict=True,
    ):
        if execution_input.translation_csv_path != runtime_split.translation_csv_path:
            raise SamplesWorkflowInvariantError("execution translation CSV path is misaligned")
        if execution_input.keypoint_root != runtime_split.keypoint_root:
            raise SamplesWorkflowInvariantError("execution keypoint root is misaligned")
        if execution_input.keypoint_json_root != runtime_split.keypoint_json_root:
            raise SamplesWorkflowInvariantError("execution keypoint JSON root is misaligned")
        if execution_input.keypoint_video_root != runtime_split.keypoint_video_root:
            raise SamplesWorkflowInvariantError("execution keypoint video root is misaligned")


def _validate_restore_operation_alignment(
    config: SamplesWorkflowConfig,
    layout: SamplesLayout,
    plan: SamplesRuntimePlan,
) -> None:
    expected_operation_count = 2 * len(config.splits)
    if len(plan.restore_operations) != expected_operation_count:
        raise SamplesWorkflowInvariantError("restore operation count is misaligned")

    for index, (drive_split, runtime_split) in enumerate(
        zip(layout.drive_splits, layout.runtime.splits, strict=True)
    ):
        copy_operation = plan.restore_operations[index * 2]
        extract_operation = plan.restore_operations[index * 2 + 1]
        if not isinstance(copy_operation, FileCopyOperation):
            raise SamplesWorkflowInvariantError("restore operation pair must start with file copy")
        if not isinstance(extract_operation, ArchiveExtractOperation):
            raise SamplesWorkflowInvariantError(
                "restore operation pair must end with archive extract"
            )
        if copy_operation.source_path != drive_split.translation_csv_path:
            raise SamplesWorkflowInvariantError("translation copy source path is misaligned")
        if copy_operation.target_path != runtime_split.translation_csv_path:
            raise SamplesWorkflowInvariantError("translation copy target path is misaligned")
        if extract_operation.archive_path != drive_split.keypoint_archive_path:
            raise SamplesWorkflowInvariantError("keypoint extract archive path is misaligned")
        if extract_operation.extraction_root != layout.runtime.keypoint_extract_root:
            raise SamplesWorkflowInvariantError("keypoint extract root is misaligned")


def _expected_split_sequence(config: SamplesWorkflowConfig) -> tuple[str, ...]:
    return config.splits
