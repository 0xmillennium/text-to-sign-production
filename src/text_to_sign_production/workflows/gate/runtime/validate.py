from __future__ import annotations

from text_to_sign_production.workflows.foundation.execution import (
    ArchiveExtractOperation,
    FileCopyOperation,
)
from text_to_sign_production.workflows.gate.contracts import (
    GateRuntimePlan,
    GateWorkflowConfig,
    GateWorkflowInvariantError,
)
from text_to_sign_production.workflows.gate.layout import GateLayout


def validate_gate_runtime_plan(
    config: GateWorkflowConfig,
    layout: GateLayout,
    plan: GateRuntimePlan,
) -> None:
    _validate_config_layout_alignment(config, layout, plan)
    _validate_split_alignment(config, layout, plan)
    _validate_execution_inputs_alignment(layout, plan)
    _validate_restore_operation_alignment(config, layout, plan)


def _validate_config_layout_alignment(
    config: GateWorkflowConfig,
    layout: GateLayout,
    plan: GateRuntimePlan,
) -> None:
    if layout.config != config:
        raise GateWorkflowInvariantError("layout.config must equal config")
    expected_gates_config_original_path = config.project_root / config.gates_config_relpath
    if layout.runtime.gates_config_original_path != expected_gates_config_original_path:
        raise GateWorkflowInvariantError("layout original gates config path is misaligned")
    if plan.execution_inputs.gates_config_path != layout.runtime.gates_config_path:
        raise GateWorkflowInvariantError("plan gates config path is misaligned")


def _validate_split_alignment(
    config: GateWorkflowConfig,
    layout: GateLayout,
    plan: GateRuntimePlan,
) -> None:
    expected = _expected_split_sequence(config)
    drive_splits = tuple(split_layout.split for split_layout in layout.drive_splits)
    runtime_splits = tuple(split_layout.split for split_layout in layout.runtime.splits)
    execution_splits = tuple(
        split_input.split for split_input in plan.execution_inputs.split_inputs
    )
    if drive_splits != expected:
        raise GateWorkflowInvariantError("drive split sequence is misaligned")
    if runtime_splits != expected:
        raise GateWorkflowInvariantError("runtime split sequence is misaligned")
    if execution_splits != expected:
        raise GateWorkflowInvariantError("execution split sequence is misaligned")


def _validate_execution_inputs_alignment(
    layout: GateLayout,
    plan: GateRuntimePlan,
) -> None:
    for runtime_split, execution_input in zip(
        layout.runtime.splits,
        plan.execution_inputs.split_inputs,
        strict=True,
    ):
        if execution_input.translation_csv_path != runtime_split.translation_csv_path:
            raise GateWorkflowInvariantError("execution translation CSV path is misaligned")
        if execution_input.keypoint_root != runtime_split.keypoint_root:
            raise GateWorkflowInvariantError("execution keypoint root is misaligned")
        if execution_input.keypoint_json_root != runtime_split.keypoint_json_root:
            raise GateWorkflowInvariantError("execution keypoint JSON root is misaligned")
        if execution_input.keypoint_video_root != runtime_split.keypoint_video_root:
            raise GateWorkflowInvariantError("execution keypoint video root is misaligned")


def _validate_restore_operation_alignment(
    config: GateWorkflowConfig,
    layout: GateLayout,
    plan: GateRuntimePlan,
) -> None:
    expected_operation_count = 1 + (2 * len(config.splits))
    if len(plan.restore_operations) != expected_operation_count:
        raise GateWorkflowInvariantError("restore operation count is misaligned")

    config_copy_operation = plan.restore_operations[0]
    if not isinstance(config_copy_operation, FileCopyOperation):
        raise GateWorkflowInvariantError("first restore operation must snapshot gates config")
    if config_copy_operation.source_path != layout.runtime.gates_config_original_path:
        raise GateWorkflowInvariantError("gates config snapshot source path is misaligned")
    if config_copy_operation.target_path != layout.runtime.gates_config_path:
        raise GateWorkflowInvariantError("gates config snapshot target path is misaligned")

    for index, (drive_split, runtime_split) in enumerate(
        zip(layout.drive_splits, layout.runtime.splits, strict=True)
    ):
        operation_index = 1 + (index * 2)
        copy_operation = plan.restore_operations[operation_index]
        extract_operation = plan.restore_operations[operation_index + 1]
        if not isinstance(copy_operation, FileCopyOperation):
            raise GateWorkflowInvariantError("restore operation pair must start with file copy")
        if not isinstance(extract_operation, ArchiveExtractOperation):
            raise GateWorkflowInvariantError("restore operation pair must end with archive extract")
        if copy_operation.source_path != drive_split.translation_csv_path:
            raise GateWorkflowInvariantError("translation copy source path is misaligned")
        if copy_operation.target_path != runtime_split.translation_csv_path:
            raise GateWorkflowInvariantError("translation copy target path is misaligned")
        if extract_operation.archive_path != drive_split.keypoint_archive_path:
            raise GateWorkflowInvariantError("keypoint extract archive path is misaligned")
        if extract_operation.extraction_root != layout.runtime.keypoint_extract_root:
            raise GateWorkflowInvariantError("keypoint extract root is misaligned")


def _expected_split_sequence(config: GateWorkflowConfig) -> tuple[str, ...]:
    return config.splits
