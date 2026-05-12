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
from text_to_sign_production.workflows.gate.constants import (
    GATE_STAGE_RUNTIME_RESTORE,
    GATE_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.gate.contracts import (
    GateRuntimePlan,
    GateSplitRuntimeInputs,
    GateWorkflowConfig,
    GateWorkflowExecutionInputs,
    GateWorkflowInvariantError,
)
from text_to_sign_production.workflows.gate.layout import (
    GateDriveSplitLayout,
    GateLayout,
    GateRuntimeSplitLayout,
)


def build_gate_runtime_plan(
    config: GateWorkflowConfig,
    layout: GateLayout,
) -> GateRuntimePlan:
    if layout.config != config:
        raise GateWorkflowInvariantError("layout.config must match the provided config")
    return GateRuntimePlan(
        restore_operations=_build_restore_operations(config, layout),
        execution_inputs=_build_execution_inputs(layout),
    )


def _build_execution_inputs(layout: GateLayout) -> GateWorkflowExecutionInputs:
    return GateWorkflowExecutionInputs(
        gates_config_path=layout.runtime.gates_config_path,
        gates_config_provenance=file_provenance(
            "gates config",
            layout.runtime.gates_config_original_path,
            execution_path=layout.runtime.gates_config_path,
        ),
        split_inputs=tuple(
            GateSplitRuntimeInputs(
                split=runtime_split.split,
                translation_csv_path=runtime_split.translation_csv_path,
                keypoint_root=runtime_split.keypoint_root,
                keypoint_json_root=runtime_split.keypoint_json_root,
                keypoint_video_root=runtime_split.keypoint_video_root,
            )
            for runtime_split in layout.runtime.splits
        ),
    )


def _build_restore_operations(
    config: GateWorkflowConfig,
    layout: GateLayout,
) -> tuple[WorkflowOperation, ...]:
    if len(layout.drive_splits) != len(config.splits):
        raise GateWorkflowInvariantError("drive split layout count must match config splits")
    if len(layout.runtime.splits) != len(config.splits):
        raise GateWorkflowInvariantError("runtime split layout count must match config splits")

    operations: list[WorkflowOperation] = []
    operations.append(_build_gates_config_copy_operation(layout))
    for split, drive_split, runtime_split in zip(
        config.splits,
        layout.drive_splits,
        layout.runtime.splits,
        strict=True,
    ):
        if drive_split.split != split or runtime_split.split != split:
            raise GateWorkflowInvariantError("layout split sequence must match config splits")
        operations.append(_build_translation_copy_operation(drive_split, runtime_split))
        operations.append(
            _build_keypoint_extract_operation(
                drive_split,
                layout.runtime.keypoint_extract_root,
            )
        )
    return tuple(operations)


def _build_gates_config_copy_operation(layout: GateLayout) -> FileCopyOperation:
    expected_input_bytes = _maybe_input_bytes(layout.runtime.gates_config_original_path)
    return FileCopyOperation(
        label="snapshot gates config",
        source_path=layout.runtime.gates_config_original_path,
        target_path=layout.runtime.gates_config_path,
        failure_message="Failed to snapshot gates config for gate execution",
        overwrite_policy="atomic_replace",
        expected_input_bytes=expected_input_bytes,
        progress=_restore_progress_spec(expected_input_bytes),
    )


def _build_translation_copy_operation(
    drive_split: GateDriveSplitLayout,
    runtime_split: GateRuntimeSplitLayout,
) -> FileCopyOperation:
    expected_input_bytes = _maybe_input_bytes(drive_split.translation_csv_path)
    return FileCopyOperation(
        label=f"restore translation csv [{drive_split.split}]",
        source_path=drive_split.translation_csv_path,
        target_path=runtime_split.translation_csv_path,
        failure_message=f"Failed to restore translation CSV for split '{drive_split.split}'",
        overwrite_policy="atomic_replace",
        expected_input_bytes=expected_input_bytes,
        progress=_restore_progress_spec(expected_input_bytes),
    )


def _build_keypoint_extract_operation(
    drive_split: GateDriveSplitLayout,
    keypoint_extract_root: Path,
) -> ArchiveExtractOperation:
    expected_input_bytes = _maybe_input_bytes(drive_split.keypoint_archive_path)
    return ArchiveExtractOperation(
        label=f"restore keypoint archive [{drive_split.split}]",
        archive_path=drive_split.keypoint_archive_path,
        extraction_root=keypoint_extract_root,
        failure_message=f"Failed to restore keypoint archive for split '{drive_split.split}'",
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
            workflow_id=GATE_WORKFLOW_NAME,
            stage_id=GATE_STAGE_RUNTIME_RESTORE,
            label="Restore gate runtime assets",
            unit="bytes",
            owner_module="text_to_sign_production.workflows.gate.runtime",
            split_behavior="global",
            operation_kind="restore",
            total_semantics="input bytes when known",
            bar_eligible=False,
        ),
        expected_total=expected_total,
        live_owner="shell",
    )
