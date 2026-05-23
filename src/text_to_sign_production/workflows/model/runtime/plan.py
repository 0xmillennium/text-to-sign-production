"""Build runtime restore operations for model workflow inputs."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.workflows.foundation.execution import (
    ArchiveExtractOperation,
    FileCopyOperation,
    OperationProgressSpec,
    WorkflowOperation,
)
from text_to_sign_production.workflows.foundation.provenance import file_provenance
from text_to_sign_production.workflows.model.constants import MODEL_STAGE_RUNTIME_RESTORE
from text_to_sign_production.workflows.model.contracts import (
    ModelRuntimePlan,
    ModelRuntimeSplitInputs,
    ModelWorkflowConfig,
    ModelWorkflowExecutionInputs,
    ModelWorkflowInvariantError,
)
from text_to_sign_production.workflows.model.layout import ModelLayout
from text_to_sign_production.workflows.model.progress import model_progress_stage
from text_to_sign_production.workflows.model.request import (
    build_model_run_request_from_layout,
)


def build_model_runtime_plan(
    config: ModelWorkflowConfig,
    layout: ModelLayout,
) -> ModelRuntimePlan:
    """Build optional config snapshot and input data restore operations."""

    if layout.config != config:
        raise ModelWorkflowInvariantError("layout.config must match the provided config")
    return ModelRuntimePlan(
        restore_operations=_build_restore_operations(layout),
        execution_inputs=_build_execution_inputs(config, layout),
    )


def _build_execution_inputs(
    config: ModelWorkflowConfig,
    layout: ModelLayout,
) -> ModelWorkflowExecutionInputs:
    provenance = None
    if layout.runtime.model_config_original_path is not None:
        if layout.runtime.model_config_path is None:
            raise ModelWorkflowInvariantError("model config snapshot path is missing")
        provenance = file_provenance(
            "model config",
            layout.runtime.model_config_original_path,
            execution_path=layout.runtime.model_config_path,
        )
    semantic_provenance = None
    if layout.runtime.semantic_objective_config_original_path is not None:
        if layout.runtime.semantic_objective_config_path is None:
            raise ModelWorkflowInvariantError("semantic objective config snapshot path is missing")
        semantic_provenance = file_provenance(
            "semantic objective config",
            layout.runtime.semantic_objective_config_original_path,
            execution_path=layout.runtime.semantic_objective_config_path,
        )
    return ModelWorkflowExecutionInputs(
        request=build_model_run_request_from_layout(config, layout),
        model_config_path=layout.runtime.model_config_path,
        model_config_provenance=provenance,
        semantic_objective_config_path=layout.runtime.semantic_objective_config_path,
        semantic_objective_config_provenance=semantic_provenance,
        runtime_topology=layout.stores.runtime,
        model_run_root=layout.outputs.model_run_root,
        report_root=layout.reports.root,
        split_inputs=tuple(
            ModelRuntimeSplitInputs(
                split=split_layout.split,
                manifest_path=split_layout.runtime_manifest_path,
                passed_samples_split_root=split_layout.runtime_passed_samples_split_root,
            )
            for split_layout in layout.runtime.splits
        ),
    )


def _build_restore_operations(layout: ModelLayout) -> tuple[WorkflowOperation, ...]:
    operations: list[WorkflowOperation] = []
    if layout.runtime.model_config_original_path is not None:
        if layout.runtime.model_config_path is None:
            raise ModelWorkflowInvariantError("model config snapshot path is missing")
        operations.append(
            _copy_operation(
                label="snapshot model config",
                source_path=layout.runtime.model_config_original_path,
                target_path=layout.runtime.model_config_path,
            )
        )
    if layout.runtime.semantic_objective_config_original_path is not None:
        if layout.runtime.semantic_objective_config_path is None:
            raise ModelWorkflowInvariantError("semantic objective config snapshot path is missing")
        operations.append(
            _copy_operation(
                label="snapshot semantic objective config",
                source_path=layout.runtime.semantic_objective_config_original_path,
                target_path=layout.runtime.semantic_objective_config_path,
            )
        )
    for split_layout in layout.runtime.splits:
        split = split_layout.split.value
        operations.append(
            _copy_operation(
                label=f"restore modeling manifest [{split}]",
                source_path=split_layout.drive_manifest_path,
                target_path=split_layout.runtime_manifest_path,
            )
        )
        expected_bytes = _maybe_input_bytes(split_layout.drive_passed_archive_path)
        operations.append(
            ArchiveExtractOperation(
                label=f"restore passed samples [{split}]",
                archive_path=split_layout.drive_passed_archive_path,
                extraction_root=layout.stores.runtime.samples.split_extract_root("passed").path,
                failure_message=f"Failed to restore passed samples for split '{split}'",
                compression_kind="tar_zst",
                strip_components=None,
                members=(),
                expected_input_bytes=expected_bytes,
                progress=_restore_progress_spec(expected_bytes),
            )
        )
    return tuple(operations)


def _copy_operation(*, label: str, source_path: Path, target_path: Path) -> FileCopyOperation:
    expected_bytes = _maybe_input_bytes(source_path)
    return FileCopyOperation(
        label=label,
        source_path=source_path,
        target_path=target_path,
        failure_message=f"Failed to {label}",
        overwrite_policy="atomic_replace",
        expected_input_bytes=expected_bytes,
        progress=_restore_progress_spec(expected_bytes),
    )


def _restore_progress_spec(expected_total: int | None) -> OperationProgressSpec:
    return OperationProgressSpec(
        stage=model_progress_stage(
            stage_id=MODEL_STAGE_RUNTIME_RESTORE,
            label="Restore model workflow runtime artifacts",
            unit="bytes",
            owner_module=__name__,
            split_behavior="per_split",
            operation_kind="restore",
            total_semantics="runtime restore input bytes when known",
            bar_eligible=False,
        ),
        expected_total=expected_total,
        live_owner="shell",
    )


def _maybe_input_bytes(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


__all__ = ["build_model_runtime_plan"]
