"""Validate model runtime plans before executing restoration."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.workflows.foundation.execution import (
    ArchiveExtractOperation,
    FileCopyOperation,
)
from text_to_sign_production.workflows.model.contracts import (
    ModelRuntimePlan,
    ModelWorkflowConfig,
    ModelWorkflowInputError,
    ModelWorkflowInvariantError,
)
from text_to_sign_production.workflows.model.layout import ModelLayout, required_model_splits
from text_to_sign_production.modeling.research import ObjectiveKey
from text_to_sign_production.workflows.model.request import (
    build_model_run_request_from_layout,
)


def validate_model_runtime_plan(
    config: ModelWorkflowConfig,
    layout: ModelLayout,
    plan: ModelRuntimePlan,
) -> None:
    """Raise when a runtime plan no longer agrees with its layout and config."""

    if not isinstance(config, ModelWorkflowConfig):
        raise ModelWorkflowInputError("config must be a ModelWorkflowConfig")
    if not isinstance(layout, ModelLayout):
        raise ModelWorkflowInputError("layout must be a ModelLayout")
    if not isinstance(plan, ModelRuntimePlan):
        raise ModelWorkflowInputError("plan must be a ModelRuntimePlan")
    if layout.config != config:
        raise ModelWorkflowInvariantError("layout.config must match config")

    expected_splits = required_model_splits(config)
    observed_splits = (
        layout.runtime.required_splits,
        tuple(split_layout.split for split_layout in layout.runtime.splits),
        tuple(split_input.split for split_input in plan.execution_inputs.split_inputs),
    )
    if any(splits != expected_splits for splits in observed_splits):
        raise ModelWorkflowInvariantError("model runtime split sequence is misaligned")

    expected_request = build_model_run_request_from_layout(config, layout)
    if plan.execution_inputs.request != expected_request:
        raise ModelWorkflowInvariantError("model run request is misaligned with config")
    if plan.execution_inputs.runtime_topology != layout.stores.runtime:
        raise ModelWorkflowInvariantError("runtime topology is misaligned")
    if plan.execution_inputs.model_run_root != layout.outputs.model_run_root:
        raise ModelWorkflowInvariantError("model run root is misaligned")
    if plan.execution_inputs.report_root != layout.reports.root:
        raise ModelWorkflowInvariantError("model report root is misaligned")

    operation_offset = 0
    if config.model_config_relpath is not None:
        if layout.runtime.model_config_original_path is None or layout.runtime.model_config_path is None:
            raise ModelWorkflowInvariantError("model config layout is incomplete")
        if plan.execution_inputs.model_config_provenance is None:
            raise ModelWorkflowInvariantError("model config provenance is missing")
        config_operation = plan.restore_operations[0] if plan.restore_operations else None
        if not isinstance(config_operation, FileCopyOperation):
            raise ModelWorkflowInvariantError("first operation must snapshot model config")
        if config_operation.source_path != layout.runtime.model_config_original_path:
            raise ModelWorkflowInvariantError("model config source path is misaligned")
        if config_operation.target_path != layout.runtime.model_config_path:
            raise ModelWorkflowInvariantError("model config target path is misaligned")
        _require_under(config_operation.source_path, config.project_root, "model config source")
        _require_under(config_operation.target_path, layout.stores.runtime.repo_root, "model config target")
        _require_distinct(config_operation.source_path, config_operation.target_path)
        operation_offset = 1
    elif plan.execution_inputs.model_config_path is not None or plan.execution_inputs.model_config_provenance is not None:
        raise ModelWorkflowInvariantError("unexpected model config execution input")

    semantic_requested = ObjectiveKey.SEMANTIC_CONSISTENCY in config.auxiliary_objectives
    if semantic_requested:
        source = layout.runtime.semantic_objective_config_original_path
        target = layout.runtime.semantic_objective_config_path
        if source is None or target is None:
            raise ModelWorkflowInvariantError("semantic objective config layout is incomplete")
        if plan.execution_inputs.semantic_objective_config_path != target:
            raise ModelWorkflowInvariantError("semantic objective config execution path is misaligned")
        if plan.execution_inputs.semantic_objective_config_provenance is None:
            raise ModelWorkflowInvariantError("semantic objective config provenance is missing")
        operation = (
            plan.restore_operations[operation_offset]
            if len(plan.restore_operations) > operation_offset
            else None
        )
        if not isinstance(operation, FileCopyOperation):
            raise ModelWorkflowInvariantError("runtime plan must snapshot semantic objective config")
        if operation.source_path != source or operation.target_path != target:
            raise ModelWorkflowInvariantError("semantic objective config snapshot paths are misaligned")
        _require_under(source, config.project_root, "semantic objective config source")
        _require_under(target, layout.stores.runtime.repo_root, "semantic objective config target")
        _require_distinct(source, target)
        operation_offset += 1
    elif (
        plan.execution_inputs.semantic_objective_config_path is not None
        or plan.execution_inputs.semantic_objective_config_provenance is not None
    ):
        raise ModelWorkflowInvariantError("unexpected semantic objective config execution input")

    expected_operation_count = operation_offset + (2 * len(expected_splits))
    if len(plan.restore_operations) != expected_operation_count:
        raise ModelWorkflowInvariantError("model runtime restore operation count is misaligned")

    for index, (split_layout, split_input) in enumerate(
        zip(layout.runtime.splits, plan.execution_inputs.split_inputs, strict=True)
    ):
        if split_input.manifest_path != split_layout.runtime_manifest_path:
            raise ModelWorkflowInvariantError("modeling manifest execution input is misaligned")
        if (
            split_input.passed_samples_split_root
            != split_layout.runtime_passed_samples_split_root
        ):
            raise ModelWorkflowInvariantError("passed samples execution input is misaligned")
        copy_operation = plan.restore_operations[operation_offset + (index * 2)]
        extract_operation = plan.restore_operations[operation_offset + (index * 2) + 1]
        if not isinstance(copy_operation, FileCopyOperation):
            raise ModelWorkflowInvariantError("modeling manifest restore must be file copy")
        if not isinstance(extract_operation, ArchiveExtractOperation):
            raise ModelWorkflowInvariantError("passed sample restore must be archive extract")
        if copy_operation.source_path != split_layout.drive_manifest_path:
            raise ModelWorkflowInvariantError("modeling manifest source path is misaligned")
        if copy_operation.target_path != split_layout.runtime_manifest_path:
            raise ModelWorkflowInvariantError("modeling manifest target path is misaligned")
        if extract_operation.archive_path != split_layout.drive_passed_archive_path:
            raise ModelWorkflowInvariantError("passed archive source path is misaligned")
        if (
            extract_operation.extraction_root
            != layout.stores.runtime.samples.split_extract_root("passed").path
        ):
            raise ModelWorkflowInvariantError("passed archive extraction root is misaligned")
        _require_under(copy_operation.source_path, layout.stores.drive.repo_root, "manifest source")
        _require_under(extract_operation.archive_path, layout.stores.drive.repo_root, "archive source")
        _require_under(copy_operation.target_path, layout.stores.runtime.repo_root, "manifest target")
        _require_under(extract_operation.extraction_root, layout.stores.runtime.repo_root, "archive target")
        _require_distinct(copy_operation.source_path, copy_operation.target_path)


def _require_under(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise ModelWorkflowInvariantError(f"{label} is outside its expected repository root") from exc


def _require_distinct(source: Path, target: Path) -> None:
    if source.resolve(strict=False) == target.resolve(strict=False):
        raise ModelWorkflowInvariantError("runtime restore source and target must differ")


__all__ = ["validate_model_runtime_plan"]
