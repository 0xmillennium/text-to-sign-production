"""Visualization workflow runtime restore plan construction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from text_to_sign_production.artifacts.layout import DatasetArtifactLayout
from text_to_sign_production.core.files import require_non_empty_file
from text_to_sign_production.core.paths import ProjectLayout
from text_to_sign_production.workflows._shared.transfer import build_byte_progress_copy_command
from text_to_sign_production.workflows._visualization.constants import (
    DEFAULT_VISUALIZATION_RUN_NAME,
    VISUALIZATION_ALLOWED_SPLITS,
)
from text_to_sign_production.workflows._visualization.layout import (
    _render_output_path,
    _render_plan_metadata_path,
    _render_result_metadata_path,
    _visualization_metadata_root,
    _visualization_outputs_root,
    _visualization_publish_root,
    _visualization_run_root,
)
from text_to_sign_production.workflows._visualization.types import (
    VisualizationFileRestoreSpec,
    VisualizationRenderPlan,
    VisualizationRuntimePlan,
    VisualizationSamplePlan,
    VisualizationWorkflowConfig,
    VisualizationWorkflowInputError,
)
from text_to_sign_production.workflows._visualization.validate import (
    _output_policy,
    _render_mode,
    _validate_output_filename,
    _validate_splits,
)
from text_to_sign_production.workflows.commands import CommandSpec


def build_visualization_runtime_plan(
    *,
    project_root: Path | str,
    drive_project_root: Path | str,
    splits: Iterable[str] = VISUALIZATION_ALLOWED_SPLITS,
    render_mode: str = "skeleton_only",
    output_exists_policy: str = "fail",
) -> VisualizationRuntimePlan:
    """Build notebook-visible manifest restore specs without executing commands."""

    runtime_layout = ProjectLayout(Path(project_root))
    drive_layout = ProjectLayout(Path(drive_project_root))
    runtime_artifacts = DatasetArtifactLayout(runtime_layout)
    drive_artifacts = DatasetArtifactLayout(drive_layout)
    run_name = DEFAULT_VISUALIZATION_RUN_NAME
    resolved_splits = _validate_splits(tuple(splits))
    resolved_mode = _render_mode(render_mode)
    resolved_policy = _output_policy(output_exists_policy)

    processed_manifest_files = _visualization_file_restore_specs(
        {
            split: (
                drive_artifacts.processed_manifest_path(split),
                runtime_artifacts.processed_manifest_path(split),
            )
            for split in resolved_splits
        },
        label_prefix="processed manifest",
    )
    filtered_manifest_files = _visualization_file_restore_specs(
        {
            split: (
                drive_artifacts.filtered_manifest_path(split),
                runtime_artifacts.filtered_manifest_path(split),
            )
            for split in resolved_splits
        },
        label_prefix="filtered manifest",
    )
    manifest_files = {
        **{f"processed_{split}": spec for split, spec in processed_manifest_files.items()},
        **{f"filtered_{split}": spec for split, spec in filtered_manifest_files.items()},
    }
    workflow_config = VisualizationWorkflowConfig(
        project_root=runtime_layout.root,
        drive_project_root=drive_layout.root,
        splits=resolved_splits,
        render_mode=resolved_mode,
        output_exists_policy=resolved_policy.value,
        run_name=run_name,
    )
    return VisualizationRuntimePlan(
        project_root=runtime_layout.root,
        drive_project_root=drive_layout.root,
        run_name=run_name,
        run_root=_visualization_run_root(runtime_layout, run_name),
        outputs_root=_visualization_outputs_root(runtime_layout, run_name),
        metadata_root=_visualization_metadata_root(runtime_layout, run_name),
        publish_root=_visualization_publish_root(runtime_layout, run_name),
        splits=resolved_splits,
        render_mode=resolved_mode,
        output_exists_policy=resolved_policy.value,
        processed_manifest_files=processed_manifest_files,
        filtered_manifest_files=filtered_manifest_files,
        manifest_files=manifest_files,
        workflow_config=workflow_config,
    )


def build_visualization_render_plan(
    sample_plan: VisualizationSamplePlan,
    *,
    render_mode: str,
    output_exists_policy: str,
    output_filename: str | None = None,
    fps: float | None = None,
) -> VisualizationRenderPlan:
    """Build a primitive notebook-facing render plan without exposing core config types."""

    resolved_mode = _render_mode(render_mode)
    resolved_policy = _output_policy(output_exists_policy)
    if fps is not None and fps <= 0:
        raise VisualizationWorkflowInputError("fps must be positive when provided.")
    if output_filename is not None:
        _validate_output_filename(output_filename)
    layout = ProjectLayout(sample_plan.project_root)
    return VisualizationRenderPlan(
        project_root=layout.root,
        run_name=sample_plan.run_name,
        run_root=_visualization_run_root(layout, sample_plan.run_name),
        outputs_root=_visualization_outputs_root(layout, sample_plan.run_name),
        metadata_root=_visualization_metadata_root(layout, sample_plan.run_name),
        render_plan_metadata_path=_render_plan_metadata_path(layout, sample_plan.run_name),
        render_result_metadata_path=_render_result_metadata_path(layout, sample_plan.run_name),
        selected_sample=sample_plan.selected_sample,
        sample_id=sample_plan.sample_id,
        split=sample_plan.split,
        mode=resolved_mode,
        output_path=_render_output_path(
            layout=layout,
            run_name=sample_plan.run_name,
            sample_id=sample_plan.sample_id,
            mode=resolved_mode,
            output_filename=output_filename,
        ),
        output_exists_policy=resolved_policy.value,
        fps=fps,
        output_filename=output_filename,
    )


def _visualization_file_restore_specs(
    paths_by_split: Mapping[str, tuple[Path, Path]],
    *,
    label_prefix: str,
) -> dict[str, VisualizationFileRestoreSpec]:
    specs: dict[str, VisualizationFileRestoreSpec] = {}
    for split, (source_path, target_path) in paths_by_split.items():
        label = f"{split} {label_prefix}"
        verified_source_path = require_non_empty_file(source_path, label=f"Drive {label}")
        expected_input_bytes = verified_source_path.stat().st_size
        specs[split] = VisualizationFileRestoreSpec(
            label=label,
            split=split,
            source_path=verified_source_path,
            target_path=target_path,
            expected_input_bytes=expected_input_bytes,
            restore_command=_restore_file_command_spec(
                label=label,
                source_path=verified_source_path,
                target_path=target_path,
                expected_input_bytes=expected_input_bytes,
            ),
        )
    return specs


def _restore_file_command_spec(
    *,
    label: str,
    source_path: Path,
    target_path: Path,
    expected_input_bytes: int,
) -> CommandSpec:
    return build_byte_progress_copy_command(
        label=label,
        source_path=source_path,
        target_path=target_path,
        expected_input_bytes=expected_input_bytes,
        failure=f"Failed to restore {label}: {source_path} -> {target_path}",
    )


__all__ = ["build_visualization_render_plan", "build_visualization_runtime_plan"]
