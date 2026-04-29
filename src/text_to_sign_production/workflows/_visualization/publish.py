"""Visualization workflow publish plan construction and verification."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.files import require_non_empty_file, verify_output_file
from text_to_sign_production.core.paths import ProjectLayout
from text_to_sign_production.workflows._shared.transfer import build_byte_progress_copy_command
from text_to_sign_production.workflows._visualization.layout import _mirrored_drive_run_path
from text_to_sign_production.workflows._visualization.types import (
    VisualizationPublishPlan,
    VisualizationPublishSpec,
    VisualizationPublishSummary,
    VisualizationWorkflowError,
    VisualizationWorkflowInputError,
    VisualizationWorkflowResult,
)
from text_to_sign_production.workflows.commands import CommandSpec


def build_visualization_publish_plan(
    result: VisualizationWorkflowResult,
    *,
    drive_project_root: Path | str,
    project_root: Path | str | None = None,
) -> VisualizationPublishPlan:
    """Build notebook-visible visualization publish specs without copying files."""

    runtime_layout = ProjectLayout(Path(project_root or result.project_root))
    drive_layout = ProjectLayout(Path(drive_project_root))
    if result.project_root != runtime_layout.root:
        raise VisualizationWorkflowInputError(
            "Published Visualization result does not belong to the requested project root: "
            f"{result.project_root}"
        )
    runtime_run_root = runtime_layout.visual_debug_run_root(result.run_name)
    if result.run_root != runtime_run_root:
        raise VisualizationWorkflowInputError(
            "Published Visualization result run root does not match the canonical run root: "
            f"{result.run_root}"
        )
    drive_run_root = drive_layout.visual_debug_run_root(result.run_name)
    output_spec = _visualization_publish_spec(
        label="visualization output",
        source_path=result.output_path,
        target_path=_mirrored_drive_run_path(
            result.output_path,
            runtime_layout=runtime_layout,
            drive_layout=drive_layout,
            run_name=result.run_name,
            label="visualization output",
        ),
    )
    metadata_specs = tuple(
        _visualization_publish_spec(
            label=label,
            source_path=source_path,
            target_path=_mirrored_drive_run_path(
                source_path,
                runtime_layout=runtime_layout,
                drive_layout=drive_layout,
                run_name=result.run_name,
                label=label,
            ),
        )
        for label, source_path in (
            ("visualization render plan metadata", result.render_plan_metadata_path),
            ("visualization render result metadata", result.render_result_metadata_path),
        )
    )
    return VisualizationPublishPlan(
        run_name=result.run_name,
        runtime_project_root=runtime_layout.root,
        drive_project_root=drive_layout.root,
        runtime_run_root=runtime_run_root,
        drive_run_root=drive_run_root,
        output_spec=output_spec,
        metadata_specs=metadata_specs,
    )


def _verify_publish_plan_outputs(plan: VisualizationPublishPlan) -> VisualizationPublishSummary:
    output_target = _verify_publish_spec(plan.output_spec)
    metadata_paths: dict[str, Path] = {}
    metadata_sizes: dict[str, int] = {}
    for spec in plan.metadata_specs:
        target_path = _verify_publish_spec(spec)
        metadata_paths[spec.label] = target_path
        metadata_sizes[spec.label] = target_path.stat().st_size
    return VisualizationPublishSummary(
        output_source_path=plan.output_spec.source_path,
        output_target_path=output_target,
        output_size=output_target.stat().st_size,
        metadata_paths=metadata_paths,
        metadata_sizes=metadata_sizes,
    )


def _visualization_publish_spec(
    *,
    label: str,
    source_path: Path,
    target_path: Path,
) -> VisualizationPublishSpec:
    verified_source_path = require_non_empty_file(source_path, label=f"Runtime {label}")
    expected_input_bytes = verified_source_path.stat().st_size
    return VisualizationPublishSpec(
        label=label,
        source_path=verified_source_path,
        target_path=target_path,
        expected_input_bytes=expected_input_bytes,
        publish_command=_publish_file_command_spec(
            label=label,
            source_path=verified_source_path,
            target_path=target_path,
            expected_input_bytes=expected_input_bytes,
        ),
    )


def _publish_file_command_spec(
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
        failure=f"Failed to publish {label}: {source_path} -> {target_path}",
    )


def _verify_publish_spec(spec: VisualizationPublishSpec) -> Path:
    try:
        target_path = verify_output_file(spec.target_path, label=spec.label)
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        raise VisualizationWorkflowError(str(exc)) from exc
    observed_size = target_path.stat().st_size
    if observed_size != spec.expected_input_bytes:
        raise VisualizationWorkflowError(
            f"{spec.label} byte count mismatch: expected "
            f"{spec.expected_input_bytes}, observed {observed_size}: {target_path}"
        )
    return target_path


__all__ = ["build_visualization_publish_plan"]
