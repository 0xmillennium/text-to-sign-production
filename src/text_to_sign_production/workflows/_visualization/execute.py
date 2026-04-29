"""Visualization workflow execution orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from text_to_sign_production.core.files import (
    OutputExistsPolicy,
    ensure_dir,
    prepare_output_file,
    verify_output_file,
)
from text_to_sign_production.core.paths import repo_relative_path
from text_to_sign_production.visualization.pose import load_pose_sample
from text_to_sign_production.visualization.skeleton import SkeletonRenderConfig
from text_to_sign_production.visualization.video import (
    CANONICAL_MP4_CODEC,
    render_side_by_side_video,
    render_skeleton_video,
)
from text_to_sign_production.workflows._shared.metadata import write_json_object
from text_to_sign_production.workflows._visualization.constants import MISSING_SOURCE_VIDEO_MESSAGE
from text_to_sign_production.workflows._visualization.types import (
    SelectedVisualizationSample,
    VisualizationRenderMode,
    VisualizationRenderPlan,
    VisualizationWorkflowError,
    VisualizationWorkflowInputError,
    VisualizationWorkflowResult,
)
from text_to_sign_production.workflows._visualization.validate import (
    _output_policy,
    _validate_render_plan,
)


def run_visualization_workflow(
    plan: VisualizationRenderPlan,
) -> VisualizationWorkflowResult:
    """Render a skeleton-only or side-by-side debug MP4."""

    _validate_render_plan(plan)
    ensure_dir(plan.run_root, label="Visualization run root")
    ensure_dir(plan.outputs_root, label="Visualization outputs root")
    ensure_dir(plan.metadata_root, label="Visualization metadata root")
    write_json_object(plan.render_plan_metadata_path, _render_plan_payload(plan))
    output_path, render_metadata = _run_selected_visualization(
        selected_sample=plan.selected_sample,
        mode=plan.mode,
        output_path=plan.output_path,
        fps=plan.fps,
        skeleton_config=SkeletonRenderConfig(),
        output_policy=_output_policy(plan.output_exists_policy),
    )
    completed = VisualizationWorkflowResult(
        project_root=plan.project_root,
        run_name=plan.run_name,
        run_root=plan.run_root,
        metadata_root=plan.metadata_root,
        render_plan_metadata_path=plan.render_plan_metadata_path,
        render_result_metadata_path=plan.render_result_metadata_path,
        selected_sample=plan.selected_sample,
        output_path=output_path,
        render_metadata=render_metadata,
    )
    write_json_object(plan.render_result_metadata_path, _render_result_payload(completed))
    return completed


def _run_selected_visualization(
    *,
    selected_sample: SelectedVisualizationSample,
    mode: VisualizationRenderMode,
    output_path: Path,
    fps: float | None,
    skeleton_config: SkeletonRenderConfig,
    output_policy: OutputExistsPolicy | str,
) -> tuple[Path, Mapping[str, Any]]:
    resolved_output_policy = _output_policy(output_policy)
    existed_before = output_path.exists()
    prepared_output_path = prepare_output_file(
        output_path,
        policy=resolved_output_policy,
        label="Visualization MP4",
    )
    if resolved_output_policy.value == "skip" and existed_before:
        return (
            verify_output_file(prepared_output_path, label="Visualization MP4"),
            {"mode": mode, "codec": CANONICAL_MP4_CODEC, "skipped": True},
        )

    render_fps = _render_fps(fps, selected_sample.timing.fps, selected_sample.record.fps)

    try:
        pose_sample = load_pose_sample(selected_sample.sample_path)
        if mode == "skeleton_only":
            render_metadata = render_skeleton_video(
                pose_sample=pose_sample,
                output_path=prepared_output_path,
                fps=render_fps,
                config=skeleton_config,
                label=selected_sample.record.sample_id,
            )
        else:
            if selected_sample.source_video_path is None:
                raise VisualizationWorkflowInputError(MISSING_SOURCE_VIDEO_MESSAGE)
            render_metadata = render_side_by_side_video(
                source_video_path=selected_sample.source_video_path,
                pose_sample=pose_sample,
                output_path=prepared_output_path,
                fps=render_fps,
                config=skeleton_config,
                label=selected_sample.record.sample_id,
            )
    except VisualizationWorkflowInputError:
        raise
    except Exception as exc:
        raise VisualizationWorkflowError(f"Visualization workflow failed: {exc}") from exc

    return (
        verify_output_file(prepared_output_path, label="Visualization MP4"),
        render_metadata,
    )


def _render_fps(*candidates: float | None) -> float:
    for candidate in candidates:
        if candidate is not None and candidate > 0:
            return float(candidate)
    return 24.0


def _render_plan_payload(plan: VisualizationRenderPlan) -> dict[str, Any]:
    return {
        "schema_version": "t2sp-visualization-render-plan-v1",
        "run_name": plan.run_name,
        "run_root": _portable_path(plan.run_root, project_root=plan.project_root),
        "sample_id": plan.sample_id,
        "split": plan.split,
        "mode": plan.mode,
        "output_path": _portable_path(plan.output_path, project_root=plan.project_root),
        "output_exists_policy": plan.output_exists_policy,
        "fps": plan.fps,
        "selected_sample_path": _portable_path(
            plan.selected_sample.sample_path,
            project_root=plan.project_root,
        ),
        "source_video_path": (
            None
            if plan.selected_sample.source_video_path is None
            else _portable_path(
                plan.selected_sample.source_video_path,
                project_root=plan.project_root,
            )
        ),
    }


def _render_result_payload(result: VisualizationWorkflowResult) -> dict[str, Any]:
    return {
        "schema_version": "t2sp-visualization-render-result-v1",
        "run_name": result.run_name,
        "run_root": _portable_path(result.run_root, project_root=result.project_root),
        "sample_id": result.selected_sample.record.sample_id,
        "split": result.selected_sample.record.split,
        "output_path": _portable_path(result.output_path, project_root=result.project_root),
        "render_metadata": _portable_render_metadata(
            result.render_metadata,
            project_root=result.project_root,
        ),
    }


def _portable_render_metadata(
    metadata: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    payload = dict(metadata)
    for key in ("output_path", "video_path"):
        value = payload.get(key)
        if isinstance(value, str):
            path = Path(value)
            if path.is_absolute() and path.resolve().is_relative_to(project_root):
                payload[key] = repo_relative_path(path, repo_root=project_root)
    return payload


def _portable_path(path: Path, *, project_root: Path) -> str:
    return repo_relative_path(path, repo_root=project_root)


__all__ = ["run_visualization_workflow"]
