"""Notebook-facing visualization workflow over canonical processed samples."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from text_to_sign_production.core.files import (
    OutputExistsPolicy,
    prepare_output_file,
    require_file,
    verify_output_file,
)
from text_to_sign_production.core.paths import (
    DEFAULT_REPO_ROOT,
    ProjectLayout,
    resolve_manifest_path,
)
from text_to_sign_production.data.processed_samples import (
    ManifestSampleRecord,
    ProcessedSampleDataError,
    TimingMetadata,
    build_sample_index,
    select_sample,
)
from text_to_sign_production.visualization.pose import load_pose_sample
from text_to_sign_production.visualization.skeleton import SkeletonRenderConfig
from text_to_sign_production.visualization.video import (
    render_side_by_side_video,
    render_skeleton_video,
)

VisualizationRenderMode = Literal["skeleton_only", "side_by_side"]
_MISSING_SOURCE_VIDEO_MESSAGE = (
    "Source video is missing. Extract the required raw BFH archive in the notebook/runtime first."
)
_MISSING_SAMPLE_FILE_MESSAGE = (
    "Selected processed sample file is missing. "
    "Extract dataset_build_samples_{split}.tar.zst in the notebook/runtime first."
)


@dataclass(frozen=True, slots=True)
class VisualizationSelectionConfig:
    project_root: Path | str | None = None
    data_root: Path | str | None = None
    splits: tuple[str, ...] | None = None
    sample_id: str | None = None
    random_selection: bool = False
    seed: int | None = None
    require_sample_files: bool = False


@dataclass(frozen=True, slots=True)
class SelectedVisualizationSample:
    record: ManifestSampleRecord
    timing: TimingMetadata
    sample_path: Path
    source_video_path: Path | None


@dataclass(frozen=True, slots=True)
class VisualizationWorkflowConfig:
    selection: VisualizationSelectionConfig
    mode: VisualizationRenderMode = "skeleton_only"
    output_filename: str | None = None
    fps: float | None = None
    skeleton_config: SkeletonRenderConfig = field(default_factory=SkeletonRenderConfig)
    output_policy: OutputExistsPolicy = OutputExistsPolicy.FAIL
    output_dir: Path | str | None = None


@dataclass(frozen=True, slots=True)
class VisualizationWorkflowResult:
    selected_sample: SelectedVisualizationSample
    output_path: Path
    render_metadata: Mapping[str, Any]


class VisualizationWorkflowError(RuntimeError):
    """Raised when visualization workflow orchestration fails."""


class VisualizationWorkflowInputError(ValueError):
    """Raised when visualization workflow inputs are missing or invalid."""


def select_visualization_sample(
    config: VisualizationSelectionConfig,
) -> SelectedVisualizationSample:
    """Select one processed sample by exact id or deterministic random choice."""

    _validate_selection_modes(config)
    layout = _layout_from_selection(config)
    try:
        index = build_sample_index(
            layout.data_root,
            splits=config.splits,
            require_sample_files=config.require_sample_files,
            processed_manifests_root=layout.processed_v1_manifests_root,
            processed_samples_root=layout.processed_v1_samples_root,
            filtered_manifests_root=layout.filtered_manifests_root,
            normalized_manifests_root=layout.normalized_manifests_root,
            raw_manifests_root=layout.raw_manifests_root,
        )
        record = select_sample(
            index,
            sample_id=config.sample_id,
            random_selection=config.random_selection,
            seed=config.seed,
        )
        timing = index.timing_for(record)
    except (FileNotFoundError, ProcessedSampleDataError) as exc:
        raise VisualizationWorkflowInputError(str(exc)) from exc

    source_video_path = _resolve_source_video_path(
        record=record,
        timing=timing,
        layout=layout,
    )
    return SelectedVisualizationSample(
        record=record,
        timing=timing,
        sample_path=record.sample_path,
        source_video_path=source_video_path,
    )


def validate_visualization_inputs(config: VisualizationWorkflowConfig) -> None:
    """Validate notebook/runtime visualization controls without rendering."""

    selected_sample = _select_validated_sample(config)
    _resolve_output_path(config, selected_sample)


def run_visualization_workflow(
    config: VisualizationWorkflowConfig,
) -> VisualizationWorkflowResult:
    """Select a sample and render a skeleton-only or side-by-side debug MP4."""

    selected_sample = _select_validated_sample(config)
    output_path = _resolve_output_path(config, selected_sample)
    existed_before = output_path.exists()
    output_path = prepare_output_file(
        output_path,
        policy=config.output_policy,
        label="Visualization MP4",
    )
    if config.output_policy == OutputExistsPolicy.SKIP and existed_before:
        return VisualizationWorkflowResult(
            selected_sample=selected_sample,
            output_path=verify_output_file(output_path, label="Visualization MP4"),
            render_metadata={"mode": config.mode, "skipped": True},
        )

    fps = _render_fps(config.fps, selected_sample.timing.fps, selected_sample.record.fps)

    try:
        pose_sample = load_pose_sample(selected_sample.sample_path)
        if config.mode == "skeleton_only":
            render_metadata = render_skeleton_video(
                pose_sample=pose_sample,
                output_path=output_path,
                fps=fps,
                config=config.skeleton_config,
                label=selected_sample.record.sample_id,
            )
        else:
            if selected_sample.source_video_path is None:
                raise VisualizationWorkflowInputError(_MISSING_SOURCE_VIDEO_MESSAGE)
            render_metadata = render_side_by_side_video(
                source_video_path=selected_sample.source_video_path,
                pose_sample=pose_sample,
                output_path=output_path,
                fps=fps,
                config=config.skeleton_config,
                label=selected_sample.record.sample_id,
            )
    except VisualizationWorkflowInputError:
        raise
    except Exception as exc:
        raise VisualizationWorkflowError(f"Visualization workflow failed: {exc}") from exc

    return VisualizationWorkflowResult(
        selected_sample=selected_sample,
        output_path=verify_output_file(output_path, label="Visualization MP4"),
        render_metadata=render_metadata,
    )


def _select_validated_sample(config: VisualizationWorkflowConfig) -> SelectedVisualizationSample:
    _validate_workflow_config(config)
    selected_sample = select_visualization_sample(config.selection)
    _validate_selected_runtime_inputs(selected_sample, mode=config.mode)
    return selected_sample


def _layout_from_selection(config: VisualizationSelectionConfig) -> ProjectLayout:
    if config.project_root is not None and config.data_root is not None:
        project_layout = ProjectLayout(Path(config.project_root))
        data_root = Path(config.data_root).expanduser().resolve()
        if data_root != project_layout.data_root:
            raise VisualizationWorkflowInputError(
                f"project_root and data_root disagree: {project_layout.data_root} != {data_root}"
            )
        return project_layout
    if config.project_root is not None:
        return ProjectLayout(Path(config.project_root))
    if config.data_root is not None:
        data_root = Path(config.data_root).expanduser().resolve()
        if data_root.name != "data":
            raise VisualizationWorkflowInputError(
                f"data_root must point at a project data directory: {data_root}"
            )
        return ProjectLayout(data_root.parent)
    return ProjectLayout(DEFAULT_REPO_ROOT)


def _validate_workflow_config(config: VisualizationWorkflowConfig) -> None:
    _validate_selection_modes(config.selection)
    if config.mode not in ("skeleton_only", "side_by_side"):
        raise VisualizationWorkflowInputError(f"Unsupported visualization mode: {config.mode}")
    if config.fps is not None and config.fps <= 0:
        raise VisualizationWorkflowInputError("fps must be positive when provided.")
    if config.output_filename is not None:
        output_filename = config.output_filename.strip()
        if not output_filename:
            raise VisualizationWorkflowInputError("output_filename must not be blank.")
        if Path(output_filename).name != output_filename:
            raise VisualizationWorkflowInputError("output_filename must be a filename, not a path.")
    if config.output_dir is not None:
        output_dir = Path(config.output_dir).expanduser()
        if output_dir.exists() and not output_dir.is_dir():
            raise VisualizationWorkflowInputError(f"output_dir is not a directory: {output_dir}")


def _validate_selected_runtime_inputs(
    selected_sample: SelectedVisualizationSample,
    *,
    mode: VisualizationRenderMode,
) -> None:
    try:
        require_file(selected_sample.sample_path, label="Selected processed sample file")
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        raise VisualizationWorkflowInputError(
            _MISSING_SAMPLE_FILE_MESSAGE.format(split=selected_sample.record.split)
            + f" Missing path: {selected_sample.sample_path}"
        ) from exc
    if mode == "side_by_side":
        if selected_sample.source_video_path is None:
            raise VisualizationWorkflowInputError(_MISSING_SOURCE_VIDEO_MESSAGE)
        try:
            require_file(selected_sample.source_video_path, label="Source video")
        except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
            raise VisualizationWorkflowInputError(_MISSING_SOURCE_VIDEO_MESSAGE) from exc


def _validate_selection_modes(config: VisualizationSelectionConfig) -> None:
    sample_id = config.sample_id
    has_sample_id = sample_id is not None
    if sample_id is not None and not sample_id.strip():
        raise VisualizationWorkflowInputError("sample_id must not be blank when provided.")
    if has_sample_id == config.random_selection:
        raise VisualizationWorkflowInputError(
            "Exactly one visualization selection mode is required: sample_id or random_selection."
        )


def _resolve_source_video_path(
    *,
    record: ManifestSampleRecord,
    timing: TimingMetadata,
    layout: ProjectLayout,
) -> Path | None:
    source_video_value = record.source_video_path or timing.source_video_path
    if source_video_value is None:
        return None
    try:
        return resolve_manifest_path(source_video_value, data_root=layout.data_root)
    except ValueError as exc:
        raise VisualizationWorkflowInputError(
            f"Invalid source_video_path for sample {record.sample_id!r}: {exc}"
        ) from exc


def _resolve_output_path(
    config: VisualizationWorkflowConfig,
    selected_sample: SelectedVisualizationSample,
) -> Path:
    layout = _layout_from_selection(config.selection)
    output_dir = (
        layout.visualization_artifacts_root
        if config.output_dir is None
        else Path(config.output_dir).expanduser().resolve()
    )
    filename = config.output_filename
    if filename is None:
        sample_id = _sanitize_filename_token(selected_sample.record.sample_id)
        filename = f"{sample_id}__{config.mode}.mp4"
    elif Path(filename).suffix.lower() != ".mp4":
        filename = f"{filename}.mp4"
    return output_dir / filename


def _render_fps(*candidates: float | None) -> float:
    for candidate in candidates:
        if candidate is not None and candidate > 0:
            return float(candidate)
    return 24.0


def _sanitize_filename_token(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "sample"


__all__ = [
    "SelectedVisualizationSample",
    "VisualizationRenderMode",
    "VisualizationSelectionConfig",
    "VisualizationWorkflowConfig",
    "VisualizationWorkflowError",
    "VisualizationWorkflowInputError",
    "VisualizationWorkflowResult",
    "run_visualization_workflow",
    "select_visualization_sample",
    "validate_visualization_inputs",
]
