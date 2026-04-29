"""Visualization workflow layout and path construction helpers."""

from __future__ import annotations

import re
from pathlib import Path

from text_to_sign_production.artifacts.layout import DatasetArtifactLayout
from text_to_sign_production.artifacts.refs import ManifestSampleRecord, TimingMetadata
from text_to_sign_production.core.paths import (
    ProjectLayout,
    repo_relative_path,
    resolve_manifest_path,
)
from text_to_sign_production.workflows._visualization.constants import (
    DEFAULT_VISUALIZATION_OUTPUT_FILENAME,
    VISUALIZATION_METADATA_DIRNAME,
    VISUALIZATION_OUTPUTS_DIRNAME,
    VISUALIZATION_PUBLISH_DIRNAME,
    VISUALIZATION_RENDER_PLAN_FILENAME,
    VISUALIZATION_RENDER_RESULT_FILENAME,
)
from text_to_sign_production.workflows._visualization.types import (
    VisualizationRenderMode,
    VisualizationWorkflowError,
    VisualizationWorkflowInputError,
)


def _visualization_run_root(layout: ProjectLayout, run_name: str) -> Path:
    return layout.visual_debug_run_root(run_name)


def _visualization_outputs_root(layout: ProjectLayout, run_name: str) -> Path:
    return _visualization_run_root(layout, run_name) / VISUALIZATION_OUTPUTS_DIRNAME


def _visualization_metadata_root(layout: ProjectLayout, run_name: str) -> Path:
    return _visualization_run_root(layout, run_name) / VISUALIZATION_METADATA_DIRNAME


def _visualization_publish_root(layout: ProjectLayout, run_name: str) -> Path:
    return _visualization_run_root(layout, run_name) / VISUALIZATION_PUBLISH_DIRNAME


def _render_plan_metadata_path(layout: ProjectLayout, run_name: str) -> Path:
    return _visualization_metadata_root(layout, run_name) / VISUALIZATION_RENDER_PLAN_FILENAME


def _render_result_metadata_path(layout: ProjectLayout, run_name: str) -> Path:
    return _visualization_metadata_root(layout, run_name) / VISUALIZATION_RENDER_RESULT_FILENAME


def _render_output_path(
    *,
    layout: ProjectLayout,
    run_name: str,
    sample_id: str,
    mode: VisualizationRenderMode,
    output_filename: str | None,
) -> Path:
    return _visualization_outputs_root(layout, run_name) / _render_output_filename(
        sample_id=sample_id,
        mode=mode,
        output_filename=output_filename,
    )


def _render_output_filename(
    *,
    sample_id: str,
    mode: VisualizationRenderMode,
    output_filename: str | None,
) -> str:
    filename = _validated_render_output_filename(
        output_filename or DEFAULT_VISUALIZATION_OUTPUT_FILENAME
    )
    if filename == DEFAULT_VISUALIZATION_OUTPUT_FILENAME:
        return filename
    return filename or f"{_sanitize_filename_token(sample_id)}__{mode}.mp4"


def _validated_render_output_filename(value: str) -> str:
    filename = value.strip()
    if not filename:
        raise VisualizationWorkflowInputError("output_filename must not be blank.")
    candidate = Path(filename)
    if (
        candidate.is_absolute()
        or candidate.name != filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or ".." in candidate.parts
    ):
        raise VisualizationWorkflowInputError("output_filename must be a safe single filename.")
    if candidate.suffix.lower() != ".mp4":
        raise VisualizationWorkflowInputError("output_filename must end in .mp4.")
    return filename


def _resolve_source_video_path(
    *,
    record: ManifestSampleRecord,
    timing: TimingMetadata,
    layout: ProjectLayout,
) -> Path | None:
    source_video_value = record.source_video_path or timing.source_video_path
    if source_video_value is None:
        return _canonical_source_video_path(record=record, layout=layout)
    try:
        return resolve_manifest_path(source_video_value, data_root=layout.data_root)
    except ValueError as exc:
        raise VisualizationWorkflowInputError(
            f"Invalid source_video_path for sample {record.sample_id!r}: {exc}"
        ) from exc


def _canonical_source_video_path(
    *,
    record: ManifestSampleRecord,
    layout: ProjectLayout,
) -> Path:
    artifact_layout = DatasetArtifactLayout(layout)
    return (
        artifact_layout.raw_bfh_keypoints_openpose_root(record.split)
        / "video"
        / f"{record.sample_id}.mp4"
    )


def _mirrored_drive_run_path(
    path: Path,
    *,
    runtime_layout: ProjectLayout,
    drive_layout: ProjectLayout,
    run_name: str,
    label: str,
) -> Path:
    relative_path = Path(repo_relative_path(path, repo_root=runtime_layout.root))
    expected_prefix = Path("runs") / "visualization" / "visual-debug" / run_name
    if not relative_path.is_relative_to(expected_prefix):
        raise VisualizationWorkflowError(
            f"{label} must be published from the canonical Visualization run tree: {path}"
        )
    return drive_layout.root / relative_path


def _sanitize_filename_token(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "sample"


__all__: list[str] = []
