"""Visualization workflow sample selection and sample restore planning."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.artifacts.index import build_sample_index
from text_to_sign_production.artifacts.layout import DatasetArtifactLayout
from text_to_sign_production.artifacts.refs import ProcessedSampleDataError
from text_to_sign_production.artifacts.select import select_sample
from text_to_sign_production.core.files import require_non_empty_file
from text_to_sign_production.core.paths import ProjectLayout
from text_to_sign_production.workflows._visualization.layout import _resolve_source_video_path
from text_to_sign_production.workflows._visualization.restore import (
    _archive_restore_spec,
    _raw_video_restore_spec,
)
from text_to_sign_production.workflows._visualization.types import (
    SelectedVisualizationSample,
    VisualizationRuntimePlan,
    VisualizationSamplePlan,
    VisualizationSampleSummary,
    VisualizationWorkflowInputError,
    _SampleSelectionConfig,
)
from text_to_sign_production.workflows._visualization.validate import (
    _render_mode,
    _selection_mode,
    _validate_selection_modes,
)


def build_visualization_sample_plan(
    runtime_plan: VisualizationRuntimePlan,
    *,
    selection_mode: str,
    sample_id: str | None = None,
    random_seed: int | None = None,
) -> VisualizationSamplePlan:
    """Select from restored manifests and build selected sample/raw restore specs."""

    resolved_selection_mode = _selection_mode(selection_mode)
    selected_sample_id = sample_id.strip() if sample_id is not None else None
    if resolved_selection_mode == "sample_id":
        if not selected_sample_id:
            raise VisualizationWorkflowInputError("sample_id is required for sample_id selection.")
    elif selected_sample_id:
        raise VisualizationWorkflowInputError("sample_id must be blank for random selection.")

    selected_sample = _select_visualization_sample(
        _SampleSelectionConfig(
            project_root=runtime_plan.project_root,
            splits=runtime_plan.splits,
            sample_id=selected_sample_id if resolved_selection_mode == "sample_id" else None,
            random_selection=resolved_selection_mode == "random",
            seed=random_seed,
            require_sample_files=False,
        )
    )
    runtime_layout = ProjectLayout(runtime_plan.project_root)
    drive_layout = ProjectLayout(runtime_plan.drive_project_root)
    drive_artifacts = DatasetArtifactLayout(drive_layout)
    sample_archive_path = require_non_empty_file(
        drive_artifacts.processed_sample_archive(selected_sample.record.split),
        label=f"Drive {selected_sample.record.split} processed sample archive",
    )
    sample_archive = _archive_restore_spec(
        label=f"{selected_sample.record.split} processed sample archive",
        archive_path=sample_archive_path,
        extraction_root=runtime_layout.root,
    )
    raw_video = (
        _raw_video_restore_spec(
            selected_sample=selected_sample,
            runtime_layout=runtime_layout,
            drive_layout=drive_layout,
        )
        if runtime_plan.render_mode == "side_by_side"
        else None
    )
    return VisualizationSamplePlan(
        project_root=runtime_plan.project_root,
        drive_project_root=runtime_plan.drive_project_root,
        run_name=runtime_plan.run_name,
        run_root=runtime_plan.run_root,
        selected_sample=selected_sample,
        sample_id=selected_sample.record.sample_id,
        split=selected_sample.record.split,
        sample_archive=sample_archive,
        selected_sample_path=selected_sample.sample_path,
        raw_video=raw_video,
    )


def summarize_visualization_sample(
    sample_plan: VisualizationSamplePlan,
    *,
    selection_mode: str,
    random_seed: int | None = None,
    render_mode: str,
) -> VisualizationSampleSummary:
    """Summarize selected sample context for notebook display without exposing internals."""

    resolved_selection_mode = _selection_mode(selection_mode)
    resolved_render_mode = _render_mode(render_mode)
    raw_video = sample_plan.raw_video
    return VisualizationSampleSummary(
        sample_id=sample_plan.sample_id,
        split=sample_plan.split,
        text=sample_plan.selected_sample.record.text or None,
        selection_mode=resolved_selection_mode,
        random_seed=random_seed if resolved_selection_mode == "random" else None,
        sample_path=sample_plan.selected_sample_path,
        source_video_path=sample_plan.selected_sample.source_video_path,
        render_mode=resolved_render_mode,
        processed_archive_path=sample_plan.sample_archive.archive_path,
        processed_archive_size=sample_plan.sample_archive.expected_input_bytes,
        raw_archive_path=raw_video.archive_path if raw_video is not None else None,
        raw_archive_size=raw_video.expected_input_bytes if raw_video is not None else None,
        raw_restore_required=raw_video is not None,
    )


def _select_visualization_sample(
    config: _SampleSelectionConfig,
) -> SelectedVisualizationSample:
    """Select one processed sample by exact id or deterministic random choice."""

    _validate_selection_modes(config)
    layout = ProjectLayout(Path(config.project_root))
    artifact_layout = DatasetArtifactLayout(layout)
    try:
        index = build_sample_index(
            layout.data_root,
            splits=config.splits,
            require_sample_files=config.require_sample_files,
            processed_manifests_root=artifact_layout.processed_v1_manifests_root,
            processed_samples_root=artifact_layout.processed_v1_samples_root,
            filtered_manifests_root=artifact_layout.filtered_manifests_root,
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


__all__ = ["build_visualization_sample_plan", "summarize_visualization_sample"]
