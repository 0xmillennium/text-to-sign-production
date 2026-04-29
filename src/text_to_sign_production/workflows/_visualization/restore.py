"""Visualization workflow runtime restore verification and restore specs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from text_to_sign_production.artifacts.layout import DatasetArtifactLayout
from text_to_sign_production.core.files import (
    require_dir,
    require_dir_contains,
    require_file,
    require_non_empty_file,
    verify_output_file,
)
from text_to_sign_production.core.paths import ProjectLayout
from text_to_sign_production.workflows._shared.archives import build_tar_zstd_extract_command
from text_to_sign_production.workflows._visualization.constants import MISSING_SOURCE_VIDEO_MESSAGE
from text_to_sign_production.workflows._visualization.types import (
    SelectedVisualizationSample,
    VisualizationArchiveRestoreSpec,
    VisualizationRawVideoRestoreSpec,
    VisualizationRuntimeInputSummary,
    VisualizationRuntimePlan,
    VisualizationSamplePlan,
    VisualizationWorkflowError,
    VisualizationWorkflowInputError,
)


def verify_visualization_runtime_inputs(
    plan: VisualizationRuntimePlan,
    sample_plan: VisualizationSamplePlan | None = None,
    *,
    verify_raw_video: bool = True,
) -> VisualizationRuntimeInputSummary:
    """Verify restored Visualization manifests and optional selected sample/raw inputs."""

    manifest_paths = _verify_manifest_files(plan)
    selected_sample_path = None
    selected_source_video_path = None
    if sample_plan is not None:
        selected_sample_path = _verify_selected_sample_file(sample_plan)
        if verify_raw_video:
            selected_source_video_path = _verify_selected_raw_video_inputs(sample_plan)
    return VisualizationRuntimeInputSummary(
        manifest_paths=manifest_paths,
        selected_sample_path=selected_sample_path,
        selected_source_video_path=selected_source_video_path,
    )


def _verify_manifest_files(plan: VisualizationRuntimePlan) -> Mapping[str, Path]:
    restored: dict[str, Path] = {}
    try:
        for key, spec in plan.manifest_files.items():
            restored_path = verify_output_file(spec.target_path, label=spec.label)
            observed_size = restored_path.stat().st_size
            if observed_size != spec.expected_input_bytes:
                raise VisualizationWorkflowError(
                    f"{spec.label} byte count mismatch: expected "
                    f"{spec.expected_input_bytes}, observed {observed_size}: {restored_path}"
                )
            restored[key] = restored_path
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        raise VisualizationWorkflowError(str(exc)) from exc
    return restored


def _verify_selected_sample_file(plan: VisualizationSamplePlan) -> Path:
    try:
        return verify_output_file(plan.selected_sample_path, label="Selected processed sample")
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        raise VisualizationWorkflowError(str(exc)) from exc


def _verify_selected_raw_video_inputs(plan: VisualizationSamplePlan) -> Path | None:
    spec = plan.raw_video
    if spec is None:
        return None
    try:
        require_dir(spec.expected_split_root, label=f"Runtime {spec.split} raw BFH split root")
        require_dir(spec.expected_openpose_root, label=f"Runtime {spec.split} openpose_output root")
        require_dir_contains(
            spec.expected_video_root,
            "*.mp4",
            label=f"Runtime {spec.split} source video root",
        )
        return require_file(spec.expected_source_video_path, label="Selected source video")
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, ValueError) as exc:
        raise VisualizationWorkflowError(str(exc)) from exc


def _archive_restore_spec(
    *,
    label: str,
    archive_path: Path,
    extraction_root: Path,
) -> VisualizationArchiveRestoreSpec:
    archive_size = archive_path.stat().st_size
    return VisualizationArchiveRestoreSpec(
        label=label,
        archive_path=archive_path,
        extraction_root=extraction_root,
        expected_input_bytes=archive_size,
        extract=build_tar_zstd_extract_command(
            label=label,
            archive_path=archive_path,
            extraction_root=extraction_root,
            mkdir_path=extraction_root,
            expected_input_bytes=archive_size,
            failure=f"Failed to extract {label} into {extraction_root}: {archive_path}",
        ),
    )


def _raw_video_restore_spec(
    *,
    selected_sample: SelectedVisualizationSample,
    runtime_layout: ProjectLayout,
    drive_layout: ProjectLayout,
) -> VisualizationRawVideoRestoreSpec:
    if selected_sample.source_video_path is None:
        raise VisualizationWorkflowInputError(MISSING_SOURCE_VIDEO_MESSAGE)
    split = selected_sample.record.split
    runtime_artifacts = DatasetArtifactLayout(runtime_layout)
    drive_artifacts = DatasetArtifactLayout(drive_layout)
    archive_path = require_non_empty_file(
        drive_artifacts.raw_bfh_keypoints_archive(split),
        label=f"Drive {split} raw BFH archive",
    )
    expected_split_root = runtime_artifacts.raw_bfh_keypoints_split_root(split)
    expected_openpose_root = runtime_artifacts.raw_bfh_keypoints_openpose_root(split)
    expected_video_root = expected_openpose_root / "video"
    extraction_root = expected_split_root.parent
    archive_size = archive_path.stat().st_size
    label = f"{split} raw BFH archive"
    return VisualizationRawVideoRestoreSpec(
        split=split,
        archive_path=archive_path,
        extraction_root=extraction_root,
        expected_split_root=expected_split_root,
        expected_openpose_root=expected_openpose_root,
        expected_video_root=expected_video_root,
        expected_source_video_path=selected_sample.source_video_path,
        expected_input_bytes=archive_size,
        extract=build_tar_zstd_extract_command(
            label=label,
            archive_path=archive_path,
            extraction_root=extraction_root,
            mkdir_path=expected_split_root,
            expected_input_bytes=archive_size,
            failure=f"Failed to extract {label} into {extraction_root}: {archive_path}",
        ),
    )


__all__ = ["verify_visualization_runtime_inputs"]
