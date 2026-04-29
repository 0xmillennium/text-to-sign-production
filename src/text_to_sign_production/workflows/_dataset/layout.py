"""Dataset workflow layout and path resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from text_to_sign_production.artifacts.layout import DatasetArtifactLayout
from text_to_sign_production.core.paths import (
    ProjectLayout,
    repo_relative_path,
    resolve_repo_path,
)
from text_to_sign_production.data.constants import SPLITS
from text_to_sign_production.data.raw import SplitPaths
from text_to_sign_production.workflows._dataset.constants import (
    DATASET_PUBLISH_DIRNAME,
    DEFAULT_DATASET_FILTER_CONFIG_RELATIVE_PATH,
    DEFAULT_DATASET_QUALITY_CONFIG_RELATIVE_PATH,
    DEFAULT_DATASET_RUN_NAME,
)
from text_to_sign_production.workflows._dataset.reports import _dataset_report_paths
from text_to_sign_production.workflows._dataset.types import (
    DatasetWorkflowConfig,
    DatasetWorkflowError,
)


def _layout_from_config(config: DatasetWorkflowConfig) -> ProjectLayout:
    return ProjectLayout(Path(config.project_root))


def _filter_config_path(config: DatasetWorkflowConfig, layout: ProjectLayout) -> Path:
    configured = config.filter_config_path or DEFAULT_DATASET_FILTER_CONFIG_RELATIVE_PATH
    return resolve_repo_path(configured, repo_root=layout.root)


def _quality_config_path(config: DatasetWorkflowConfig, layout: ProjectLayout) -> Path:
    configured = config.quality_config_path or DEFAULT_DATASET_QUALITY_CONFIG_RELATIVE_PATH
    return resolve_repo_path(configured, repo_root=layout.root)


def _dataset_run_root(layout: ProjectLayout) -> Path:
    return layout.dataset_build_run_root(DEFAULT_DATASET_RUN_NAME)


def _dataset_publish_root(layout: ProjectLayout) -> Path:
    return _dataset_run_root(layout) / DATASET_PUBLISH_DIRNAME


def _split_paths_by_split(
    layout: ProjectLayout,
    splits: tuple[str, ...],
) -> dict[str, SplitPaths]:
    artifact_layout = DatasetArtifactLayout(layout)
    split_paths_by_split: dict[str, SplitPaths] = {}
    for split in splits:
        openpose_root = artifact_layout.raw_bfh_keypoints_openpose_root(split)
        split_paths_by_split[split] = SplitPaths(
            split=split,
            translation_path=artifact_layout.how2sign_translation_file(split),
            keypoints_json_root=openpose_root / "json",
            video_root=openpose_root / "video",
        )
    return split_paths_by_split


def _dataset_output_paths(
    *,
    layout: ProjectLayout,
    splits: tuple[str, ...],
) -> dict[str, Any]:
    artifact_layout = DatasetArtifactLayout(layout)
    return {
        "raw_manifest_paths": {split: artifact_layout.raw_manifest_path(split) for split in splits},
        "normalized_manifest_paths": {
            split: artifact_layout.normalized_manifest_path(split) for split in splits
        },
        "filtered_manifest_paths": {
            split: artifact_layout.filtered_manifest_path(split) for split in splits
        },
        "processed_manifest_paths": {
            split: artifact_layout.processed_manifest_path(split) for split in splits
        },
        "processed_samples_root": artifact_layout.processed_v1_samples_root,
        **_dataset_report_paths(layout=layout, splits=splits),
    }


def _mirrored_drive_data_path(
    path: Path,
    *,
    runtime_layout: ProjectLayout,
    drive_layout: ProjectLayout,
    label: str,
) -> Path:
    relative_path = Path(repo_relative_path(path, repo_root=runtime_layout.root))
    if not relative_path.parts or relative_path.parts[0] != "data":
        raise DatasetWorkflowError(
            f"{label} must be published from the canonical data tree: {path}"
        )
    return drive_layout.root / relative_path


def _split_from_publish_key(key: str) -> str | None:
    key_parts = key.split("_")
    for split in SPLITS:
        if split in key_parts or key == split:
            return split
    return None


__all__: list[str] = []
