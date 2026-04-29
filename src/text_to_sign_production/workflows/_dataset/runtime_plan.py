"""Dataset workflow runtime restore plan construction."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from text_to_sign_production.artifacts.layout import DatasetArtifactLayout
from text_to_sign_production.core.files import require_non_empty_file
from text_to_sign_production.core.paths import ProjectLayout
from text_to_sign_production.data.constants import SPLITS
from text_to_sign_production.workflows._dataset.types import (
    DatasetFileTransferSpec,
    DatasetRawArchiveRestoreSpec,
    DatasetRuntimePlan,
    DatasetWorkflowConfig,
)
from text_to_sign_production.workflows._dataset.validate import _validate_splits
from text_to_sign_production.workflows._shared.archives import build_tar_zstd_extract_command
from text_to_sign_production.workflows._shared.transfer import build_byte_progress_copy_command
from text_to_sign_production.workflows.commands import CommandSpec


def build_dataset_runtime_plan(
    *,
    project_root: Path | str,
    drive_project_root: Path | str,
    splits: Iterable[str] = SPLITS,
    filter_config_path: Path | str | None = None,
    validate_outputs: bool = True,
    fail_on_validation_errors: bool = True,
) -> DatasetRuntimePlan:
    """Build notebook-visible raw input transfer/restore specs without executing commands."""

    runtime_layout = ProjectLayout(Path(project_root))
    drive_layout = ProjectLayout(Path(drive_project_root))
    runtime_artifacts = DatasetArtifactLayout(runtime_layout)
    drive_artifacts = DatasetArtifactLayout(drive_layout)
    resolved_splits = _validate_splits(tuple(splits))

    raw_translation_specs: dict[str, DatasetFileTransferSpec] = {}
    raw_bfh_archive_specs: dict[str, DatasetRawArchiveRestoreSpec] = {}
    for split in resolved_splits:
        source_translation = require_non_empty_file(
            drive_artifacts.how2sign_translation_file(split),
            label=f"Drive {split} translation CSV",
        )
        target_translation = runtime_artifacts.how2sign_translation_file(split)
        translation_size = source_translation.stat().st_size
        translation_label = f"{split} raw translation CSV"
        raw_translation_specs[split] = DatasetFileTransferSpec(
            split=split,
            label=translation_label,
            source_path=source_translation,
            target_path=target_translation,
            expected_input_bytes=translation_size,
            copy_command=_copy_file_command_spec(
                label=translation_label,
                source_path=source_translation,
                target_path=target_translation,
                expected_input_bytes=translation_size,
            ),
        )

        archive_path = require_non_empty_file(
            drive_artifacts.raw_bfh_keypoints_archive(split),
            label=f"Drive {split} raw BFH archive",
        )
        expected_split_root = runtime_artifacts.raw_bfh_keypoints_split_root(split)
        expected_openpose_root = runtime_artifacts.raw_bfh_keypoints_openpose_root(split)
        extraction_root = expected_split_root.parent
        archive_size = archive_path.stat().st_size
        archive_label = f"{split} raw BFH archive"
        raw_bfh_archive_specs[split] = DatasetRawArchiveRestoreSpec(
            split=split,
            label=archive_label,
            archive_path=archive_path,
            extraction_root=extraction_root,
            expected_split_root=expected_split_root,
            expected_openpose_root=expected_openpose_root,
            expected_json_root=expected_openpose_root / "json",
            expected_video_root=expected_openpose_root / "video",
            expected_input_bytes=archive_size,
            extract_command=_extract_raw_archive_command_spec(
                label=archive_label,
                archive_path=archive_path,
                extraction_root=extraction_root,
                expected_split_root=expected_split_root,
                expected_input_bytes=archive_size,
            ),
        )

    workflow_config = DatasetWorkflowConfig(
        project_root=runtime_layout.root,
        splits=resolved_splits,
        filter_config_path=filter_config_path,
        validate_outputs=validate_outputs,
        fail_on_validation_errors=fail_on_validation_errors,
    )
    return DatasetRuntimePlan(
        project_root=runtime_layout.root,
        drive_project_root=drive_layout.root,
        splits=resolved_splits,
        raw_translation_specs=raw_translation_specs,
        raw_bfh_archive_specs=raw_bfh_archive_specs,
        workflow_config=workflow_config,
    )


def _copy_file_command_spec(
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
        failure=f"Failed to copy {label}: {source_path} -> {target_path}",
    )


def _extract_raw_archive_command_spec(
    *,
    label: str,
    archive_path: Path,
    extraction_root: Path,
    expected_split_root: Path,
    expected_input_bytes: int,
) -> CommandSpec:
    return build_tar_zstd_extract_command(
        label=label,
        archive_path=archive_path,
        extraction_root=extraction_root,
        mkdir_path=expected_split_root,
        expected_input_bytes=expected_input_bytes,
        failure=f"Failed to extract {label} into {extraction_root}: {archive_path}",
    )


__all__ = ["build_dataset_runtime_plan"]
