"""Dataset workflow input and stage-output validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from text_to_sign_production.core.files import (
    require_dir,
    require_dir_contains,
    require_non_empty_file,
    verify_output_file,
)
from text_to_sign_production.data.constants import SPLITS
from text_to_sign_production.data.schemas import ValidationIssue
from text_to_sign_production.data.validate import validate_manifest
from text_to_sign_production.workflows._dataset.layout import (
    _filter_config_path,
    _layout_from_config,
    _split_paths_by_split,
)
from text_to_sign_production.workflows._dataset.types import (
    DatasetWorkflowConfig,
    DatasetWorkflowError,
    DatasetWorkflowInputError,
)


def validate_dataset_inputs(config: DatasetWorkflowConfig) -> None:
    """Validate canonical runtime inputs without creating derived outputs."""

    layout = _layout_from_config(config)
    splits = _validate_splits(config.splits)
    try:
        require_non_empty_file(_filter_config_path(config, layout), label="Filter config")
        split_paths_by_split = _split_paths_by_split(layout, splits)
        for split, split_paths in split_paths_by_split.items():
            require_non_empty_file(
                split_paths.translation_path,
                label=f"{split} How2Sign translation file",
            )
            require_dir(split_paths.keypoints_json_root, label=f"{split} BFH keypoint JSON root")
            require_dir_contains(
                split_paths.keypoints_json_root,
                "*",
                label=f"{split} BFH keypoint JSON root",
            )
            require_dir(split_paths.video_root, label=f"{split} BFH source video root")
            require_dir_contains(
                split_paths.video_root,
                "*.mp4",
                label=f"{split} BFH source video root",
            )
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, ValueError) as exc:
        raise DatasetWorkflowInputError(str(exc)) from exc


def _validate_splits(splits: tuple[str, ...]) -> tuple[str, ...]:
    if not splits:
        raise DatasetWorkflowInputError("At least one dataset split is required.")
    duplicates = sorted({split for split in splits if splits.count(split) > 1})
    if duplicates:
        raise DatasetWorkflowInputError(f"Duplicate dataset split(s): {duplicates}")
    invalid = [split for split in splits if split not in SPLITS]
    if invalid:
        expected = ", ".join(SPLITS)
        raise DatasetWorkflowInputError(
            f"Unsupported dataset split(s) {invalid!r}; expected values from: {expected}."
        )
    return tuple(split for split in SPLITS if split in set(splits))


def _verify_files(paths: Mapping[str, Path], *, label: str) -> None:
    for key, path in paths.items():
        verify_output_file(path, label=f"{label} {key}")


def _validate_stage_outputs(
    validation_issues: dict[str, tuple[ValidationIssue, ...]],
    *,
    stage: str,
    manifest_paths: Mapping[str, Path],
    kind: str,
    data_root: Path,
    enabled: bool,
) -> None:
    if not enabled:
        return
    for split, manifest_path in manifest_paths.items():
        verify_output_file(manifest_path, label=f"{stage} manifest {split}")
        validation_issues[f"{stage}:{split}"] = tuple(
            validate_manifest(manifest_path, kind=kind, data_root=data_root)
        )


def _raise_for_validation_errors(
    validation_issues: Mapping[str, tuple[ValidationIssue, ...]],
) -> None:
    failures = [
        f"{stage_key}:{issue.code}"
        for stage_key, issues in validation_issues.items()
        for issue in issues
        if issue.severity == "error"
    ]
    if failures:
        rendered = ", ".join(failures)
        raise DatasetWorkflowError(f"Dataset workflow validation failed: {rendered}")


__all__ = ["validate_dataset_inputs"]
