"""Notebook-facing Dataset Build workflow over the canonical project layout."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from text_to_sign_production.core.files import (
    ensure_dir,
    require_dir,
    require_dir_contains,
    require_non_empty_file,
    verify_output_dir,
    verify_output_file,
)
from text_to_sign_production.core.paths import (
    DEFAULT_REPO_ROOT,
    ProjectLayout,
    repo_relative_path,
    resolve_repo_path,
)
from text_to_sign_production.data.constants import DEFAULT_FILTER_CONFIG_RELATIVE_PATH, SPLITS
from text_to_sign_production.data.filtering import filter_all_splits
from text_to_sign_production.data.manifests import export_final_manifests
from text_to_sign_production.data.normalize import normalize_all_splits
from text_to_sign_production.data.processed_samples import build_sample_index
from text_to_sign_production.data.raw import SplitPaths, build_raw_manifests, get_split_paths
from text_to_sign_production.data.schemas import ValidationIssue
from text_to_sign_production.data.validate import validate_manifest


@dataclass(frozen=True, slots=True)
class DatasetWorkflowConfig:
    """Dataset workflow configuration.

    ``project_root`` is the supported notebook-facing surface. ``data_root`` remains only
    as transitional compatibility for callers that still pass the project data directory.
    """

    project_root: Path | str | None = None
    splits: tuple[str, ...] = SPLITS
    filter_config_path: Path | str | None = None
    data_root: Path | str | None = None
    validate_outputs: bool = True
    fail_on_validation_errors: bool = True


@dataclass(frozen=True, slots=True)
class DatasetWorkflowResult:
    project_root: Path
    data_root: Path
    dataset_artifacts_root: Path
    splits: tuple[str, ...]
    raw_manifest_paths: Mapping[str, Path]
    normalized_manifest_paths: Mapping[str, Path]
    filtered_manifest_paths: Mapping[str, Path]
    processed_manifest_paths: Mapping[str, Path]
    processed_samples_root: Path
    interim_report_paths: Mapping[str, Path]
    processed_report_paths: Mapping[str, Path]
    processed_sample_archive_members: Mapping[str, tuple[Path, ...]]
    assumption_report: Mapping[str, Any]
    filter_report: Mapping[str, Any]
    export_report: Mapping[str, Any]
    validation_issues: Mapping[str, tuple[ValidationIssue, ...]]


class DatasetWorkflowError(RuntimeError):
    """Raised when the dataset workflow fails during orchestration."""


class DatasetWorkflowInputError(ValueError):
    """Raised when dataset workflow inputs are missing or invalid."""


def validate_dataset_inputs(config: DatasetWorkflowConfig) -> None:
    """Validate canonical runtime inputs without creating derived outputs."""

    layout = _layout_from_config(config)
    splits = _validate_splits(config.splits)
    try:
        require_non_empty_file(_filter_config_path(config, layout), label="Filter config")
        for split in splits:
            split_paths = get_split_paths(split, layout=layout)
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


def run_dataset_workflow(config: DatasetWorkflowConfig) -> DatasetWorkflowResult:
    """Run raw, normalized, filtered, and processed manifest stages in order."""

    validate_dataset_inputs(config)
    layout = _layout_from_config(config)
    splits = _validate_splits(config.splits)
    filter_config_path = _filter_config_path(config, layout)
    paths = _dataset_output_paths(layout=layout, splits=splits)
    split_paths_by_split = _split_paths_by_split(layout, splits)
    validation_issues: dict[str, tuple[ValidationIssue, ...]] = {}

    try:
        ensure_dir(layout.dataset_artifacts_root, label="Dataset artifact root")
        assumption_report = build_raw_manifests(
            splits=splits,
            split_paths_by_split=split_paths_by_split,
            raw_manifests_root=layout.raw_manifests_root,
            interim_reports_root=layout.interim_reports_root,
            raw_root=layout.how2sign_root,
            data_root=layout.data_root,
        )
        _verify_files(paths["raw_manifest_paths"], label="Raw manifest")
        _validate_stage_outputs(
            validation_issues,
            stage="raw",
            manifest_paths=paths["raw_manifest_paths"],
            kind="raw",
            data_root=layout.data_root,
            enabled=config.validate_outputs,
        )

        normalize_all_splits(
            splits=splits,
            raw_manifests_root=layout.raw_manifests_root,
            normalized_manifests_root=layout.normalized_manifests_root,
            processed_samples_root=layout.processed_v1_samples_root,
            data_root=layout.data_root,
        )
        _verify_files(paths["normalized_manifest_paths"], label="Normalized manifest")
        _validate_stage_outputs(
            validation_issues,
            stage="normalized",
            manifest_paths=paths["normalized_manifest_paths"],
            kind="normalized",
            data_root=layout.data_root,
            enabled=config.validate_outputs,
        )

        filter_report = filter_all_splits(
            filter_config_path,
            splits=splits,
            normalized_manifests_root=layout.normalized_manifests_root,
            filtered_manifests_root=layout.filtered_manifests_root,
            interim_reports_root=layout.interim_reports_root,
            data_root=layout.data_root,
        )
        _verify_files(paths["filtered_manifest_paths"], label="Filtered manifest")
        _validate_stage_outputs(
            validation_issues,
            stage="filtered",
            manifest_paths=paths["filtered_manifest_paths"],
            kind="normalized",
            data_root=layout.data_root,
            enabled=config.validate_outputs,
        )
        export_report = export_final_manifests(
            assumption_report=assumption_report,
            filter_report=filter_report,
            splits=splits,
            raw_manifests_root=layout.raw_manifests_root,
            filtered_manifests_root=layout.filtered_manifests_root,
            processed_manifests_root=layout.processed_v1_manifests_root,
            processed_samples_root=layout.processed_v1_samples_root,
            processed_reports_root=layout.processed_v1_reports_root,
            data_root=layout.data_root,
        )
        _verify_files(paths["processed_manifest_paths"], label="Processed manifest")
        _verify_files(paths["processed_report_paths"], label="Processed report")
        for split in splits:
            verify_output_dir(
                layout.processed_samples_split_root(split),
                label=f"{split} processed samples root",
            )
        _validate_stage_outputs(
            validation_issues,
            stage="processed",
            manifest_paths=paths["processed_manifest_paths"],
            kind="processed",
            data_root=layout.data_root,
            enabled=config.validate_outputs,
        )
        processed_sample_archive_members = _collect_processed_sample_archive_members(
            layout=layout,
            splits=splits,
        )
    except DatasetWorkflowInputError:
        raise
    except Exception as exc:
        raise DatasetWorkflowError(f"Dataset workflow failed: {exc}") from exc

    if config.validate_outputs and config.fail_on_validation_errors:
        _raise_for_validation_errors(validation_issues)

    return DatasetWorkflowResult(
        project_root=layout.root,
        data_root=layout.data_root,
        dataset_artifacts_root=layout.dataset_artifacts_root,
        splits=splits,
        raw_manifest_paths=paths["raw_manifest_paths"],
        normalized_manifest_paths=paths["normalized_manifest_paths"],
        filtered_manifest_paths=paths["filtered_manifest_paths"],
        processed_manifest_paths=paths["processed_manifest_paths"],
        processed_samples_root=paths["processed_samples_root"],
        interim_report_paths=paths["interim_report_paths"],
        processed_report_paths=paths["processed_report_paths"],
        processed_sample_archive_members=processed_sample_archive_members,
        assumption_report=assumption_report,
        filter_report=filter_report,
        export_report=export_report,
        validation_issues=validation_issues,
    )


def _layout_from_config(config: DatasetWorkflowConfig) -> ProjectLayout:
    if config.project_root is not None and config.data_root is not None:
        project_layout = ProjectLayout(Path(config.project_root))
        data_root = Path(config.data_root).expanduser().resolve()
        if data_root != project_layout.data_root:
            raise DatasetWorkflowInputError(
                f"project_root and data_root disagree: {project_layout.data_root} != {data_root}"
            )
        return project_layout
    if config.project_root is not None:
        return ProjectLayout(Path(config.project_root))
    if config.data_root is not None:
        data_root = Path(config.data_root).expanduser().resolve()
        if data_root.name != "data":
            raise DatasetWorkflowInputError(
                f"data_root must point at a project data directory: {data_root}"
            )
        return ProjectLayout(data_root.parent)
    return ProjectLayout(DEFAULT_REPO_ROOT)


def _filter_config_path(config: DatasetWorkflowConfig, layout: ProjectLayout) -> Path:
    configured = config.filter_config_path or DEFAULT_FILTER_CONFIG_RELATIVE_PATH
    return resolve_repo_path(configured, repo_root=layout.root)


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


def _split_paths_by_split(
    layout: ProjectLayout,
    splits: tuple[str, ...],
) -> dict[str, SplitPaths]:
    return {split: get_split_paths(split, layout=layout) for split in splits}


def _dataset_output_paths(
    *,
    layout: ProjectLayout,
    splits: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "raw_manifest_paths": {
            split: layout.raw_manifests_root / f"raw_{split}.jsonl" for split in splits
        },
        "normalized_manifest_paths": {
            split: layout.normalized_manifests_root / f"normalized_{split}.jsonl"
            for split in splits
        },
        "filtered_manifest_paths": {
            split: layout.filtered_manifests_root / f"filtered_{split}.jsonl" for split in splits
        },
        "processed_manifest_paths": {
            split: layout.processed_manifest_path(split) for split in splits
        },
        "processed_samples_root": layout.processed_v1_samples_root,
        "interim_report_paths": {
            "assumption": layout.interim_reports_root / "assumption-report.json",
            "filter": layout.interim_reports_root / "filter-report.json",
        },
        "processed_report_paths": {
            "assumption_markdown": layout.processed_v1_reports_root / "assumption-report.md",
            "data_quality_json": layout.processed_v1_reports_root / "data-quality-report.json",
            "data_quality_markdown": layout.processed_v1_reports_root / "data-quality-report.md",
            "split_json": layout.processed_v1_reports_root / "split-report.json",
            "split_markdown": layout.processed_v1_reports_root / "split-report.md",
        },
    }


def _verify_files(paths: Mapping[str, Path], *, label: str) -> None:
    for key, path in paths.items():
        verify_output_file(path, label=f"{label} {key}")


def _collect_processed_sample_archive_members(
    *,
    layout: ProjectLayout,
    splits: tuple[str, ...],
) -> dict[str, tuple[Path, ...]]:
    sample_index = build_sample_index(
        layout.data_root,
        splits=splits,
        require_sample_files=True,
        processed_manifests_root=layout.processed_v1_manifests_root,
        processed_samples_root=layout.processed_v1_samples_root,
        filtered_manifests_root=layout.filtered_manifests_root,
        normalized_manifests_root=layout.normalized_manifests_root,
        raw_manifests_root=layout.raw_manifests_root,
    )
    members_by_split: dict[str, list[Path]] = {split: [] for split in splits}
    for record in sample_index.records:
        project_relative = Path(repo_relative_path(record.sample_path, repo_root=layout.root))
        expected_root = Path("data") / "processed" / "v1" / "samples" / record.split
        if not project_relative.is_relative_to(expected_root) or project_relative.suffix != ".npz":
            raise ValueError(
                "Processed sample archive member must be a project-relative .npz under "
                f"{expected_root.as_posix()}: {project_relative.as_posix()}"
            )
        members_by_split[record.split].append(project_relative)
    return {split: tuple(members_by_split[split]) for split in splits}


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


__all__ = [
    "DatasetWorkflowConfig",
    "DatasetWorkflowError",
    "DatasetWorkflowInputError",
    "DatasetWorkflowResult",
    "run_dataset_workflow",
    "validate_dataset_inputs",
]
