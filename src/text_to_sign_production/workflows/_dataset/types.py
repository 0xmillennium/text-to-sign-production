"""Dataclass DTOs and error contracts for the Dataset workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from text_to_sign_production.data.constants import SPLITS
from text_to_sign_production.data.schemas import ValidationIssue
from text_to_sign_production.workflows.commands import CommandSpec


@dataclass(frozen=True, slots=True)
class DatasetWorkflowConfig:
    """Dataset workflow configuration."""

    project_root: Path | str
    splits: tuple[str, ...] = SPLITS
    filter_config_path: Path | str | None = None
    quality_config_path: Path | str | None = None
    validate_outputs: bool = True
    fail_on_validation_errors: bool = True


@dataclass(frozen=True, slots=True)
class DatasetWorkflowResult:
    project_root: Path
    data_root: Path
    run_root: Path
    splits: tuple[str, ...]
    raw_manifest_paths: Mapping[str, Path]
    normalized_manifest_paths: Mapping[str, Path]
    filtered_manifest_paths: Mapping[str, Path]
    processed_manifest_paths: Mapping[str, Path]
    processed_samples_root: Path
    interim_report_paths: Mapping[str, Path]
    processed_report_paths: Mapping[str, Path]
    quality_report_paths: Mapping[str, Path]
    leakage_report_paths: Mapping[str, Path]
    tier_manifest_paths: Mapping[str, Mapping[str, Path]]
    tier_counts: Mapping[str, Mapping[str, int]]
    blocking_leakage: bool
    quality_config_hash: str | None
    processed_sample_archive_members: Mapping[str, tuple[Path, ...]]
    assumption_report: Mapping[str, Any]
    filter_report: Mapping[str, Any]
    export_report: Mapping[str, Any]
    validation_issues: Mapping[str, tuple[ValidationIssue, ...]]


@dataclass(frozen=True, slots=True)
class DatasetWorkflowOutputSummary:
    """Notebook-facing summary of verified Dataset workflow outputs."""

    splits: tuple[str, ...]
    processed_manifest_paths: Mapping[str, Path]
    processed_sample_archive_member_counts: Mapping[str, int]
    validation_error_count: int
    validation_warning_count: int
    validation_issue_counts: Mapping[str, int]
    raw_source_issue_counts: Mapping[str, int]
    dropped_sample_counts: Mapping[str, int]
    top_drop_reasons: Mapping[str, tuple[tuple[str, int], ...]]
    report_paths: Mapping[str, Path]
    processed_report_paths: Mapping[str, Path]
    quality_report_paths: Mapping[str, Path]
    leakage_report_paths: Mapping[str, Path]
    tier_manifest_paths: Mapping[str, Mapping[str, Path]]
    tier_counts: Mapping[str, Mapping[str, int]]
    blocking_leakage: bool
    quality_config_hash: str | None
    interim_report_paths: Mapping[str, Path]


@dataclass(frozen=True, slots=True)
class DatasetPublishSummary:
    """Notebook-facing summary of verified Dataset publish outputs."""

    file_paths: Mapping[str, Path]
    file_sizes: Mapping[str, int]
    sample_archive_paths: Mapping[str, Path]
    sample_archive_sizes: Mapping[str, int]
    sample_archive_member_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class DatasetFileTransferSpec:
    """One raw source file copy from Drive into the runtime worktree."""

    split: str
    label: str
    source_path: Path
    target_path: Path
    expected_input_bytes: int
    copy_command: CommandSpec


@dataclass(frozen=True, slots=True)
class DatasetRawArchiveRestoreSpec:
    """One source-format raw BFH archive extraction contract."""

    split: str
    label: str
    archive_path: Path
    extraction_root: Path
    expected_split_root: Path
    expected_openpose_root: Path
    expected_json_root: Path
    expected_video_root: Path
    expected_input_bytes: int
    extract_command: CommandSpec


@dataclass(frozen=True, slots=True)
class DatasetRuntimePlan:
    """Notebook-facing runtime input plan for Dataset Build."""

    project_root: Path
    drive_project_root: Path
    splits: tuple[str, ...]
    raw_translation_specs: Mapping[str, DatasetFileTransferSpec]
    raw_bfh_archive_specs: Mapping[str, DatasetRawArchiveRestoreSpec]
    workflow_config: DatasetWorkflowConfig


@dataclass(frozen=True, slots=True)
class DatasetArchiveSpec:
    """One project-generated dataset archive contract."""

    label: str
    archive_path: Path
    archive_parent: Path
    source_root: Path
    member_list_path: Path
    observed_member_list_path: Path
    members: tuple[str, ...]
    expected_member_count: int
    split: str | None
    create_command: CommandSpec
    verify_command: CommandSpec


@dataclass(frozen=True, slots=True)
class DatasetFilePublishSpec:
    """One small generated dataset file publish/copy contract."""

    label: str
    split: str | None
    source_path: Path
    target_path: Path
    expected_input_bytes: int
    publish_command: CommandSpec


@dataclass(frozen=True, slots=True)
class DatasetPublishPlan:
    """Notebook-facing publish plan for Dataset Build outputs."""

    interim_manifest_files: Mapping[str, DatasetFilePublishSpec]
    processed_manifest_files: Mapping[str, DatasetFilePublishSpec]
    interim_report_files: Mapping[str, DatasetFilePublishSpec]
    processed_report_files: Mapping[str, DatasetFilePublishSpec]
    sample_archives: Mapping[str, DatasetArchiveSpec]


@dataclass(frozen=True, slots=True)
class DatasetQualityReadinessSummary:
    """Operator-facing quality readiness summary for the Dataset notebook.

    The notebook prints this to confirm quality/leakage/tier artifacts
    are present before proceeding to Base check_quality runs.
    """

    quality_config_path: str | None
    quality_config_hash: str | None
    quality_report_paths: Mapping[str, str]
    leakage_report_paths: Mapping[str, str]
    blocking_for_complete: bool
    tier_manifest_paths: Mapping[str, Mapping[str, str]]
    tier_counts: Mapping[str, Mapping[str, int]]
    complete_dataset_gate: str
    complete_dataset_gate_pass: bool


class DatasetWorkflowError(RuntimeError):
    """Raised when the dataset workflow fails during orchestration."""


class DatasetWorkflowInputError(ValueError):
    """Raised when dataset workflow inputs are missing or invalid."""


__all__ = [
    "DatasetArchiveSpec",
    "DatasetFilePublishSpec",
    "DatasetFileTransferSpec",
    "DatasetPublishPlan",
    "DatasetPublishSummary",
    "DatasetQualityReadinessSummary",
    "DatasetRawArchiveRestoreSpec",
    "DatasetRuntimePlan",
    "DatasetWorkflowConfig",
    "DatasetWorkflowError",
    "DatasetWorkflowInputError",
    "DatasetWorkflowOutputSummary",
    "DatasetWorkflowResult",
]
