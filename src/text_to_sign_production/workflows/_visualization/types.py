"""Dataclass DTOs and error contracts for the Visualization workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from text_to_sign_production.artifacts.refs import ManifestSampleRecord, TimingMetadata
from text_to_sign_production.workflows._visualization.constants import (
    DEFAULT_VISUALIZATION_OUTPUT_FILENAME,
    DEFAULT_VISUALIZATION_RUN_NAME,
    VISUALIZATION_ALLOWED_SPLITS,
)
from text_to_sign_production.workflows.commands import CommandSpec

VisualizationRenderMode = Literal["skeleton_only", "side_by_side"]
VisualizationSelectionMode = Literal["sample_id", "random"]


@dataclass(frozen=True, slots=True)
class VisualizationWorkflowConfig:
    """Notebook-facing visualization workflow configuration."""

    project_root: Path | str
    drive_project_root: Path | str
    splits: tuple[str, ...] = VISUALIZATION_ALLOWED_SPLITS
    selection_mode: VisualizationSelectionMode = "random"
    sample_id: str | None = None
    random_seed: int | None = None
    render_mode: VisualizationRenderMode = "skeleton_only"
    output_exists_policy: str = "fail"
    output_filename: str | None = DEFAULT_VISUALIZATION_OUTPUT_FILENAME
    fps: float | None = None
    run_name: str = DEFAULT_VISUALIZATION_RUN_NAME


@dataclass(frozen=True, slots=True)
class _SampleSelectionConfig:
    """Internal sample selection config with an explicit project root."""

    project_root: Path | str
    splits: tuple[str, ...] | None = None
    sample_id: str | None = None
    random_selection: bool = False
    seed: int | None = None
    require_sample_files: bool = False


@dataclass(frozen=True, slots=True)
class SelectedVisualizationSample:
    """A selected processed sample plus joined timing metadata."""

    record: ManifestSampleRecord
    timing: TimingMetadata
    sample_path: Path
    source_video_path: Path | None


@dataclass(frozen=True, slots=True)
class VisualizationArchiveRestoreSpec:
    """One project-generated archive restore contract."""

    label: str
    archive_path: Path
    extraction_root: Path
    expected_input_bytes: int
    extract: CommandSpec


@dataclass(frozen=True, slots=True)
class VisualizationFileRestoreSpec:
    """One direct small-file restore contract for visualization inputs."""

    label: str
    split: str
    source_path: Path
    target_path: Path
    expected_input_bytes: int
    restore_command: CommandSpec


@dataclass(frozen=True, slots=True)
class VisualizationRuntimePlan:
    """Notebook-facing runtime restore plan for visualization."""

    project_root: Path
    drive_project_root: Path
    run_name: str
    run_root: Path
    outputs_root: Path
    metadata_root: Path
    publish_root: Path
    splits: tuple[str, ...]
    render_mode: VisualizationRenderMode
    output_exists_policy: str
    processed_manifest_files: Mapping[str, VisualizationFileRestoreSpec]
    filtered_manifest_files: Mapping[str, VisualizationFileRestoreSpec]
    manifest_files: Mapping[str, VisualizationFileRestoreSpec]
    workflow_config: VisualizationWorkflowConfig


@dataclass(frozen=True, slots=True)
class VisualizationRawVideoRestoreSpec:
    """One source-format raw BFH archive restore contract."""

    split: str
    archive_path: Path
    extraction_root: Path
    expected_split_root: Path
    expected_openpose_root: Path
    expected_video_root: Path
    expected_source_video_path: Path
    expected_input_bytes: int
    extract: CommandSpec


@dataclass(frozen=True, slots=True)
class VisualizationSamplePlan:
    """Notebook-facing selected sample restore plan."""

    project_root: Path
    drive_project_root: Path
    run_name: str
    run_root: Path
    selected_sample: SelectedVisualizationSample
    sample_id: str
    split: str
    sample_archive: VisualizationArchiveRestoreSpec
    selected_sample_path: Path
    raw_video: VisualizationRawVideoRestoreSpec | None


@dataclass(frozen=True, slots=True)
class VisualizationSampleSummary:
    """Notebook-facing operational summary for the selected visualization sample."""

    sample_id: str
    split: str
    text: str | None
    selection_mode: str
    random_seed: int | None
    sample_path: Path
    source_video_path: Path | None
    render_mode: str
    processed_archive_path: Path
    processed_archive_size: int
    raw_archive_path: Path | None
    raw_archive_size: int | None
    raw_restore_required: bool


@dataclass(frozen=True, slots=True)
class VisualizationRuntimeInputSummary:
    """Notebook-facing summary of restored Visualization runtime inputs."""

    manifest_paths: Mapping[str, Path]
    selected_sample_path: Path | None = None
    selected_source_video_path: Path | None = None


@dataclass(frozen=True, slots=True)
class VisualizationRenderPlan:
    """Notebook-facing render plan with primitive configuration values."""

    project_root: Path
    run_name: str
    run_root: Path
    outputs_root: Path
    metadata_root: Path
    render_plan_metadata_path: Path
    render_result_metadata_path: Path
    selected_sample: SelectedVisualizationSample
    sample_id: str
    split: str
    mode: VisualizationRenderMode
    output_path: Path
    output_exists_policy: str
    fps: float | None
    output_filename: str | None


@dataclass(frozen=True, slots=True)
class VisualizationWorkflowResult:
    """Result from a completed visualization render."""

    project_root: Path
    run_name: str
    run_root: Path
    metadata_root: Path
    render_plan_metadata_path: Path
    render_result_metadata_path: Path
    selected_sample: SelectedVisualizationSample
    output_path: Path
    render_metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class VisualizationPublishSpec:
    """One visualization output publish/copy contract."""

    label: str
    source_path: Path
    target_path: Path
    expected_input_bytes: int
    publish_command: CommandSpec


@dataclass(frozen=True, slots=True)
class VisualizationPublishPlan:
    """Notebook-facing publish plan for Visualization outputs."""

    run_name: str
    runtime_project_root: Path
    drive_project_root: Path
    runtime_run_root: Path
    drive_run_root: Path
    output_spec: VisualizationPublishSpec
    metadata_specs: tuple[VisualizationPublishSpec, ...]


@dataclass(frozen=True, slots=True)
class VisualizationWorkflowOutputSummary:
    """Notebook-facing summary of verified visualization workflow outputs."""

    sample_id: str
    split: str
    mode: str
    output_path: Path
    output_size: int
    codec: str
    render_plan_metadata_path: Path
    render_result_metadata_path: Path


@dataclass(frozen=True, slots=True)
class VisualizationPublishSummary:
    """Notebook-facing summary of verified visualization publish outputs."""

    output_source_path: Path
    output_target_path: Path
    output_size: int
    metadata_paths: Mapping[str, Path]
    metadata_sizes: Mapping[str, int]


class VisualizationWorkflowError(RuntimeError):
    """Raised when visualization workflow orchestration fails."""


class VisualizationWorkflowInputError(ValueError):
    """Raised when visualization workflow inputs are missing or invalid."""


__all__ = [
    "SelectedVisualizationSample",
    "VisualizationArchiveRestoreSpec",
    "VisualizationFileRestoreSpec",
    "VisualizationPublishPlan",
    "VisualizationPublishSpec",
    "VisualizationPublishSummary",
    "VisualizationRawVideoRestoreSpec",
    "VisualizationRenderMode",
    "VisualizationRenderPlan",
    "VisualizationRuntimeInputSummary",
    "VisualizationRuntimePlan",
    "VisualizationSamplePlan",
    "VisualizationSampleSummary",
    "VisualizationSelectionMode",
    "VisualizationWorkflowConfig",
    "VisualizationWorkflowError",
    "VisualizationWorkflowInputError",
    "VisualizationWorkflowOutputSummary",
    "VisualizationWorkflowResult",
]
