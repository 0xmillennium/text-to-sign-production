"""Dataclass DTOs and error contracts for the Base workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.core.files import OutputExistsPolicy
from text_to_sign_production.modeling.inference.qualitative import (
    DEFAULT_QUALITATIVE_PANEL_SIZE,
    QualitativeExportResult,
)
from text_to_sign_production.modeling.inference.writer import SplitPredictionWriteResult
from text_to_sign_production.modeling.training.config import BaselineTrainingConfig
from text_to_sign_production.modeling.training.train import BaselineTrainingRunResult
from text_to_sign_production.workflows._base.constants import (
    BASE_DEFAULT_PREDICTION_SPLITS,
    BASE_DEFAULT_RUN_MODE,
    BASE_RUN_QUALITATIVE_EXPORT,
    BaseRunMode,
)
from text_to_sign_production.workflows.commands import CommandSpec


@dataclass(frozen=True, slots=True)
class BaseRunLayout:
    """Canonical reproducible M0 run-root layout."""

    project_root: Path
    run_name: str
    run_root: Path
    config_dir: Path
    config_snapshot_path: Path
    baseline_config_copy_path: Path
    training_dir: Path
    training_metrics_path: Path
    training_live_log_path: Path
    training_summary_path: Path
    checkpoints_dir: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    predictions_dir: Path
    reports_dir: Path
    baseline_report_json_path: Path
    baseline_report_markdown_path: Path
    failure_modes_json_path: Path
    failure_modes_markdown_path: Path
    qualitative_dir: Path
    run_summary_path: Path

    def prediction_split_dir(self, split: str) -> Path:
        """Return the canonical prediction directory for a split."""

        return self.predictions_dir / split

    def prediction_manifest_path(self, split: str) -> Path:
        """Return the canonical prediction manifest path for a split."""

        return self.prediction_split_dir(split) / "manifest.jsonl"

    def prediction_samples_dir(self, split: str) -> Path:
        """Return the canonical prediction sample directory for a split."""

        return self.prediction_split_dir(split) / "samples"


@dataclass(frozen=True, slots=True)
class BaseWorkflowConfig:
    project_root: Path | str
    drive_project_root: Path | str | None = None
    run_name: str = "smoke_run"
    git_revision: str | None = None
    config_path: Path | str | None = None
    run_mode: BaseRunMode | str = BASE_DEFAULT_RUN_MODE
    prediction_splits: tuple[str, ...] = BASE_DEFAULT_PREDICTION_SPLITS
    run_qualitative_export: bool = BASE_RUN_QUALITATIVE_EXPORT
    panel_size: int = DEFAULT_QUALITATIVE_PANEL_SIZE
    validate_processed_inputs: bool = True
    output_policy: OutputExistsPolicy | str = OutputExistsPolicy.FAIL
    limit_train_samples: int | None = None
    limit_validation_samples: int | None = None
    limit_prediction_samples: int | None = None
    epoch_count: int = 1
    min_epochs: int = 1
    early_stopping_patience: int = 1
    shuffle_train: bool = False


@dataclass(frozen=True, slots=True)
class BaseWorkflowResult:
    config: BaselineTrainingConfig
    training: BaselineTrainingRunResult
    predictions: Mapping[str, SplitPredictionWriteResult]
    qualitative: QualitativeExportResult | None
    run_root: Path
    run_layout: BaseRunLayout
    config_snapshot_path: Path
    baseline_config_copy_path: Path
    baseline_report_json_path: Path
    baseline_report_markdown_path: Path
    failure_modes_json_path: Path
    failure_modes_markdown_path: Path
    run_summary_path: Path
    verification: BaseRunVerificationSummary


@dataclass(frozen=True, slots=True)
class BaseProcessedManifestRestoreSpec:
    """One direct processed manifest restore contract."""

    split: str
    label: str
    source_path: Path
    target_path: Path
    expected_input_bytes: int
    restore_command: CommandSpec


@dataclass(frozen=True, slots=True)
class BaseProcessedSampleArchiveRestoreSpec:
    """One processed sample archive restore contract."""

    split: str
    label: str
    archive_path: Path
    data_root: Path
    extraction_root: Path
    expected_split_root: Path
    target_manifest_path: Path
    expected_input_bytes: int
    extract_command: CommandSpec


@dataclass(frozen=True, slots=True)
class BaseRuntimePlan:
    """Notebook-facing processed artifact restore plan for an M0 run."""

    project_root: Path
    drive_project_root: Path
    run_name: str
    train_split: str
    validation_split: str
    prediction_splits: tuple[str, ...]
    required_splits: tuple[str, ...]
    run_mode: BaseRunMode
    limit_train_samples: int | None
    limit_validation_samples: int | None
    limit_prediction_samples: int | None
    epoch_count: int
    min_epochs: int
    early_stopping_patience: int
    training_surface: str
    validation_surface: str
    live_log_path: Path
    resume_supported: bool
    periodic_persistence: bool
    shuffle_train: bool
    run_qualitative_export: bool
    panel_size: int
    run_layout: BaseRunLayout
    expected_publish_run_root: Path
    processed_manifest_files: Mapping[str, BaseProcessedManifestRestoreSpec]
    processed_sample_archives: Mapping[str, BaseProcessedSampleArchiveRestoreSpec]
    workflow_config: BaseWorkflowConfig


@dataclass(frozen=True, slots=True)
class BaseProcessedSplitInputVerification:
    """Structured verification summary for one restored processed split."""

    split: str
    manifest_path: Path
    manifest_record_count: int
    samples_dir: Path
    sample_count: int
    checked_manifest_sample_count: int


@dataclass(frozen=True, slots=True)
class BaseProcessedRestoreVerificationSummary:
    """Notebook-facing verification summary for restored Base inputs."""

    required_splits: tuple[str, ...]
    split_inputs: Mapping[str, BaseProcessedSplitInputVerification]


@dataclass(frozen=True, slots=True)
class BasePredictionSplitVerification:
    """Verification summary for one prediction split."""

    split: str
    manifest_path: Path
    manifest_record_count: int
    samples_dir: Path
    sample_count: int
    validated_sample_path: Path | None


@dataclass(frozen=True, slots=True)
class BaseRunVerificationSummary:
    """Structured verification summary for future notebook display."""

    run_root: Path
    config_snapshot_path: Path
    baseline_config_copy_path: Path
    training_metrics_path: Path
    training_live_log_path: Path
    training_metric_record_count: int
    training_summary_path: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    prediction_splits: Mapping[str, BasePredictionSplitVerification]
    baseline_report_json_path: Path
    baseline_report_markdown_path: Path
    failure_modes_json_path: Path
    failure_modes_markdown_path: Path
    qualitative: BaseQualitativeVerificationSummary | None
    run_summary_path: Path


@dataclass(frozen=True, slots=True)
class BaseQualitativeVerificationSummary:
    """Verification summary for optional qualitative Base artifacts."""

    output_dir: Path
    panel_definition_path: Path
    records_path: Path
    record_count: int
    panel_summary_path: Path
    evidence_bundle_path: Path
    sample_count: int
    sample_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BaseWorkflowOutputSummary:
    """Notebook-facing summary of a completed Base workflow run."""

    run_root: Path
    config_snapshot_path: Path
    baseline_config_copy_path: Path
    training_metrics_path: Path
    training_live_log_path: Path
    training_metric_record_count: int
    training_summary_path: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    prediction_splits: Mapping[str, BasePredictionSplitVerification]
    baseline_report_json_path: Path
    baseline_report_markdown_path: Path
    failure_modes_json_path: Path
    failure_modes_markdown_path: Path
    qualitative: BaseQualitativeVerificationSummary | None
    run_summary_path: Path
    verification_status: str


@dataclass(frozen=True, slots=True)
class BaseFilePublishSpec:
    """One small inspectable Base file publish/copy contract."""

    label: str
    group: str
    split: str | None
    source_path: Path
    target_path: Path
    source_size: int
    source_sha256: str
    copy_command: CommandSpec


@dataclass(frozen=True, slots=True)
class BaseCheckpointPublishSpec:
    """One runtime checkpoint compression publish contract."""

    role: str
    source_path: Path
    target_path: Path
    source_size: int
    source_sha256: str
    compress_command: CommandSpec


@dataclass(frozen=True, slots=True)
class BasePredictionSampleArchiveSpec:
    """One split prediction sample archive publish contract."""

    split: str
    runtime_manifest_path: Path
    runtime_samples_dir: Path
    member_list_path: Path
    observed_member_list_path: Path
    target_archive_path: Path
    members: tuple[str, ...]
    expected_member_count: int
    archive_command: CommandSpec
    list_command: CommandSpec


@dataclass(frozen=True, slots=True)
class BaseQualitativeSampleArchiveSpec:
    """Qualitative reference/prediction sample archive publish contract."""

    runtime_qualitative_dir: Path
    member_list_path: Path
    observed_member_list_path: Path
    target_archive_path: Path
    members: tuple[str, ...]
    expected_member_count: int
    archive_command: CommandSpec
    list_command: CommandSpec


@dataclass(frozen=True, slots=True)
class BasePublishPlan:
    """Notebook-facing publish plan for Base run outputs."""

    run_name: str
    runtime_project_root: Path
    drive_project_root: Path
    runtime_run_root: Path
    drive_run_root: Path
    direct_file_specs: tuple[BaseFilePublishSpec, ...]
    checkpoint_specs: tuple[BaseCheckpointPublishSpec, ...]
    checkpoint_manifest_source_path: Path
    checkpoint_manifest_target_path: Path
    prediction_sample_archive_specs: tuple[BasePredictionSampleArchiveSpec, ...]
    qualitative_sample_archive_spec: BaseQualitativeSampleArchiveSpec | None


@dataclass(frozen=True, slots=True)
class BasePublishedFileSummary:
    """Verified direct publish file summary."""

    label: str
    group: str
    split: str | None
    source_path: Path
    target_path: Path
    source_size: int
    target_size: int
    source_sha256: str
    target_sha256: str


@dataclass(frozen=True, slots=True)
class BasePublishedCheckpointSummary:
    """Verified checkpoint publish summary."""

    role: str
    source_path: Path
    target_path: Path
    source_size: int
    compressed_size: int
    source_sha256: str
    compressed_sha256: str


@dataclass(frozen=True, slots=True)
class BasePublishedArchiveSummary:
    """Verified sample archive publish summary."""

    label: str
    target_archive_path: Path
    archive_size: int
    observed_member_list_path: Path
    expected_member_count: int
    observed_member_count: int


@dataclass(frozen=True, slots=True)
class BasePublishSummary:
    """Notebook-facing summary of verified Base publish outputs."""

    direct_files: tuple[BasePublishedFileSummary, ...]
    checkpoints: tuple[BasePublishedCheckpointSummary, ...]
    checkpoint_manifest: BasePublishedFileSummary
    prediction_archives: tuple[BasePublishedArchiveSummary, ...]
    qualitative_archive: BasePublishedArchiveSummary | None
    missing_count: int
    invalid_count: int
    verification_status: str


@dataclass(frozen=True, slots=True)
class BaseRuntimeReadinessSummary:
    """Operator-facing readiness summary printed by the notebook before training.

    All fields are produced by the workflow layer so the notebook can print
    them without parsing JSON reports or computing policy.
    """

    run_mode: str
    run_name: str
    live_log_path: str
    recovery_persistence_enabled: bool
    recovery_root: str | None
    recovery_manifest_path: str | None
    training_surface: str
    validation_surface: str
    quality_tier_config_path: str | None
    quality_tier_config_hash: str | None
    tier_counts: Mapping[str, Mapping[str, int]] | None
    primary_metric_name: str
    target_standardization_enabled: bool
    target_standardization_fit_surface: str
    target_standardization_leakage_policy: str
    target_standardization_metadata_path: str | None
    target_standardization_inverse_prediction: bool
    max_epochs: int
    min_epochs: int
    early_stopping_patience: int
    progress_expectation: str
    output_policy: str
    resume_supported: bool
    duration_estimate: str
    complete_mode_warning: str | None
    check_mode_warning: str | None
    risk_control_reminders: tuple[str, ...]


class BaseWorkflowError(RuntimeError):
    """Raised when baseline workflow orchestration fails."""


class BaseWorkflowInputError(ValueError):
    """Raised when baseline workflow inputs are missing or invalid."""


__all__ = [
    "BaseRunLayout",
    "BaseWorkflowConfig",
    "BaseWorkflowResult",
    "BaseProcessedManifestRestoreSpec",
    "BaseProcessedSampleArchiveRestoreSpec",
    "BaseRuntimePlan",
    "BaseProcessedSplitInputVerification",
    "BaseProcessedRestoreVerificationSummary",
    "BasePredictionSplitVerification",
    "BaseRunVerificationSummary",
    "BaseQualitativeVerificationSummary",
    "BaseWorkflowOutputSummary",
    "BaseRuntimeReadinessSummary",
    "BaseFilePublishSpec",
    "BaseCheckpointPublishSpec",
    "BasePredictionSampleArchiveSpec",
    "BaseQualitativeSampleArchiveSpec",
    "BasePublishPlan",
    "BasePublishedFileSummary",
    "BasePublishedCheckpointSummary",
    "BasePublishedArchiveSummary",
    "BasePublishSummary",
    "BaseWorkflowError",
    "BaseWorkflowInputError",
]
