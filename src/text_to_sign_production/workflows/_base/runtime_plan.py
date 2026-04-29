"""Base workflow runtime restore plan construction."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.artifacts.layout import DatasetArtifactLayout
from text_to_sign_production.core.files import OutputExistsPolicy, require_non_empty_file
from text_to_sign_production.core.paths import ProjectLayout
from text_to_sign_production.modeling.training.config import load_baseline_training_config
from text_to_sign_production.workflows._base.constants import (
    BASE_DEFAULT_RUN_MODE,
    BaseRunMode,
)
from text_to_sign_production.workflows._base.layout import _config_path, build_base_run_layout
from text_to_sign_production.workflows._base.types import (
    BaseProcessedManifestRestoreSpec,
    BaseProcessedSampleArchiveRestoreSpec,
    BaseRuntimePlan,
    BaseWorkflowConfig,
)
from text_to_sign_production.workflows._base.validate import (
    _ordered_unique_splits,
    _output_policy,
    _run_mode_policy,
    _workflow_config_with_run_mode_policy,
)
from text_to_sign_production.workflows._shared.archives import build_tar_zstd_extract_command
from text_to_sign_production.workflows._shared.transfer import build_byte_progress_copy_command
from text_to_sign_production.workflows.commands import CommandSpec


def build_base_runtime_plan(
    *,
    project_root: Path | str,
    drive_project_root: Path | str,
    run_name: str = "smoke_run",
    config_path: Path | str | None = None,
    run_mode: BaseRunMode | str = BASE_DEFAULT_RUN_MODE,
    output_policy: OutputExistsPolicy | str = OutputExistsPolicy.FAIL,
) -> BaseRuntimePlan:
    """Build notebook-visible processed artifact restore specs without executing commands."""

    runtime_layout = ProjectLayout(Path(project_root))
    drive_layout = ProjectLayout(Path(drive_project_root))
    runtime_artifacts = DatasetArtifactLayout(runtime_layout)
    drive_artifacts = DatasetArtifactLayout(drive_layout)
    resolved_output_policy = _output_policy(output_policy)
    run_mode_policy = _run_mode_policy(run_mode)
    run_layout = build_base_run_layout(project_root=runtime_layout.root, run_name=run_name)
    resolved_config_path = _config_path(
        BaseWorkflowConfig(project_root=runtime_layout.root, config_path=config_path),
        runtime_layout,
    )
    workflow_config = _workflow_config_with_run_mode_policy(
        BaseWorkflowConfig(
            project_root=runtime_layout.root,
            drive_project_root=drive_layout.root,
            run_name=run_name,
            config_path=resolved_config_path,
            run_mode=run_mode_policy.run_mode,
            output_policy=resolved_output_policy,
        )
    )
    training_config = load_baseline_training_config(
        resolved_config_path,
        validate_paths=False,
        checkpoint_output_dir=run_layout.checkpoints_dir,
        repo_root=runtime_layout.root,
    )
    required_splits = _ordered_unique_splits(
        (
            training_config.data.train_split,
            training_config.data.val_split,
            *workflow_config.prediction_splits,
        )
    )

    processed_manifest_files: dict[str, BaseProcessedManifestRestoreSpec] = {}
    processed_sample_archives: dict[str, BaseProcessedSampleArchiveRestoreSpec] = {}
    for split in required_splits:
        source_manifest = require_non_empty_file(
            drive_artifacts.processed_manifest_path(split),
            label=f"Drive {split} processed manifest",
        )
        target_manifest = runtime_artifacts.processed_manifest_path(split)
        manifest_size = source_manifest.stat().st_size
        manifest_label = f"{split} processed manifest"
        processed_manifest_files[split] = BaseProcessedManifestRestoreSpec(
            split=split,
            label=manifest_label,
            source_path=source_manifest,
            target_path=target_manifest,
            expected_input_bytes=manifest_size,
            restore_command=_copy_file_command_spec(
                label=manifest_label,
                source_path=source_manifest,
                target_path=target_manifest,
                expected_input_bytes=manifest_size,
            ),
        )

        source_archive = require_non_empty_file(
            drive_artifacts.processed_sample_archive(split),
            label=f"Drive {split} processed sample archive",
        )
        archive_size = source_archive.stat().st_size
        archive_label = f"{split} processed sample archive"
        processed_sample_archives[split] = BaseProcessedSampleArchiveRestoreSpec(
            split=split,
            label=archive_label,
            archive_path=source_archive,
            data_root=runtime_layout.data_root,
            extraction_root=runtime_artifacts.processed_v1_samples_root,
            expected_split_root=runtime_artifacts.processed_samples_split_root(split),
            target_manifest_path=target_manifest,
            expected_input_bytes=archive_size,
            extract_command=_extract_processed_sample_archive_command_spec(
                label=archive_label,
                archive_path=source_archive,
                extraction_root=runtime_artifacts.processed_v1_samples_root,
                split=split,
                expected_input_bytes=archive_size,
            ),
        )
    if run_mode_policy.run_mode == "complete":
        source_quality_train = require_non_empty_file(
            drive_artifacts.processed_tier_manifest_path(
                "quality", training_config.data.train_split
            ),
            label="Drive quality train tier manifest",
        )
        target_quality_train = runtime_artifacts.processed_tier_manifest_path(
            "quality",
            training_config.data.train_split,
        )
        quality_size = source_quality_train.stat().st_size
        processed_manifest_files["quality_train"] = BaseProcessedManifestRestoreSpec(
            split=training_config.data.train_split,
            label="quality train tier manifest",
            source_path=source_quality_train,
            target_path=target_quality_train,
            expected_input_bytes=quality_size,
            restore_command=_copy_file_command_spec(
                label="quality train tier manifest",
                source_path=source_quality_train,
                target_path=target_quality_train,
                expected_input_bytes=quality_size,
            ),
        )

    return BaseRuntimePlan(
        project_root=runtime_layout.root,
        drive_project_root=drive_layout.root,
        run_name=run_name,
        train_split=training_config.data.train_split,
        validation_split=training_config.data.val_split,
        prediction_splits=workflow_config.prediction_splits,
        required_splits=required_splits,
        run_mode=run_mode_policy.run_mode,
        limit_train_samples=workflow_config.limit_train_samples,
        limit_validation_samples=workflow_config.limit_validation_samples,
        limit_prediction_samples=workflow_config.limit_prediction_samples,
        epoch_count=workflow_config.epoch_count,
        min_epochs=workflow_config.min_epochs,
        early_stopping_patience=workflow_config.early_stopping_patience,
        training_surface="quality" if run_mode_policy.run_mode == "complete" else "broad",
        validation_surface="broad/grouped" if run_mode_policy.run_mode == "complete" else "broad",
        live_log_path=run_layout.training_live_log_path,
        resume_supported=True,
        periodic_persistence=run_mode_policy.run_mode == "complete",
        shuffle_train=workflow_config.shuffle_train,
        run_qualitative_export=workflow_config.run_qualitative_export,
        panel_size=workflow_config.panel_size,
        run_layout=run_layout,
        expected_publish_run_root=drive_layout.base_m0_run_root(run_name),
        processed_manifest_files=processed_manifest_files,
        processed_sample_archives=processed_sample_archives,
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
        failure=f"Failed to restore {label}: {source_path} -> {target_path}",
    )


def _extract_processed_sample_archive_command_spec(
    *,
    label: str,
    archive_path: Path,
    extraction_root: Path,
    split: str,
    expected_input_bytes: int,
) -> CommandSpec:
    member_prefix = (Path("data") / "processed" / "v1" / "samples" / split).as_posix()
    return build_tar_zstd_extract_command(
        label=label,
        archive_path=archive_path,
        extraction_root=extraction_root,
        mkdir_path=extraction_root,
        expected_input_bytes=expected_input_bytes,
        failure=f"Failed to extract {label} into {extraction_root}: {archive_path}",
        strip_components=4,
        members=(member_prefix,),
    )


__all__ = ["build_base_runtime_plan"]
