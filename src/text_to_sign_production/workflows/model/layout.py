"""Artifact-topology-backed layout for the model production workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.artifacts.store import ArtifactStores, build_artifact_stores
from text_to_sign_production.core import build_repo_roots
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.data import resolve_modeling_manifest_path
from text_to_sign_production.modeling.research import ObjectiveKey
from text_to_sign_production.workflows.model.contracts.config import ModelWorkflowConfig


@dataclass(frozen=True, slots=True)
class ModelRuntimeSplitLayout:
    split: SampleSplit
    drive_manifest_path: Path
    drive_passed_archive_path: Path
    runtime_manifest_path: Path
    runtime_passed_samples_split_root: Path


@dataclass(frozen=True, slots=True)
class ModelRuntimeLayout:
    root: Path
    model_config_original_path: Path | None
    model_config_path: Path | None
    semantic_objective_config_original_path: Path | None
    semantic_objective_config_path: Path | None
    required_splits: tuple[SampleSplit, ...]
    splits: tuple[ModelRuntimeSplitLayout, ...]


@dataclass(frozen=True, slots=True)
class ModelGeneratedPoseSplitOutputLayout:
    split: SampleSplit
    manifest_path: Path
    samples_root: Path


@dataclass(frozen=True, slots=True)
class ModelOutputLayout:
    model_run_root: Path
    effective_config_path: Path
    research_spec_path: Path
    run_metadata_path: Path
    runtime_support_manifest_path: Path
    stage_artifacts_index_path: Path
    checkpoints_root: Path
    training_root: Path
    intermediates_root: Path
    generated_pose_split_outputs: tuple[ModelGeneratedPoseSplitOutputLayout, ...]


@dataclass(frozen=True, slots=True)
class ModelReportLayout:
    root: Path
    model_run_summary_report_path: Path
    model_spec_report_path: Path
    architecture_report_path: Path
    training_summary_report_path: Path
    validation_summary_report_path: Path
    checkpoint_selection_report_path: Path
    limitations_report_path: Path
    generated_pose_summary_report_path: Path
    risk_controls_report_path: Path
    performance_summary_json_path: Path
    performance_summary_report_path: Path
    a100_readiness_summary_json_path: Path
    a100_readiness_summary_report_path: Path
    stage_artifacts_detail_json_path: Path
    training_artifact_summary_json_path: Path
    generated_pose_artifact_summary_json_path: Path
    index_json_path: Path
    validation_root: Path
    validation_pairing_manifest_path: Path
    validation_metric_results_path: Path
    validation_channel_metric_results_path: Path
    validation_aggregate_metrics_path: Path
    validation_channel_aggregate_metrics_path: Path
    validation_limitations_path: Path
    semantic_objective_root: Path


@dataclass(frozen=True, slots=True)
class ModelGeneratedPosePublishSplitLayout:
    split: SampleSplit
    runtime_root: Path
    drive_root: Path


@dataclass(frozen=True, slots=True)
class ModelPublishLayout:
    root: Path
    model_run_root: Path
    reports_root: Path
    generated_pose_run_roots: tuple[ModelGeneratedPosePublishSplitLayout, ...]


@dataclass(frozen=True, slots=True)
class ModelLayout:
    config: ModelWorkflowConfig
    stores: ArtifactStores
    runtime: ModelRuntimeLayout
    outputs: ModelOutputLayout
    reports: ModelReportLayout
    publish: ModelPublishLayout


def build_model_layout(config: ModelWorkflowConfig) -> ModelLayout:
    """Build model workflow paths from canonical topology helpers."""

    stores = build_artifact_stores(
        build_repo_roots(config.runtime_root),
        build_repo_roots(config.drive_project_root),
    )
    return ModelLayout(
        config=config,
        stores=stores,
        runtime=_build_runtime_layout(config, stores),
        outputs=_build_output_layout(config, stores),
        reports=_build_report_layout(config, stores),
        publish=_build_publish_layout(config, stores),
    )


def required_model_splits(config: ModelWorkflowConfig) -> tuple[SampleSplit, ...]:
    """Return required input splits in stable first-occurrence order."""

    splits: list[SampleSplit] = []
    for split in (config.train_split, config.validation_split, *config.prediction_splits):
        if split not in splits:
            splits.append(split)
    return tuple(splits)


def _build_runtime_layout(
    config: ModelWorkflowConfig,
    stores: ArtifactStores,
) -> ModelRuntimeLayout:
    required_splits = required_model_splits(config)
    config_original_path = (
        None
        if config.model_config_relpath is None
        else config.project_root / config.model_config_relpath
    )
    config_snapshot_path = (
        None
        if config.model_config_relpath is None
        else config.runtime_root
        / "provenance"
        / "config"
        / config.model_config_relpath.name
    )
    semantic_requested = ObjectiveKey.SEMANTIC_CONSISTENCY in config.auxiliary_objectives
    semantic_config_original_path = (
        config.project_root / "configs" / "modeling" / "objectives" / "semantic_consistency.yaml"
        if semantic_requested
        else None
    )
    semantic_config_snapshot_path = (
        config.runtime_root
        / "provenance"
        / "config"
        / "objectives"
        / "semantic_consistency.yaml"
        if semantic_requested
        else None
    )
    return ModelRuntimeLayout(
        root=config.runtime_root,
        model_config_original_path=config_original_path,
        model_config_path=config_snapshot_path,
        semantic_objective_config_original_path=semantic_config_original_path,
        semantic_objective_config_path=semantic_config_snapshot_path,
        required_splits=required_splits,
        splits=tuple(
            ModelRuntimeSplitLayout(
                split=split,
                drive_manifest_path=resolve_modeling_manifest_path(
                    stores.drive,
                    config.manifest_family,
                    split,
                ),
                drive_passed_archive_path=stores.drive.samples.split_archive(
                    "passed",
                    split,
                ).path,
                runtime_manifest_path=resolve_modeling_manifest_path(
                    stores.runtime,
                    config.manifest_family,
                    split,
                ),
                runtime_passed_samples_split_root=stores.runtime.samples.passed_split_dir(
                    split
                ).path,
            )
            for split in required_splits
        ),
    )


def _build_output_layout(
    config: ModelWorkflowConfig,
    stores: ArtifactStores,
) -> ModelOutputLayout:
    runtime_models = stores.runtime.models
    runtime_evaluations = stores.runtime.evaluations
    model_key = config.model_key.value
    return ModelOutputLayout(
        model_run_root=runtime_models.model_run_root(model_key, config.run_name).path,
        effective_config_path=runtime_models.model_effective_config_file(
            model_key,
            config.run_name,
        ).path,
        research_spec_path=runtime_models.model_research_spec_file(
            model_key,
            config.run_name,
        ).path,
        run_metadata_path=runtime_models.model_run_metadata_file(
            model_key,
            config.run_name,
        ).path,
        runtime_support_manifest_path=runtime_models.model_runtime_support_manifest_file(
            model_key,
            config.run_name,
        ).path,
        stage_artifacts_index_path=runtime_models.model_stage_artifacts_index_file(
            model_key,
            config.run_name,
        ).path,
        checkpoints_root=runtime_models.model_checkpoints_root(model_key, config.run_name).path,
        training_root=runtime_models.model_training_root(model_key, config.run_name).path,
        intermediates_root=runtime_models.model_intermediates_root(model_key, config.run_name).path,
        generated_pose_split_outputs=tuple(
            ModelGeneratedPoseSplitOutputLayout(
                split=split,
                manifest_path=runtime_evaluations.generated_pose_manifest(
                    model_key,
                    config.run_name,
                    split,
                ).path,
                samples_root=runtime_evaluations.generated_pose_samples_root(
                    model_key,
                    config.run_name,
                    split,
                ).path,
            )
            for split in config.prediction_splits
        ),
    )


def _build_report_layout(
    config: ModelWorkflowConfig,
    stores: ArtifactStores,
) -> ModelReportLayout:
    reports = stores.runtime.reports
    model_key = config.model_key.value
    return ModelReportLayout(
        root=reports.model_run_root(model_key, config.run_name).path,
        model_run_summary_report_path=reports.model_report_file(
            model_key,
            config.run_name,
            "model_run_summary.md",
        ).path,
        model_spec_report_path=reports.model_spec_report(model_key, config.run_name).path,
        architecture_report_path=reports.model_architecture_report(model_key, config.run_name).path,
        training_summary_report_path=reports.model_training_summary_report(
            model_key,
            config.run_name,
        ).path,
        validation_summary_report_path=reports.model_validation_report_file(
            model_key,
            config.run_name,
            "validation_summary.md",
        ).path,
        checkpoint_selection_report_path=reports.model_report_file(
            model_key,
            config.run_name,
            "checkpoint_selection.md",
        ).path,
        limitations_report_path=reports.model_report_file(
            model_key,
            config.run_name,
            "limitations.md",
        ).path,
        generated_pose_summary_report_path=reports.model_generated_pose_summary_report(
            model_key,
            config.run_name,
        ).path,
        risk_controls_report_path=reports.model_risk_controls_report(
            model_key,
            config.run_name,
        ).path,
        performance_summary_json_path=reports.model_report_file(
            model_key,
            config.run_name,
            "performance_summary.json",
        ).path,
        performance_summary_report_path=reports.model_report_file(
            model_key,
            config.run_name,
            "performance_summary.md",
        ).path,
        a100_readiness_summary_json_path=reports.model_report_file(
            model_key,
            config.run_name,
            "a100_readiness_summary.json",
        ).path,
        a100_readiness_summary_report_path=reports.model_report_file(
            model_key,
            config.run_name,
            "a100_readiness_summary.md",
        ).path,
        stage_artifacts_detail_json_path=reports.model_report_file(
            model_key,
            config.run_name,
            "stage_artifacts_detail.json",
        ).path,
        training_artifact_summary_json_path=reports.model_report_file(
            model_key,
            config.run_name,
            "training_artifact_summary.json",
        ).path,
        generated_pose_artifact_summary_json_path=reports.model_report_file(
            model_key,
            config.run_name,
            "generated_pose_artifact_summary.json",
        ).path,
        index_json_path=reports.model_index(model_key, config.run_name).path,
        validation_root=reports.model_validation_root(model_key, config.run_name).path,
        validation_pairing_manifest_path=reports.model_validation_report_file(
            model_key, config.run_name, "pairing_manifest.jsonl"
        ).path,
        validation_metric_results_path=reports.model_validation_report_file(
            model_key, config.run_name, "metric_results.jsonl"
        ).path,
        validation_channel_metric_results_path=reports.model_validation_report_file(
            model_key, config.run_name, "channel_metric_results.jsonl"
        ).path,
        validation_aggregate_metrics_path=reports.model_validation_report_file(
            model_key, config.run_name, "aggregate_metrics.json"
        ).path,
        validation_channel_aggregate_metrics_path=reports.model_validation_report_file(
            model_key, config.run_name, "channel_aggregate_metrics.json"
        ).path,
        validation_limitations_path=reports.model_validation_report_file(
            model_key, config.run_name, "limitations.json"
        ).path,
        semantic_objective_root=reports.model_run_root(model_key, config.run_name).path
        / "semantic_objective",
    )


def _build_publish_layout(
    config: ModelWorkflowConfig,
    stores: ArtifactStores,
) -> ModelPublishLayout:
    model_key = config.model_key.value
    return ModelPublishLayout(
        root=stores.drive.repo_root,
        model_run_root=stores.drive.models.model_run_root(model_key, config.run_name).path,
        reports_root=stores.drive.reports.model_run_root(model_key, config.run_name).path,
        generated_pose_run_roots=tuple(
            ModelGeneratedPosePublishSplitLayout(
                split=split,
                runtime_root=stores.runtime.evaluations.generated_pose_split_root(
                    model_key,
                    config.run_name,
                    split,
                ).path,
                drive_root=stores.drive.evaluations.generated_pose_split_root(
                    model_key,
                    config.run_name,
                    split,
                ).path,
            )
            for split in config.prediction_splits
        ),
    )


__all__ = [
    "ModelGeneratedPosePublishSplitLayout",
    "ModelGeneratedPoseSplitOutputLayout",
    "ModelLayout",
    "ModelOutputLayout",
    "ModelPublishLayout",
    "ModelReportLayout",
    "ModelRuntimeLayout",
    "ModelRuntimeSplitLayout",
    "build_model_layout",
    "required_model_splits",
]
