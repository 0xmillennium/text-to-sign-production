"""Base workflow execution orchestration."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from text_to_sign_production.artifacts.layout import DatasetArtifactLayout
from text_to_sign_production.core.files import (
    ensure_dir,
    prepare_output_dir,
    require_file,
    sha256_file,
    verify_output_file,
)
from text_to_sign_production.core.paths import ProjectLayout
from text_to_sign_production.core.progress import StdoutProgressReporter
from text_to_sign_production.data.jsonl import iter_jsonl
from text_to_sign_production.modeling.inference.qualitative import (
    PREDICTION_ARTIFACTS_DIRNAME,
    REFERENCE_ARTIFACTS_DIRNAME,
    QualitativeExportResult,
    export_qualitative_panel,
)
from text_to_sign_production.modeling.inference.writer import (
    SplitPredictionWriteResult,
    write_split_predictions,
)
from text_to_sign_production.modeling.training.config import (
    baseline_config_with_data_overrides,
    baseline_config_with_training_overrides,
    load_baseline_training_config,
)
from text_to_sign_production.modeling.training.train import run_baseline_training
from text_to_sign_production.workflows._base.layout import (
    _config_path,
    _layout_from_config,
    build_base_run_layout,
)
from text_to_sign_production.workflows._base.recovery import build_epoch_recovery_callback
from text_to_sign_production.workflows._base.reports import (
    _project_relative_manifest_path_formatter,
    _sample_limits,
    write_base_config_snapshot,
    write_base_run_summary,
    write_baseline_report,
    write_failure_mode_report,
)
from text_to_sign_production.workflows._base.types import (
    BaseRunLayout,
    BaseWorkflowConfig,
    BaseWorkflowError,
    BaseWorkflowInputError,
    BaseWorkflowResult,
)
from text_to_sign_production.workflows._base.validate import (
    _output_policy,
    _prediction_splits,
    _run_mode_policy,
    _workflow_config_with_run_mode_policy,
    validate_base_inputs,
)
from text_to_sign_production.workflows._base.verify import verify_base_run_outputs


def run_base_workflow(config: BaseWorkflowConfig) -> BaseWorkflowResult:
    """Run M0 training, full split prediction export, reports, and verification."""

    config = _workflow_config_with_run_mode_policy(config)
    validate_base_inputs(config)
    layout = _layout_from_config(config)
    artifact_layout = DatasetArtifactLayout(layout)
    config_path = _config_path(config, layout)
    run_layout = build_base_run_layout(project_root=layout.root, run_name=config.run_name)
    resolved_output_policy = _output_policy(config.output_policy)
    run_root = prepare_output_dir(
        run_layout.run_root,
        policy=resolved_output_policy,
        label="Base run root",
    )
    _create_run_subdirectories(run_layout)

    try:
        path_formatter = _project_relative_manifest_path_formatter(project_root=layout.root)
        training_config = load_baseline_training_config(
            config_path,
            validate_paths=config.validate_processed_inputs,
            checkpoint_output_dir=run_layout.checkpoints_dir,
            repo_root=layout.root,
        )
        surface_info = _surface_info(
            config=config,
            layout=layout,
            artifact_layout=artifact_layout,
            train_split=training_config.data.train_split,
            val_split=training_config.data.val_split,
        )
        if surface_info["train_manifest_override"] is not None:
            training_config = baseline_config_with_data_overrides(
                training_config,
                train_manifest=cast(Path, surface_info["train_manifest_override"]),
            )
        effective_training_config = baseline_config_with_training_overrides(
            training_config,
            epochs=config.epoch_count,
            min_epochs=config.min_epochs,
            early_stopping_patience=config.early_stopping_patience,
            shuffle_train=config.shuffle_train,
        )
        prediction_splits = _prediction_splits(config.prediction_splits, effective_training_config)
        _copy_baseline_config(config_path, run_layout.baseline_config_copy_path)
        write_base_config_snapshot(
            run_layout.config_snapshot_path,
            effective_training_config,
            workflow_config=config,
            run_layout=run_layout,
            prediction_splits=prediction_splits,
            project_root=layout.root,
        )
        training_result = run_baseline_training(
            config_path,
            checkpoint_output_dir=run_layout.checkpoints_dir,
            training_output_dir=run_layout.training_dir,
            repo_root=layout.root,
            run_mode=config.run_mode,
            run_mode_statement=_run_mode_policy(config.run_mode).run_mode_statement,
            limit_train_samples=config.limit_train_samples,
            limit_validation_samples=config.limit_validation_samples,
            epoch_count=config.epoch_count,
            min_epochs=config.min_epochs,
            early_stopping_patience=config.early_stopping_patience,
            shuffle_train=config.shuffle_train,
            resume=resolved_output_policy.value == "resume",
            on_epoch_artifacts=build_epoch_recovery_callback(
                recovery_root=_periodic_publish_run_root(config, run_layout.run_name),
                reporter=StdoutProgressReporter(prefix="[baseline]"),
            ),
            training_surface=str(surface_info["training_surface"]),
            validation_surface=str(surface_info["validation_surface"]),
            quality_tier_config_path=cast(Path | None, surface_info["quality_tier_config_path"]),
            quality_tier_config_hash=cast(str | None, surface_info["quality_tier_config_hash"]),
            quality_tier_counts=cast(
                dict[str, dict[str, int]],
                surface_info["quality_tier_counts"],
            ),
            validation_tier_by_sample_id=cast(
                dict[str, str],
                surface_info["validation_tier_by_sample_id"],
            ),
            train_manifest_override=cast(Path | None, surface_info["train_manifest_override"]),
            path_formatter=path_formatter,
        )

        checkpoint_path = (
            training_result.best_checkpoint_path or training_result.last_checkpoint_path
        )
        _create_prediction_output_directories(run_layout, prediction_splits=prediction_splits)
        prediction_results: dict[str, SplitPredictionWriteResult] = {}
        for split in prediction_splits:
            prediction_results[split] = write_split_predictions(
                effective_training_config,
                run_name=config.run_name,
                split=split,
                manifest_path=artifact_layout.processed_manifest_path(split),
                output_dir=run_layout.prediction_split_dir(split),
                checkpoint_path=checkpoint_path,
                data_root=layout.data_root,
                limit_prediction_samples=config.limit_prediction_samples,
                manifest_path_formatter=path_formatter,
            )

        write_baseline_report(
            run_layout=run_layout,
            config=effective_training_config,
            training_result=training_result,
            prediction_results=prediction_results,
            config_path=config_path,
            workflow_config=config,
            sample_limits=_sample_limits(config),
            git_revision=config.git_revision,
        )
        write_failure_mode_report(run_layout)

        qualitative_result: QualitativeExportResult | None = None
        if config.run_qualitative_export:
            qualitative_output_dir = _create_qualitative_output_directories(run_layout)
            qualitative_result = export_qualitative_panel(
                config_path,
                output_dir=qualitative_output_dir,
                checkpoint_path=checkpoint_path,
                checkpoint_output_dir=run_layout.checkpoints_dir,
                training_summary_path=training_result.summary_path,
                run_summary_path=run_layout.run_summary_path,
                panel_size=config.panel_size,
                repo_root=layout.root,
                path_formatter=path_formatter,
                progress_reporter=StdoutProgressReporter(prefix="[baseline]"),
            )

        write_base_run_summary(
            run_layout=run_layout,
            config=effective_training_config,
            training_result=training_result,
            prediction_results=prediction_results,
            qualitative_result=qualitative_result,
            workflow_config=config,
            path_formatter=path_formatter,
        )
        verification = verify_base_run_outputs(
            run_layout=run_layout,
            prediction_splits=prediction_splits,
            limit_prediction_samples=config.limit_prediction_samples,
            qualitative_result=qualitative_result,
        )
    except BaseWorkflowInputError:
        raise
    except Exception as exc:
        raise BaseWorkflowError(f"Baseline workflow failed: {exc}") from exc

    return BaseWorkflowResult(
        config=effective_training_config,
        training=training_result,
        predictions=prediction_results,
        qualitative=qualitative_result,
        run_root=run_root,
        run_layout=run_layout,
        config_snapshot_path=run_layout.config_snapshot_path,
        baseline_config_copy_path=run_layout.baseline_config_copy_path,
        baseline_report_json_path=run_layout.baseline_report_json_path,
        baseline_report_markdown_path=run_layout.baseline_report_markdown_path,
        failure_modes_json_path=run_layout.failure_modes_json_path,
        failure_modes_markdown_path=run_layout.failure_modes_markdown_path,
        run_summary_path=run_layout.run_summary_path,
        verification=verification,
    )


def _create_run_subdirectories(run_layout: BaseRunLayout) -> None:
    for path, label in (
        (run_layout.config_dir, "Base config directory"),
        (run_layout.training_dir, "Base training directory"),
        (run_layout.checkpoints_dir, "Base checkpoints directory"),
        (run_layout.predictions_dir, "Base predictions directory"),
        (run_layout.reports_dir, "Base reports directory"),
    ):
        ensure_dir(path, label=label)


def _copy_baseline_config(source_path: Path, target_path: Path) -> Path:
    require_file(source_path, label="Baseline source config")
    ensure_dir(target_path.parent, label="Base config directory")
    shutil.copyfile(source_path, target_path)
    return verify_output_file(target_path, label="Base baseline config copy")


def _create_prediction_output_directories(
    run_layout: BaseRunLayout,
    *,
    prediction_splits: Iterable[str],
) -> None:
    for split in prediction_splits:
        ensure_dir(
            run_layout.prediction_split_dir(split),
            label=f"Base {split} prediction directory",
        )
        ensure_dir(
            run_layout.prediction_samples_dir(split),
            label=f"Base {split} prediction samples directory",
        )


def _create_qualitative_output_directories(run_layout: BaseRunLayout) -> Path:
    output_dir = ensure_dir(
        run_layout.qualitative_dir,
        label="Base qualitative output directory",
    )
    ensure_dir(
        output_dir / REFERENCE_ARTIFACTS_DIRNAME,
        label="Base qualitative reference artifact directory",
    )
    ensure_dir(
        output_dir / PREDICTION_ARTIFACTS_DIRNAME,
        label="Base qualitative prediction artifact directory",
    )
    return output_dir


def _surface_info(
    *,
    config: BaseWorkflowConfig,
    layout: ProjectLayout,
    artifact_layout: DatasetArtifactLayout,
    train_split: str,
    val_split: str,
) -> dict[str, object]:
    quality_config_path = layout.root / "configs" / "data" / "quality-tier-v1.yaml"
    quality_config_hash = (
        sha256_file(quality_config_path) if quality_config_path.is_file() else None
    )
    quality_train_manifest = artifact_layout.processed_tier_manifest_path("quality", train_split)
    training_surface = "quality" if config.run_mode == "complete" else "broad"
    train_manifest_override: Path | None = None
    if training_surface == "quality":
        if not quality_train_manifest.is_file():
            raise BaseWorkflowInputError(
                "complete run_mode requires the quality train tier manifest: "
                f"{quality_train_manifest}"
            )
        train_manifest_override = quality_train_manifest
    tier_counts = _quality_tier_counts(artifact_layout)
    return {
        "training_surface": training_surface,
        "validation_surface": "broad/grouped" if config.run_mode == "complete" else "broad",
        "quality_tier_config_path": quality_config_path if quality_config_path.is_file() else None,
        "quality_tier_config_hash": quality_config_hash,
        "quality_tier_counts": tier_counts,
        "validation_tier_by_sample_id": _validation_tier_by_sample_id(
            artifact_layout,
            split=val_split,
        ),
        "train_manifest_override": train_manifest_override,
    }


def _quality_tier_counts(
    artifact_layout: DatasetArtifactLayout,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_counts: dict[str, int] = {}
        for tier in ("broad", "quality", "audit_low_quality", "dropped"):
            path = artifact_layout.processed_tier_manifest_path(tier, split)
            if path.is_file():
                split_counts[tier] = sum(1 for _ in iter_jsonl(path))
        if split_counts:
            counts[split] = split_counts
    return counts


def _validation_tier_by_sample_id(
    artifact_layout: DatasetArtifactLayout,
    *,
    split: str,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for tier in ("quality", "audit_low_quality"):
        path = artifact_layout.processed_tier_manifest_path(tier, split)
        if not path.is_file():
            continue
        for record in iter_jsonl(path):
            sample_id = record.get("sample_id")
            if isinstance(sample_id, str):
                mapping[sample_id] = tier
    return mapping


def _periodic_publish_run_root(
    config: BaseWorkflowConfig,
    run_name: str,
) -> Path | None:
    if config.drive_project_root is None:
        return None
    return ProjectLayout(Path(config.drive_project_root)).base_m0_run_root(run_name)


__all__ = ["run_base_workflow"]
