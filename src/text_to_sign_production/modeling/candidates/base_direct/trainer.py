"""Training-stage adapter from the provider contract to the existing M0 loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from text_to_sign_production.modeling.candidates import (
    ModelStageExecutionContext,
    ModelStageExecutionError,
)
from text_to_sign_production.modeling.candidates.base_direct.config import BaseDirectRunConfig
from text_to_sign_production.modeling.candidates.base_direct.dataset import (
    resolve_base_direct_runtime_data_paths,
    validate_base_direct_runtime_data_paths,
)
from text_to_sign_production.modeling.data import read_modeling_manifest
from text_to_sign_production.workflows.model.processing.training_progress import (
    ModelWorkflowTrainingProgressSink,
)


@dataclass(frozen=True, slots=True)
class BaseDirectTrainingStageResult:
    summary_path: Path
    metrics_path: Path
    live_log_path: Path
    last_checkpoint_path: Path
    best_checkpoint_path: Path | None
    target_standardization_path: Path | None
    compatibility_config_path: Path | None
    train_sample_count: int
    validation_sample_count: int
    final_train_loss: float | None
    final_validation_loss: float | None
    best_metric_name: str
    best_metric_value: float | None
    completed_epoch: int | None
    config_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "summary_path",
            "metrics_path",
            "live_log_path",
            "last_checkpoint_path",
        ):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))
        for field_name in (
            "best_checkpoint_path",
            "target_standardization_path",
            "compatibility_config_path",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, Path(value))
        _require_non_negative_count("train_sample_count", self.train_sample_count)
        _require_non_negative_count("validation_sample_count", self.validation_sample_count)
        if self.completed_epoch is not None:
            _require_non_negative_count("completed_epoch", self.completed_epoch)
        if not isinstance(self.best_metric_name, str) or not self.best_metric_name.strip():
            raise ModelStageExecutionError("best_metric_name must be non-empty.")
        if not isinstance(self.config_hash, str) or not self.config_hash.strip():
            raise ModelStageExecutionError("config_hash must be non-empty.")


def base_direct_compatibility_config_path(context: ModelStageExecutionContext) -> Path:
    """Return the provider-owned legacy loop adapter config path."""

    return (
        context.topology.models.model_intermediate_root(
            "base_direct",
            context.request.run_name,
            "config",
        ).path
        / "baseline_training_compat.yaml"
    )


def run_base_direct_training_stage(
    *,
    context: ModelStageExecutionContext,
    config: BaseDirectRunConfig,
) -> BaseDirectTrainingStageResult:
    """Train the direct baseline with restored workflow data and canonical outputs."""

    paths = resolve_base_direct_runtime_data_paths(context)
    validate_base_direct_runtime_data_paths(paths)
    train_manifest = read_modeling_manifest(
        context.topology,
        context.request.manifest_family,
        context.request.train_split,
    )
    validation_manifest = read_modeling_manifest(
        context.topology,
        context.request.manifest_family,
        context.request.validation_split,
    )
    if not train_manifest.entries:
        raise ModelStageExecutionError("base_direct training manifest contains no samples.")
    if not validation_manifest.entries:
        raise ModelStageExecutionError("base_direct validation manifest contains no samples.")

    compatibility_path = base_direct_compatibility_config_path(context)
    checkpoint_root = context.topology.models.model_checkpoints_root(
        "base_direct",
        context.request.run_name,
    ).path
    training_root = context.topology.models.model_training_root(
        "base_direct",
        context.request.run_name,
    ).path
    compatibility_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    training_root.mkdir(parents=True, exist_ok=True)
    compatibility_path.write_text(
        yaml.safe_dump(
            _compatibility_payload(config, paths.train_manifest, paths.validation_manifest),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    try:
        from text_to_sign_production.modeling.training.logging import TextTrainingRunLogSink
        from text_to_sign_production.modeling.training.train import run_baseline_training
    except ModuleNotFoundError as exc:
        raise ModelStageExecutionError(
            "base_direct training requires the modeling torch/transformers dependencies."
        ) from exc

    if context.progress_session is not None:
        context.progress_session.status(
            "base_direct run start",
            run_mode=context.request.run_mode.value,
            training_surface=context.request.manifest_family.family_id,
            validation_surface=context.request.manifest_family.family_id,
        )

    try:
        result = run_baseline_training(
            compatibility_path,
            checkpoint_output_dir=checkpoint_root,
            training_output_dir=training_root,
            run_mode=context.request.run_mode.value,
            run_mode_statement="M0 direct baseline provider execution.",
            limit_train_samples=config.data.limit_train_samples,
            limit_validation_samples=config.data.limit_validation_samples,
            epoch_count=config.training.epochs,
            min_epochs=config.training.min_epochs,
            early_stopping_patience=config.training.early_stopping_patience,
            shuffle_train=config.training.shuffle_train,
            training_surface=context.request.manifest_family.family_id,
            validation_surface=context.request.manifest_family.family_id,
            train_manifest_override=paths.train_manifest,
            val_manifest_override=paths.validation_manifest,
            path_formatter=lambda path: _runtime_path_value(context, path),
            progress_sink=ModelWorkflowTrainingProgressSink(
                progress_session=context.progress_session,
                provider_label="base_direct",
            ),
            log_sink=TextTrainingRunLogSink(
                prefix="[base_direct]",
                log_path=context.topology.models.model_training_log_file(
                    "base_direct",
                    context.request.run_name,
                ).path,
                stream=None,
            ),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise ModelStageExecutionError(f"base_direct training failed: {exc}") from exc
    return BaseDirectTrainingStageResult(
        summary_path=result.summary_path,
        metrics_path=result.metrics_path,
        live_log_path=result.live_log_path,
        last_checkpoint_path=result.last_checkpoint_path,
        best_checkpoint_path=result.best_checkpoint_path,
        target_standardization_path=result.target_standardization_path,
        compatibility_config_path=compatibility_path,
        train_sample_count=result.train_sample_count,
        validation_sample_count=result.validation_sample_count,
        final_train_loss=result.final_train_loss,
        final_validation_loss=result.final_validation_loss,
        best_metric_name=result.best_metric_name,
        best_metric_value=result.best_metric_value,
        completed_epoch=result.completed_epoch,
        config_hash=result.config_hash,
    )


def _compatibility_payload(
    config: BaseDirectRunConfig,
    train_manifest: Path,
    validation_manifest: Path,
) -> dict[str, Any]:
    """Build only the old-loop adapter fields; Stage 3 remains output authority."""

    return {
        "adapter_note": (
            "This file is an internal legacy training-loop adapter. "
            "Final generated-pose outputs use t2sp-generated-pose-v1 and "
            "t2sp-generated-pose-manifest-v1."
        ),
        "baseline": {
            "id": "m0-direct-text-to-full-bfh",
            "name": "M0 Direct Text-to-Full-BFH Baseline",
            "role": "m0_comparison_floor",
            "channels": ["body", "left_hand", "right_hand", "face"],
            "channel_policy": "full_bfh",
            "length_policy": "reference_length",
            "confidence_policy": "synthetic_validity_not_model_uncertainty",
            "prediction_schema_version": "t2sp-baseline-prediction-v1",
            "prediction_manifest_schema_version": "t2sp-baseline-prediction-manifest-v1",
        },
        "data": {
            "train_manifest": str(train_manifest),
            "val_manifest": str(validation_manifest),
            "train_split": config.data.train_split.value,
            "val_split": config.data.validation_split.value,
            "prediction_splits": [split.value for split in config.data.prediction_splits],
        },
        "text_encoder": {
            "model_name": config.text_encoder.model_name,
            "revision": config.text_encoder.revision,
            "max_length": config.text_encoder.max_length,
            "local_files_only": config.text_encoder.local_files_only,
            "trainable": config.text_encoder.trainable,
            "freeze_strategy": config.text_encoder.freeze_strategy,
            "encoder_learning_rate": config.text_encoder.encoder_learning_rate,
        },
        "model": {
            "decoder_hidden_dim": config.model.decoder_hidden_dim,
            "decoder_layers": config.model.decoder_layers,
            "decoder_dropout": config.model.decoder_dropout,
            "frame_position_encoding_dim": config.model.frame_position_encoding_dim,
        },
        "loss": {"channel_weights": dict(config.loss.channel_weights)},
        "training": {
            "epochs": config.training.epochs,
            "min_epochs": config.training.min_epochs,
            "early_stopping_patience": config.training.early_stopping_patience,
            "early_stopping_metric": config.training.early_stopping_metric,
            "early_stopping_mode": config.training.early_stopping_mode,
            "validate_every_epochs": config.training.validate_every_epochs,
            "batch_size": config.training.batch_size,
            "shuffle_train": config.training.shuffle_train,
            "num_workers": config.training.num_workers,
            "pin_memory": config.training.pin_memory,
            "persistent_workers": config.training.persistent_workers,
            "prefetch_factor": config.training.prefetch_factor,
            "non_blocking_transfers": config.training.non_blocking_transfers,
            "seed": config.training.seed,
            "device": config.training.device,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "max_grad_norm": config.training.max_grad_norm,
            "mixed_precision": config.training.mixed_precision,
            "length_bucketed_batching": config.training.length_bucketed_batching,
        },
        "optimizer": {
            "name": config.optimizer.name,
            "decoder_learning_rate": config.optimizer.decoder_learning_rate,
            "weight_decay": config.optimizer.weight_decay,
        },
        "scheduler": {
            "name": config.scheduler.name,
            "warmup_ratio": config.scheduler.warmup_ratio,
        },
        "target_standardization": {
            "enabled": config.target_standardization.enabled,
            "epsilon": config.target_standardization.epsilon,
        },
        "checkpoint": {},
    }


def _runtime_path_value(context: ModelStageExecutionContext, path: Path) -> str:
    resolved = Path(path).resolve(strict=False)
    try:
        return resolved.relative_to(context.topology.repo_root.resolve(strict=False)).as_posix()
    except ValueError:
        return str(resolved)


def _require_non_negative_count(field_name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ModelStageExecutionError(f"{field_name} must be a non-negative integer.")


__all__ = [
    "BaseDirectTrainingStageResult",
    "base_direct_compatibility_config_path",
    "run_base_direct_training_stage",
]
