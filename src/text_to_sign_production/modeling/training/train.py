"""Reusable train loop for the M0 direct text-to-full-BFH baseline model."""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader, Sampler, Subset

from text_to_sign_production.core.files import sha256_json
from text_to_sign_production.core.progress import (
    BatchProgress,
    ProgressReporter,
    StdoutProgressReporter,
)
from text_to_sign_production.modeling.backbones import FlanT5TextBackbone
from text_to_sign_production.modeling.data import (
    M0_TARGET_CHANNELS,
    ProcessedPoseBatch,
    ProcessedPoseDataset,
    ProcessedPoseItem,
    collate_processed_pose_samples,
)
from text_to_sign_production.modeling.models import BaselineTextToPoseModel

from .checkpointing import (
    CheckpointMetrics,
    require_checkpoint_output_dir,
    save_training_checkpoint,
    should_replace_best_checkpoint,
    write_run_summary,
)
from .config import (
    BaselineTrainingConfig,
    baseline_config_to_dict,
    baseline_config_with_data_overrides,
    baseline_config_with_training_overrides,
    load_baseline_training_config,
)
from .evaluate import (
    ValidationEpochResult,
    count_valid_contributing_frames,
    move_batch_to_device,
    run_validation_epoch,
)
from .losses import channel_balanced_masked_pose_mse_loss
from .standardization import (
    TargetStandardization,
    standardize_batch_targets,
)

TRAINING_METRICS_FILENAME = "metrics.jsonl"
TRAINING_SUMMARY_FILENAME = "summary.json"


@dataclass(frozen=True, slots=True)
class TrainingStepResult:
    """Scalar outputs from one optimization step."""

    loss: float
    valid_frame_count: int
    channel_losses: dict[str, float]
    channel_valid_point_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class TrainingEpochResult:
    """Aggregated training outputs for one epoch."""

    loss: float
    valid_frame_count: int
    channel_losses: dict[str, float]
    channel_valid_point_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class MixedPrecisionPolicy:
    enabled: bool
    dtype: torch.dtype | None
    scaler_enabled: bool
    name: str


@dataclass(frozen=True, slots=True)
class TrainingEpochArtifacts:
    """Epoch-local artifact paths emitted after each training epoch.

    The modeling layer writes runtime-local checkpoints, metrics, and live-log
    files.  It does *not* copy, compress, or publish them.  Instead it calls
    an optional ``on_epoch_artifacts`` callback so that the workflow layer can
    decide recovery-persistence policy.
    """

    epoch: int
    run_mode: str | None
    metrics_path: Path
    live_log_path: Path
    last_checkpoint_path: Path
    best_checkpoint_path: Path | None
    best_checkpoint_updated: bool
    checkpoint_manifest_payload: Mapping[str, object]
    best_metric_name: str
    best_metric_value: float | None
    best_epoch: int | None


@dataclass(frozen=True, slots=True)
class BaselineTrainingRunResult:
    """Paths and final scalar outputs from one baseline training run."""

    summary_path: Path
    metrics_path: Path
    last_checkpoint_path: Path
    best_checkpoint_path: Path | None
    final_train_loss: float
    final_validation_loss: float
    final_validation_metric: float
    best_validation_loss: float | None
    best_metric_name: str
    best_metric_value: float | None
    best_epoch: int | None
    completed_epoch: int
    early_stopping_state: Mapping[str, Any]
    target_standardization_path: Path | None
    live_log_path: Path
    config_hash: str
    model_metadata: Mapping[str, Any]
    train_sample_count: int
    validation_sample_count: int
    history: tuple[Mapping[str, Any], ...]


def set_reproducibility_seed(seed: int | None) -> None:
    """Seed local random number generators used by the baseline training loop."""

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_training_device(device_policy: str) -> torch.device:
    """Resolve the configured device policy for a baseline run."""

    if device_policy == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_policy)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"Configured CUDA device is not available: {device_policy}")
    return device


def build_baseline_model(config: BaselineTrainingConfig) -> BaselineTextToPoseModel:
    """Build the default M0 direct full-BFH baseline model from config."""

    backbone = FlanT5TextBackbone(
        model_name=config.backbone.name,
        revision=config.backbone.revision,
        max_length=config.backbone.max_length,
        local_files_only=config.backbone.local_files_only,
        trainable=config.backbone.trainable,
        freeze_strategy=config.backbone.freeze_strategy,
    )
    return BaselineTextToPoseModel(
        backbone,
        decoder_hidden_dim=config.model.decoder_hidden_dim,
        decoder_layers=config.model.decoder_layers,
        decoder_dropout=config.model.decoder_dropout,
        frame_position_encoding_dim=config.model.frame_position_encoding_dim,
    )


def build_optimizer(config: BaselineTrainingConfig, model: nn.Module) -> Optimizer:
    """Build the configured optimizer for trainable baseline parameters."""

    encoder_parameters: list[nn.Parameter] = []
    decoder_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            encoder_parameters.append(parameter)
        else:
            decoder_parameters.append(parameter)
    if not encoder_parameters and not decoder_parameters:
        raise ValueError("Baseline model has no trainable parameters.")
    if config.optimizer.name == "adamw":
        parameter_groups: list[dict[str, Any]] = []
        if encoder_parameters:
            parameter_groups.append(
                {
                    "params": encoder_parameters,
                    "lr": config.backbone.encoder_learning_rate,
                    "name": "text_encoder",
                }
            )
        if decoder_parameters:
            parameter_groups.append(
                {
                    "params": decoder_parameters,
                    "lr": config.optimizer.decoder_learning_rate,
                    "name": "decoder_heads",
                }
            )
        return torch.optim.AdamW(
            parameter_groups,
            weight_decay=config.optimizer.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer.name!r}")


def train_step(
    model: nn.Module,
    batch: ProcessedPoseBatch,
    optimizer: Optimizer,
    *,
    valid_frame_count: int | None = None,
    channel_weights: Mapping[str, float] | None = None,
) -> TrainingStepResult:
    """Run one masked-regression optimization step."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    predictions = model(batch)
    loss_result = channel_balanced_masked_pose_mse_loss(
        predictions=predictions,
        targets=batch,
        padding_mask=batch.padding_mask,
        frame_valid_mask=batch.frame_valid_mask,
        channel_weights=channel_weights,
    )
    torch.autograd.backward(loss_result.total_loss)
    optimizer.step()

    return TrainingStepResult(
        loss=float(loss_result.total_loss.detach().cpu().item()),
        valid_frame_count=(
            count_valid_contributing_frames(batch)
            if valid_frame_count is None
            else valid_frame_count
        ),
        channel_losses={
            channel: float(loss.detach().cpu().item())
            for channel, loss in loss_result.channel_losses.items()
        },
        channel_valid_point_counts=loss_result.detached_valid_point_counts(),
    )


def run_training_epoch(
    model: nn.Module,
    batches: Iterable[ProcessedPoseBatch],
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    scaler: Any | None,
    *,
    device: torch.device,
    non_blocking: bool = False,
    progress_label: str = "",
    progress_total: int | None = None,
    progress_reporter: ProgressReporter | None = None,
    progress_interval_batches: int = 100,
    channel_weights: Mapping[str, float] | None = None,
    target_standardization: TargetStandardization | None = None,
    mixed_precision: MixedPrecisionPolicy | None = None,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float | None = None,
) -> TrainingEpochResult:
    """Run one training epoch and aggregate masked training loss."""

    total_loss = 0.0
    total_valid_frames = 0
    channel_loss_sums = {channel: 0.0 for channel in M0_TARGET_CHANNELS}
    channel_valid_point_counts = {channel: 0 for channel in M0_TARGET_CHANNELS}
    progress = BatchProgress(
        label=progress_label or "train batch",
        total=progress_total,
        unit="batches",
        reporter=progress_reporter or StdoutProgressReporter(),
        interval=progress_interval_batches,
    )
    optimizer.zero_grad(set_to_none=True)
    pending_steps = 0
    for batch_index, batch in enumerate(batches, start=1):
        device_batch = move_batch_to_device(batch, device, non_blocking=non_blocking)
        valid_frame_count = count_valid_contributing_frames(device_batch)
        if valid_frame_count == 0:
            progress.advance()
            continue
        targets = standardize_batch_targets(device_batch, target_standardization)
        model.train()
        with _autocast_context(device, mixed_precision):
            raw_predictions = model(device_batch)
            predictions = raw_predictions
            loss_result = channel_balanced_masked_pose_mse_loss(
                predictions=predictions,
                targets=targets,
                padding_mask=device_batch.padding_mask,
                frame_valid_mask=device_batch.frame_valid_mask,
                channel_weights=channel_weights,
            )
            scaled_loss = loss_result.total_loss / gradient_accumulation_steps
        if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()  # type: ignore[no-untyped-call]
        pending_steps += 1
        should_step = pending_steps >= gradient_accumulation_steps or batch_index == (
            progress_total or batch_index
        )
        if should_step:
            if max_grad_norm is not None and max_grad_norm > 0:
                if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pending_steps = 0

        result = TrainingStepResult(
            loss=float(loss_result.total_loss.detach().cpu().item()),
            valid_frame_count=valid_frame_count,
            channel_losses={
                channel: float(loss.detach().cpu().item())
                for channel, loss in loss_result.channel_losses.items()
            },
            channel_valid_point_counts=loss_result.detached_valid_point_counts(),
        )
        total_loss += result.loss * result.valid_frame_count
        total_valid_frames += result.valid_frame_count
        for channel in M0_TARGET_CHANNELS:
            valid_point_count = result.channel_valid_point_counts[channel]
            channel_valid_point_counts[channel] += valid_point_count
            channel_loss_sums[channel] += result.channel_losses[channel] * valid_point_count
        progress.advance(
            loss=f"{total_loss / max(1, total_valid_frames):.6g}",
        )

    if total_valid_frames == 0:
        raise ValueError("Training epoch has zero valid contributing frames.")
    progress.finish(loss=f"{total_loss / total_valid_frames:.6g}")

    return TrainingEpochResult(
        loss=total_loss / total_valid_frames,
        valid_frame_count=total_valid_frames,
        channel_losses=_averaged_channel_losses(
            channel_loss_sums,
            channel_valid_point_counts,
        ),
        channel_valid_point_counts=channel_valid_point_counts,
    )


def run_baseline_training(
    config_path: Path | str,
    *,
    checkpoint_output_dir: Path | str | None = None,
    training_output_dir: Path | str | None = None,
    repo_root: Path | str | None = None,
    run_mode: str | None = None,
    run_mode_statement: str | None = None,
    limit_train_samples: int | None = None,
    limit_validation_samples: int | None = None,
    epoch_count: int | None = None,
    min_epochs: int | None = None,
    early_stopping_patience: int | None = None,
    shuffle_train: bool | None = None,
    resume: bool = False,
    on_epoch_artifacts: Callable[[TrainingEpochArtifacts], None] | None = None,
    training_surface: str = "broad",
    validation_surface: str = "broad",
    quality_tier_config_path: Path | str | None = None,
    quality_tier_config_hash: str | None = None,
    quality_tier_counts: Mapping[str, Mapping[str, int]] | None = None,
    validation_tier_by_sample_id: Mapping[str, str] | None = None,
    train_manifest_override: Path | str | None = None,
    val_manifest_override: Path | str | None = None,
    path_formatter: Callable[[Path], str],
) -> BaselineTrainingRunResult:
    """Run a config-driven M0 baseline train/val experiment."""

    config = load_baseline_training_config(
        config_path,
        checkpoint_output_dir=checkpoint_output_dir,
        repo_root=repo_root,
    )
    config = baseline_config_with_data_overrides(
        config,
        train_manifest=None if train_manifest_override is None else Path(train_manifest_override),
        val_manifest=None if val_manifest_override is None else Path(val_manifest_override),
    )
    effective_epoch_count = _positive_override(
        epoch_count,
        label="epoch_count",
        default=config.training.epochs,
    )
    resolved_limit_train_samples = _positive_optional_limit(
        limit_train_samples,
        label="limit_train_samples",
    )
    resolved_limit_validation_samples = _positive_optional_limit(
        limit_validation_samples,
        label="limit_validation_samples",
    )
    effective_shuffle_train = (
        config.training.shuffle_train if shuffle_train is None else shuffle_train
    )
    effective_config = baseline_config_with_training_overrides(
        config,
        epochs=effective_epoch_count,
        min_epochs=min_epochs,
        early_stopping_patience=early_stopping_patience,
        shuffle_train=effective_shuffle_train,
    )
    config = effective_config
    set_reproducibility_seed(config.training.seed)
    device = resolve_training_device(config.training.device)
    checkpoint_dir = require_checkpoint_output_dir(config.checkpoint.output_dir)
    if training_output_dir is None:
        raise ValueError("training_output_dir is required and must be prepared by the workflow.")
    training_dir = Path(training_output_dir).expanduser().resolve()
    if not training_dir.is_dir():
        raise FileNotFoundError(f"Training output directory does not exist: {training_dir}")
    metrics_path = training_dir / TRAINING_METRICS_FILENAME
    summary_path = training_dir / TRAINING_SUMMARY_FILENAME
    live_log_path = training_dir / "live.log"
    reporter = StdoutProgressReporter(prefix="[baseline]", log_path=live_log_path)
    if not resume:
        metrics_path.write_text("", encoding="utf-8")
        live_log_path.write_text("", encoding="utf-8")
    reporter.report(
        "run start",
        run_mode=run_mode,
        training_surface=training_surface,
        validation_surface=validation_surface,
    )

    train_dataset = _limited_dataset(
        ProcessedPoseDataset(config.data.train_manifest, split=config.data.train_split),
        resolved_limit_train_samples,
    )
    val_dataset = _limited_dataset(
        ProcessedPoseDataset(config.data.val_manifest, split=config.data.val_split),
        resolved_limit_validation_samples,
    )
    train_sample_count = len(train_dataset)
    validation_sample_count = len(val_dataset)
    train_loader = _build_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=effective_shuffle_train,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.persistent_workers,
        prefetch_factor=config.training.prefetch_factor,
        seed=config.training.seed,
        length_bucketed=config.training.length_bucketed_batching,
    )
    val_loader = _build_dataloader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.persistent_workers,
        prefetch_factor=config.training.prefetch_factor,
        seed=None,
        length_bucketed=config.training.length_bucketed_batching,
    )

    target_standardization = (
        fit_target_standardization(
            train_dataset,
            epsilon=config.target_standardization.epsilon,
            reporter=reporter,
        )
        if config.target_standardization.enabled
        else None
    )
    target_standardization_path = training_dir / "target-standardization.json"
    if target_standardization is not None:
        _write_json(target_standardization_path, target_standardization.to_dict())
    model = build_baseline_model(config).to(device=device)
    model_metadata = _model_metadata(model)
    optimizer = build_optimizer(config, model)
    total_training_steps = _total_optimizer_steps(
        batches_per_epoch=len(train_loader),
        epochs=config.training.epochs,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    )
    scheduler = build_scheduler(config, optimizer, total_training_steps=total_training_steps)
    mixed_precision = resolve_mixed_precision(config.training.mixed_precision, device)
    scaler = _build_grad_scaler(mixed_precision)
    config_summary = baseline_config_to_dict(effective_config, path_formatter=path_formatter)
    config_hash = sha256_json(config_summary)

    best_validation_loss: float | None = None
    best_metric_value: float | None = None
    best_epoch: int | None = None
    best_checkpoint_path: Path | None = None
    last_checkpoint_path = checkpoint_dir / "last.pt"
    history: list[dict[str, Any]] = []
    final_train_result: TrainingEpochResult | None = None
    final_validation_result: ValidationEpochResult | None = None
    start_epoch = 1
    if resume:
        resume_state = _resume_from_last_checkpoint(
            last_checkpoint_path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_config_hash=config_hash,
            expected_run_mode=run_mode,
            reporter=reporter,
        )
        start_epoch = resume_state["start_epoch"]
        best_metric_value = resume_state["best_metric"]
        best_validation_loss = resume_state["best_metric"]
        best_epoch = resume_state["best_epoch"]
        history = resume_state["history"]
        best_checkpoint_path = checkpoint_dir / "best.pt" if best_epoch is not None else None
    patience_epochs = 0
    early_stopping_state: dict[str, Any] = {
        "enabled": True,
        "reason": None,
        "patience": config.training.early_stopping_patience,
        "min_epochs": config.training.min_epochs,
        "metric": config.training.early_stopping_metric,
        "mode": config.training.early_stopping_mode,
    }

    for epoch in range(start_epoch, effective_epoch_count + 1):
        epoch_start_time = time.perf_counter()
        reporter.report(f"epoch {epoch}/{effective_epoch_count} start")
        train_result = run_training_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device=device,
            non_blocking=config.training.non_blocking_transfers,
            progress_label=f"epoch {epoch}/{effective_epoch_count} train batch",
            progress_total=len(train_loader),
            progress_reporter=reporter,
            progress_interval_batches=config.training.progress_interval_batches,
            channel_weights=config.loss.channel_weights,
            target_standardization=target_standardization,
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            max_grad_norm=config.training.max_grad_norm,
        )
        validation_result = run_validation_epoch(
            model,
            val_loader,
            device=device,
            non_blocking=config.training.non_blocking_transfers,
            progress_label=f"epoch {epoch}/{effective_epoch_count} val batch",
            progress_total=len(val_loader),
            progress_reporter=reporter,
            progress_interval_batches=config.training.progress_interval_batches,
            channel_weights=config.loss.channel_weights,
            tier_by_sample_id=validation_tier_by_sample_id,
            target_standardization=target_standardization,
        )
        final_train_result = train_result
        final_validation_result = validation_result
        learning_rate = _current_learning_rate(optimizer)
        epoch_metrics = _epoch_metrics_record(
            epoch=epoch,
            train_result=train_result,
            validation_result=validation_result,
            learning_rate=learning_rate,
            train_sample_count=train_sample_count,
            validation_sample_count=validation_sample_count,
        )
        history.append(epoch_metrics)
        _append_jsonl_record(metrics_path, epoch_metrics)

        metrics = CheckpointMetrics(
            train_loss=train_result.loss,
            validation_loss=validation_result.loss,
            validation_metric=validation_result.metric,
            metric_name=config.training.early_stopping_metric,
        )
        save_training_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            role="last",
            config_summary=config_summary,
            config_hash=config_hash,
            backbone_name=config.backbone.name,
            model_revision=_resolved_model_revision(model_metadata),
            seed=config.training.seed,
            metrics=metrics,
            run_mode=run_mode,
            best_metric=best_metric_value,
            best_epoch=best_epoch,
            target_standardization=(
                None if target_standardization is None else target_standardization.to_dict()
            ),
        )
        best_checkpoint_updated = should_replace_best_checkpoint(
            candidate_validation_loss=validation_result.metric,
            best_validation_loss=best_metric_value,
        )
        if best_checkpoint_updated:
            best_validation_loss = validation_result.metric
            best_metric_value = validation_result.metric
            best_epoch = epoch
            best_checkpoint_path = checkpoint_dir / "best.pt"
            save_training_checkpoint(
                best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                role="best",
                config_summary=config_summary,
                config_hash=config_hash,
                backbone_name=config.backbone.name,
                model_revision=_resolved_model_revision(model_metadata),
                seed=config.training.seed,
                metrics=metrics,
                run_mode=run_mode,
                best_metric=best_metric_value,
                best_epoch=best_epoch,
                target_standardization=(
                    None if target_standardization is None else target_standardization.to_dict()
                ),
            )
            patience_epochs = 0
        else:
            patience_epochs += 1
        save_training_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            role="last",
            config_summary=config_summary,
            config_hash=config_hash,
            backbone_name=config.backbone.name,
            model_revision=_resolved_model_revision(model_metadata),
            seed=config.training.seed,
            metrics=metrics,
            run_mode=run_mode,
            best_metric=best_metric_value,
            best_epoch=best_epoch,
            target_standardization=(
                None if target_standardization is None else target_standardization.to_dict()
            ),
        )
        elapsed_seconds = time.perf_counter() - epoch_start_time
        reporter.report(
            f"epoch {epoch}/{effective_epoch_count} summary",
            train_loss=f"{train_result.loss:.6g}",
            validation_loss=f"{validation_result.loss:.6g}",
            validation_metric=f"{validation_result.metric:.6g}",
            elapsed_seconds=f"{elapsed_seconds:.1f}",
            best_checkpoint_updated="yes" if best_checkpoint_updated else "no",
        )
        if on_epoch_artifacts is not None:
            epoch_artifacts = TrainingEpochArtifacts(
                epoch=epoch,
                run_mode=run_mode,
                metrics_path=metrics_path,
                live_log_path=live_log_path,
                last_checkpoint_path=last_checkpoint_path,
                best_checkpoint_path=best_checkpoint_path if best_checkpoint_updated else None,
                best_checkpoint_updated=best_checkpoint_updated,
                checkpoint_manifest_payload={
                    "epoch": epoch,
                    "config_hash": config_hash,
                    "train_loss": train_result.loss,
                    "validation_metric": validation_result.metric,
                    "best_metric": best_metric_value,
                    "best_epoch": best_epoch,
                },
                best_metric_name=config.training.early_stopping_metric,
                best_metric_value=best_metric_value,
                best_epoch=best_epoch,
            )
            on_epoch_artifacts(epoch_artifacts)
        if (
            epoch >= config.training.min_epochs
            and patience_epochs >= config.training.early_stopping_patience
        ):
            early_stopping_state["reason"] = "patience_exhausted"
            early_stopping_state["stopped_epoch"] = epoch
            reporter.report("early stopping", epoch=epoch, patience=patience_epochs)
            break

    if final_train_result is None or final_validation_result is None:
        raise RuntimeError("Baseline training did not run any epochs.")
    completed_epoch = int(history[-1]["epoch"]) if history else 0
    if early_stopping_state["reason"] is None:
        early_stopping_state["reason"] = "max_epochs_completed"
        early_stopping_state["stopped_epoch"] = completed_epoch

    write_run_summary(
        summary_path,
        {
            "run_mode": run_mode,
            "run_mode_statement": run_mode_statement,
            "config_path": path_formatter(config.source_path),
            "config": config_summary,
            "backbone_name": config.backbone.name,
            "model_metadata": dict(model_metadata),
            "seed": config.training.seed,
            "runtime_options": {
                "run_mode": run_mode,
                "max_epochs": effective_epoch_count,
                "min_epochs": config.training.min_epochs,
                "early_stopping_patience": config.training.early_stopping_patience,
                "early_stopping_metric": config.training.early_stopping_metric,
                "limit_train_samples": resolved_limit_train_samples,
                "limit_validation_samples": resolved_limit_validation_samples,
                "shuffle_train": effective_shuffle_train,
                "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
                "max_grad_norm": config.training.max_grad_norm,
                "mixed_precision": mixed_precision.name,
                "scheduler": {
                    "name": config.scheduler.name,
                    "warmup_ratio": config.scheduler.warmup_ratio,
                },
                "dataloader": {
                    "num_workers": config.training.num_workers,
                    "pin_memory": config.training.pin_memory,
                    "non_blocking_transfers": config.training.non_blocking_transfers,
                    "persistent_workers": config.training.persistent_workers,
                    "prefetch_factor": config.training.prefetch_factor,
                    "length_bucketed_batching": config.training.length_bucketed_batching,
                },
            },
            "surfaces": {
                "training_surface": training_surface,
                "validation_surface": validation_surface,
                "quality_tier_config_path": (
                    None
                    if quality_tier_config_path is None
                    else path_formatter(Path(quality_tier_config_path))
                ),
                "quality_tier_config_hash": quality_tier_config_hash,
                "quality_tier_counts": (
                    {} if quality_tier_counts is None else dict(quality_tier_counts)
                ),
            },
            "training_output_path": path_formatter(training_dir),
            "metrics_path": path_formatter(metrics_path),
            "live_log_path": path_formatter(live_log_path),
            "checkpoint_output_path": path_formatter(checkpoint_dir),
            "last_checkpoint_path": path_formatter(last_checkpoint_path),
            "best_checkpoint_path": (
                None if best_checkpoint_path is None else path_formatter(best_checkpoint_path)
            ),
            "final_train_loss": final_train_result.loss,
            "final_validation_loss": final_validation_result.loss,
            "final_validation_metric": final_validation_result.metric,
            "primary_metric_name": config.training.early_stopping_metric,
            "best_validation_loss": best_validation_loss,
            "best_metric_name": config.training.early_stopping_metric,
            "best_metric_value": best_metric_value,
            "best_epoch": best_epoch,
            "early_stopping": early_stopping_state,
            "train_sample_count": train_sample_count,
            "validation_sample_count": validation_sample_count,
            "target_standardization": (
                None if target_standardization is None else target_standardization.to_dict()
            ),
            "target_standardization_path": (
                None
                if target_standardization is None
                else path_formatter(target_standardization_path)
            ),
            "target_channels": list(M0_TARGET_CHANNELS),
            "history": history,
            "metric_limitation_note": (
                "Automatic numeric losses and keypoint distances are not sufficient evidence "
                "of sign intelligibility or task-solving quality."
            ),
        },
    )

    return BaselineTrainingRunResult(
        summary_path=summary_path,
        metrics_path=metrics_path,
        last_checkpoint_path=last_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        final_train_loss=final_train_result.loss,
        final_validation_loss=final_validation_result.loss,
        final_validation_metric=final_validation_result.metric,
        best_validation_loss=best_validation_loss,
        best_metric_name=config.training.early_stopping_metric,
        best_metric_value=best_metric_value,
        best_epoch=best_epoch,
        completed_epoch=completed_epoch,
        early_stopping_state=early_stopping_state,
        target_standardization_path=(
            target_standardization_path if target_standardization is not None else None
        ),
        live_log_path=live_log_path,
        config_hash=config_hash,
        model_metadata=model_metadata,
        train_sample_count=train_sample_count,
        validation_sample_count=validation_sample_count,
        history=tuple(history),
    )


def _build_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
    seed: int | None,
    length_bucketed: bool,
) -> DataLoader[ProcessedPoseBatch]:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    dataloader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "collate_fn": collate_processed_pose_samples,
        "generator": generator,
    }
    if length_bucketed:
        dataloader_kwargs["batch_sampler"] = LengthBucketBatchSampler(
            lengths=_dataset_lengths(dataset),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )
    else:
        dataloader_kwargs["batch_size"] = batch_size
        dataloader_kwargs["shuffle"] = shuffle
    if prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(cast(Any, dataset), **dataloader_kwargs)


def _limited_dataset(
    dataset: ProcessedPoseDataset,
    limit: int | None,
) -> ProcessedPoseDataset | Subset[ProcessedPoseItem]:
    if limit is None:
        return dataset
    return Subset(cast(Any, dataset), range(min(limit, len(dataset))))


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Batch sampler that groups similar sequence lengths to reduce padding."""

    def __init__(
        self,
        *,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool,
        seed: int | None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.lengths = tuple(int(length) for length in lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(range(len(self.lengths)))
        indices.sort(key=lambda index: (self.lengths[index], index))
        bucket_size = max(self.batch_size * 50, self.batch_size)
        buckets = [
            indices[index : index + bucket_size] for index in range(0, len(indices), bucket_size)
        ]
        rng = random.Random(None if self.seed is None else self.seed + self._epoch)
        self._epoch += 1
        if self.shuffle:
            for bucket in buckets:
                rng.shuffle(bucket)
            rng.shuffle(buckets)
        for bucket in buckets:
            for index in range(0, len(bucket), self.batch_size):
                yield bucket[index : index + self.batch_size]

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)


def _dataset_lengths(dataset: Any) -> tuple[int, ...]:
    if isinstance(dataset, ProcessedPoseDataset):
        return tuple(record.num_frames for record in dataset.records)
    if isinstance(dataset, Subset):
        base_lengths = _dataset_lengths(dataset.dataset)
        return tuple(base_lengths[int(index)] for index in dataset.indices)
    return tuple(int(dataset[index].length) for index in range(len(dataset)))


def _positive_optional_limit(value: int | None, *, label: str) -> int | None:
    if value is None:
        return None
    if value <= 0:
        raise ValueError(f"{label} must be positive when set.")
    return value


def _positive_override(value: int | None, *, label: str, default: int) -> int:
    if value is None:
        return default
    if value <= 0:
        raise ValueError(f"{label} must be positive when set.")
    return value


def _current_learning_rate(optimizer: Optimizer) -> float | None:
    if not optimizer.param_groups:
        return None
    learning_rate = optimizer.param_groups[0].get("lr")
    if isinstance(learning_rate, int | float):
        return float(learning_rate)
    return None


def build_scheduler(
    config: BaselineTrainingConfig,
    optimizer: Optimizer,
    *,
    total_training_steps: int,
) -> LRScheduler:
    """Build cosine decay with linear warmup."""

    warmup_steps = max(1, int(total_training_steps * config.scheduler.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_training_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)


def resolve_mixed_precision(policy: str, device: torch.device) -> MixedPrecisionPolicy:
    """Resolve hardware-aware AMP policy."""

    if policy == "off" or device.type != "cuda":
        return MixedPrecisionPolicy(enabled=False, dtype=None, scaler_enabled=False, name="off")
    if policy == "bf16" or (policy == "auto" and torch.cuda.is_bf16_supported()):
        return MixedPrecisionPolicy(
            enabled=True,
            dtype=torch.bfloat16,
            scaler_enabled=False,
            name="bf16",
        )
    if policy in {"auto", "fp16"}:
        return MixedPrecisionPolicy(
            enabled=True,
            dtype=torch.float16,
            scaler_enabled=True,
            name="fp16",
        )
    raise ValueError(f"Unsupported mixed precision policy: {policy!r}")


def _build_grad_scaler(policy: MixedPrecisionPolicy) -> Any | None:
    if not policy.scaler_enabled:
        return None
    return torch.cuda.amp.GradScaler(enabled=True)


def _autocast_context(device: torch.device, policy: MixedPrecisionPolicy | None) -> Any:
    if policy is None or not policy.enabled or policy.dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=policy.dtype)


def _total_optimizer_steps(
    *,
    batches_per_epoch: int,
    epochs: int,
    gradient_accumulation_steps: int,
) -> int:
    return max(1, math.ceil(batches_per_epoch / gradient_accumulation_steps) * epochs)


def fit_target_standardization(
    dataset: ProcessedPoseDataset | Subset[ProcessedPoseItem],
    *,
    epsilon: float,
    reporter: ProgressReporter,
) -> TargetStandardization:
    """Fit per-channel scalar mean/std from supervised train target points."""

    sums = {channel: 0.0 for channel in M0_TARGET_CHANNELS}
    square_sums = {channel: 0.0 for channel in M0_TARGET_CHANNELS}
    counts = {channel: 0 for channel in M0_TARGET_CHANNELS}
    progress = BatchProgress(
        label="target standardization sample",
        total=len(dataset),
        unit="samples",
        reporter=reporter,
        interval=100,
    )
    for item in cast(Iterable[Any], dataset):
        for channel in M0_TARGET_CHANNELS:
            values = torch.as_tensor(getattr(item, channel), dtype=torch.float32)
            confidence = torch.as_tensor(
                getattr(item, f"{channel}_confidence"), dtype=torch.float32
            )
            mask = torch.isfinite(confidence) & (confidence > 0.0)
            if not bool(mask.any().item()):
                continue
            selected = values[mask]
            sums[channel] += float(selected.sum().item())
            square_sums[channel] += float(selected.square().sum().item())
            counts[channel] += int(selected.numel())
        progress.advance()
    progress.finish()
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for channel in M0_TARGET_CHANNELS:
        if counts[channel] <= 0:
            means[channel] = 0.0
            stds[channel] = 1.0
            continue
        mean = sums[channel] / counts[channel]
        variance = max(0.0, square_sums[channel] / counts[channel] - mean * mean)
        means[channel] = mean
        stds[channel] = max(math.sqrt(variance), epsilon)
    return TargetStandardization(mean_by_channel=means, std_by_channel=stds, epsilon=epsilon)


def _resume_from_last_checkpoint(
    *,
    last_checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    scaler: Any | None,
    expected_config_hash: str,
    expected_run_mode: str | None,
    reporter: ProgressReporter,
) -> dict[str, Any]:
    from .checkpointing import load_training_checkpoint

    if not last_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Resume requested but last checkpoint is missing: {last_checkpoint_path}"
        )
    payload = load_training_checkpoint(last_checkpoint_path, map_location="cpu")
    if payload["config_hash"] != expected_config_hash:
        raise ValueError("Resume checkpoint config_hash does not match the current config.")
    if payload["run_mode"] != expected_run_mode:
        raise ValueError("Resume checkpoint run_mode does not match the current run mode.")
    model.load_state_dict(cast(Mapping[str, Any], payload["model_state_dict"]), strict=True)
    optimizer.load_state_dict(cast(dict[str, Any], payload["optimizer_state_dict"]))
    scheduler_state = payload.get("scheduler_state_dict")
    if scheduler is not None and isinstance(scheduler_state, Mapping):
        scheduler.load_state_dict(cast(dict[str, Any], scheduler_state))
    scaler_state = payload.get("scaler_state_dict")
    if scaler is not None and isinstance(scaler_state, Mapping):
        scaler.load_state_dict(scaler_state)
    completed_epoch = int(payload["completed_epoch"])
    reporter.report("resume", checkpoint=last_checkpoint_path, completed_epoch=completed_epoch)
    return {
        "start_epoch": completed_epoch + 1,
        "best_metric": payload.get("best_metric"),
        "best_epoch": payload.get("best_epoch"),
        "history": [],
    }


def _model_metadata(model: nn.Module) -> Mapping[str, Any]:
    backbone = getattr(model, "backbone", None)
    metadata = getattr(backbone, "metadata", None)
    if callable(metadata):
        return cast(Mapping[str, Any], metadata())
    return {}


def _resolved_model_revision(metadata: Mapping[str, Any]) -> str | None:
    value = metadata.get("resolved_revision") or metadata.get("requested_revision")
    return value if isinstance(value, str) else None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _epoch_metrics_record(
    *,
    epoch: int,
    train_result: TrainingEpochResult,
    validation_result: ValidationEpochResult,
    learning_rate: float | None,
    train_sample_count: int,
    validation_sample_count: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "epoch": epoch,
        "train_total_loss": train_result.loss,
        "validation_total_loss": validation_result.loss,
        "validation_masked_l2_mean": validation_result.metrics.get(
            "validation_masked_l2_mean",
            validation_result.metric,
        ),
        "validation_masked_mae": validation_result.metrics.get("validation_masked_mae"),
        "validation_masked_rmse": validation_result.metrics.get("validation_masked_rmse"),
        "validation_metric": validation_result.metric,
        "primary_metric_name": "validation_masked_l2_mean",
        "train_valid_frame_count": train_result.valid_frame_count,
        "validation_valid_frame_count": validation_result.valid_frame_count,
        "learning_rate": learning_rate,
        "train_sample_count": train_sample_count,
        "validation_sample_count": validation_sample_count,
    }
    for channel in M0_TARGET_CHANNELS:
        record[f"{channel}_loss"] = validation_result.channel_losses[channel]
        record[f"{channel}_valid_point_count"] = validation_result.channel_valid_point_counts[
            channel
        ]
        record[f"train_{channel}_loss"] = train_result.channel_losses[channel]
        record[f"train_{channel}_valid_point_count"] = train_result.channel_valid_point_counts[
            channel
        ]
        record[f"validation_{channel}_loss"] = validation_result.channel_losses[channel]
        record[f"validation_{channel}_valid_point_count"] = (
            validation_result.channel_valid_point_counts[channel]
        )
        record[f"per_channel_{channel}_masked_l2"] = validation_result.metrics.get(
            f"per_channel_{channel}_masked_l2"
        )
        record[f"per_channel_{channel}_masked_mae"] = validation_result.metrics.get(
            f"per_channel_{channel}_masked_mae"
        )
        record[f"per_channel_{channel}_valid_point_count"] = validation_result.metrics.get(
            f"per_channel_{channel}_valid_point_count"
        )
        record[f"per_channel_{channel}_valid_frame_count"] = validation_result.metrics.get(
            f"per_channel_{channel}_valid_frame_count"
        )
    record["temporal_velocity_l2_diagnostic"] = validation_result.metrics.get(
        "temporal_velocity_l2_diagnostic"
    )
    record["temporal_smoothness_diagnostic"] = validation_result.metrics.get(
        "temporal_smoothness_diagnostic"
    )
    record["validation_grouped_metrics"] = validation_result.grouped_metrics
    return record


def _averaged_channel_losses(
    channel_loss_sums: Mapping[str, float],
    channel_valid_point_counts: Mapping[str, int],
) -> dict[str, float]:
    return {
        channel: (
            channel_loss_sums[channel] / channel_valid_point_counts[channel]
            if channel_valid_point_counts[channel] > 0
            else 0.0
        )
        for channel in M0_TARGET_CHANNELS
    }


def _append_jsonl_record(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True))
        handle.write("\n")
