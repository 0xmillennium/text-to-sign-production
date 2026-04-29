"""Reusable train loop for the Sprint 3 baseline model."""

from __future__ import annotations

import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from text_to_sign_production.core.progress import iter_with_progress
from text_to_sign_production.modeling.backbones import FlanT5TextBackbone
from text_to_sign_production.modeling.data import (
    SPRINT3_TARGET_CHANNELS,
    ProcessedPoseBatch,
    ProcessedPoseDataset,
    collate_processed_pose_samples,
)
from text_to_sign_production.modeling.models import BaselineTextToPoseModel

from .checkpointing import (
    RUN_SUMMARY_FILENAME,
    CheckpointMetrics,
    ensure_checkpoint_output_dir,
    save_training_checkpoint,
    should_replace_best_checkpoint,
    write_run_summary,
)
from .config import (
    BaselineTrainingConfig,
    baseline_config_to_dict,
    load_baseline_training_config,
)
from .evaluate import (
    ValidationEpochResult,
    count_valid_contributing_frames,
    move_batch_to_device,
    run_validation_epoch,
)
from .losses import masked_pose_mse_loss


@dataclass(frozen=True, slots=True)
class TrainingStepResult:
    """Scalar outputs from one optimization step."""

    loss: float
    valid_frame_count: int


@dataclass(frozen=True, slots=True)
class TrainingEpochResult:
    """Aggregated training outputs for one epoch."""

    loss: float
    valid_frame_count: int


@dataclass(frozen=True, slots=True)
class BaselineTrainingRunResult:
    """Paths and final scalar outputs from one baseline training run."""

    summary_path: Path
    last_checkpoint_path: Path
    best_checkpoint_path: Path | None
    final_train_loss: float
    final_validation_loss: float
    final_validation_metric: float
    best_validation_loss: float | None
    best_epoch: int | None


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
    """Build the default modular Sprint 3 baseline model from config."""

    backbone = FlanT5TextBackbone(
        model_name=config.backbone.name,
        max_length=config.backbone.max_length,
        local_files_only=config.backbone.local_files_only,
        freeze=config.backbone.freeze,
    )
    return BaselineTextToPoseModel(
        backbone,
        decoder_hidden_dim=config.model.decoder_hidden_dim,
        latent_dim=config.model.latent_dim,
    )


def build_optimizer(config: BaselineTrainingConfig, model: nn.Module) -> Optimizer:
    """Build the configured optimizer for trainable baseline parameters."""

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("Baseline model has no trainable parameters.")
    if config.optimizer.name == "adamw":
        return torch.optim.AdamW(
            trainable_parameters,
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer.name!r}")


def train_step(
    model: nn.Module,
    batch: ProcessedPoseBatch,
    optimizer: Optimizer,
    *,
    valid_frame_count: int | None = None,
) -> TrainingStepResult:
    """Run one masked-regression optimization step."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    predictions = model(batch)
    loss = masked_pose_mse_loss(
        predictions=predictions,
        targets=batch,
        padding_mask=batch.padding_mask,
        frame_valid_mask=batch.frame_valid_mask,
    )
    torch.autograd.backward(loss)
    optimizer.step()

    return TrainingStepResult(
        loss=float(loss.detach().cpu().item()),
        valid_frame_count=(
            count_valid_contributing_frames(batch)
            if valid_frame_count is None
            else valid_frame_count
        ),
    )


def run_training_epoch(
    model: nn.Module,
    batches: Iterable[ProcessedPoseBatch],
    optimizer: Optimizer,
    *,
    device: torch.device,
    non_blocking: bool = False,
    progress_label: str = "",
    progress_total: int | None = None,
) -> TrainingEpochResult:
    """Run one training epoch and aggregate masked training loss."""

    total_loss = 0.0
    total_valid_frames = 0
    for batch in iter_with_progress(
        batches,
        total=progress_total,
        desc=progress_label,
        unit="batch",
    ):
        device_batch = move_batch_to_device(batch, device, non_blocking=non_blocking)
        valid_frame_count = count_valid_contributing_frames(device_batch)
        if valid_frame_count == 0:
            continue

        result = train_step(
            model,
            device_batch,
            optimizer,
            valid_frame_count=valid_frame_count,
        )
        total_loss += result.loss * result.valid_frame_count
        total_valid_frames += result.valid_frame_count

    if total_valid_frames == 0:
        raise ValueError("Training epoch has zero valid contributing frames.")

    return TrainingEpochResult(
        loss=total_loss / total_valid_frames,
        valid_frame_count=total_valid_frames,
    )


def run_baseline_training(
    config_path: Path | str,
    *,
    checkpoint_output_dir: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> BaselineTrainingRunResult:
    """Run a config-driven Sprint 3 baseline train/val experiment."""

    config = load_baseline_training_config(
        config_path,
        checkpoint_output_dir=checkpoint_output_dir,
        repo_root=repo_root,
    )
    set_reproducibility_seed(config.training.seed)
    device = resolve_training_device(config.training.device)
    checkpoint_dir = ensure_checkpoint_output_dir(config.checkpoint.output_dir)

    train_dataset = ProcessedPoseDataset(config.data.train_manifest, split=config.data.train_split)
    val_dataset = ProcessedPoseDataset(config.data.val_manifest, split=config.data.val_split)
    train_loader = _build_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle_train,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.persistent_workers,
        prefetch_factor=config.training.prefetch_factor,
        seed=config.training.seed,
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
    )

    model = build_baseline_model(config).to(device=device)
    optimizer = build_optimizer(config, model)
    config_summary = baseline_config_to_dict(config)

    best_validation_loss: float | None = None
    best_epoch: int | None = None
    best_checkpoint_path: Path | None = None
    last_checkpoint_path = checkpoint_dir / "last.pt"
    history: list[dict[str, Any]] = []
    final_train_result: TrainingEpochResult | None = None
    final_validation_result: ValidationEpochResult | None = None

    for epoch in range(1, config.training.epochs + 1):
        epoch_start_time = time.perf_counter()
        print(f"[baseline train] epoch {epoch}/{config.training.epochs} start")
        train_result = run_training_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            non_blocking=config.training.non_blocking_transfers,
            progress_label=f"[baseline train] epoch {epoch}/{config.training.epochs}",
            progress_total=len(train_loader),
        )
        validation_result = run_validation_epoch(
            model,
            val_loader,
            device=device,
            non_blocking=config.training.non_blocking_transfers,
            progress_label=f"[baseline val] epoch {epoch}/{config.training.epochs}",
            progress_total=len(val_loader),
        )
        final_train_result = train_result
        final_validation_result = validation_result
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result.loss,
                "validation_loss": validation_result.loss,
                "validation_metric": validation_result.metric,
            }
        )

        metrics = CheckpointMetrics(
            train_loss=train_result.loss,
            validation_loss=validation_result.loss,
            validation_metric=validation_result.metric,
        )
        save_training_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            role="last",
            config_summary=config_summary,
            backbone_name=config.backbone.name,
            seed=config.training.seed,
            metrics=metrics,
        )
        best_checkpoint_updated = should_replace_best_checkpoint(
            candidate_validation_loss=validation_result.loss,
            best_validation_loss=best_validation_loss,
        )
        if best_checkpoint_updated:
            best_validation_loss = validation_result.loss
            best_epoch = epoch
            best_checkpoint_path = checkpoint_dir / "best.pt"
            save_training_checkpoint(
                best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                role="best",
                config_summary=config_summary,
                backbone_name=config.backbone.name,
                seed=config.training.seed,
                metrics=metrics,
            )
        elapsed_seconds = time.perf_counter() - epoch_start_time
        print(
            f"[baseline train] epoch {epoch}/{config.training.epochs} summary: "
            f"train_loss={train_result.loss:.6g} "
            f"validation_loss={validation_result.loss:.6g} "
            f"validation_metric={validation_result.metric:.6g} "
            f"elapsed_seconds={elapsed_seconds:.1f} "
            f"best_checkpoint_updated={'yes' if best_checkpoint_updated else 'no'}"
        )

    if final_train_result is None or final_validation_result is None:
        raise RuntimeError("Baseline training did not run any epochs.")

    summary_path = checkpoint_dir / RUN_SUMMARY_FILENAME
    write_run_summary(
        summary_path,
        {
            "config_path": config.source_path.as_posix(),
            "config": config_summary,
            "backbone_name": config.backbone.name,
            "seed": config.training.seed,
            "checkpoint_output_path": checkpoint_dir.as_posix(),
            "last_checkpoint_path": last_checkpoint_path.as_posix(),
            "best_checkpoint_path": (
                None if best_checkpoint_path is None else best_checkpoint_path.as_posix()
            ),
            "final_train_loss": final_train_result.loss,
            "final_validation_loss": final_validation_result.loss,
            "final_validation_metric": final_validation_result.metric,
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "target_channels": list(SPRINT3_TARGET_CHANNELS),
            "history": history,
        },
    )

    return BaselineTrainingRunResult(
        summary_path=summary_path,
        last_checkpoint_path=last_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        final_train_loss=final_train_result.loss,
        final_validation_loss=final_validation_result.loss,
        final_validation_metric=final_validation_result.metric,
        best_validation_loss=best_validation_loss,
        best_epoch=best_epoch,
    )


def _build_dataloader(
    dataset: ProcessedPoseDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
    seed: int | None,
) -> DataLoader[ProcessedPoseBatch]:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    dataloader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "collate_fn": collate_processed_pose_samples,
        "generator": generator,
    }
    if prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(cast(Any, dataset), **dataloader_kwargs)
