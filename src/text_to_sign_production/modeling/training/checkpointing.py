"""Checkpointing and run-summary helpers for M0 baseline training."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from text_to_sign_production.modeling.contracts import CONFIDENCE_POLICY, LENGTH_POLICY
from text_to_sign_production.modeling.data import M0_CHANNEL_POLICY, M0_TARGET_CHANNELS

CHECKPOINT_SCHEMA_VERSION = "t2sp-baseline-checkpoint-v2"
RUN_SUMMARY_FILENAME = "run_summary.json"


@dataclass(frozen=True, slots=True)
class CheckpointMetrics:
    """Scalar metrics stored in a baseline training checkpoint."""

    train_loss: float
    validation_loss: float
    validation_metric: float
    metric_name: str = "validation_masked_l2_mean"


def require_checkpoint_output_dir(output_dir: Path) -> Path:
    """Require an existing checkpoint output directory."""

    if not output_dir.exists():
        raise FileNotFoundError(f"Checkpoint output directory does not exist: {output_dir}")
    if not output_dir.is_dir():
        raise NotADirectoryError(
            f"Checkpoint output path exists and is not a directory: {output_dir}"
        )
    return output_dir.resolve()


def should_replace_best_checkpoint(
    *,
    candidate_validation_loss: float,
    best_validation_loss: float | None,
) -> bool:
    """Return whether a candidate should replace the current best checkpoint."""

    if not math.isfinite(candidate_validation_loss):
        raise ValueError("candidate_validation_loss must be finite.")
    return best_validation_loss is None or candidate_validation_loss < best_validation_loss


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    scaler: Any | None,
    epoch: int,
    role: str,
    config_summary: Mapping[str, Any],
    config_hash: str,
    backbone_name: str,
    model_revision: str | None,
    seed: int | None,
    metrics: CheckpointMetrics,
    run_mode: str | None,
    best_metric: float | None,
    best_epoch: int | None,
    target_standardization: Mapping[str, Any] | None,
) -> Path:
    """Save an explicit M0 baseline training checkpoint payload."""

    if epoch <= 0:
        raise ValueError("checkpoint epoch must be positive.")
    if not role:
        raise ValueError("checkpoint role must not be blank.")
    if path.exists() and path.is_dir():
        raise ValueError(f"Checkpoint path is a directory: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Checkpoint parent directory does not exist: {path.parent}")

    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch,
        "role": role,
        "completed_epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "config": dict(config_summary),
        "config_hash": config_hash,
        "backbone_name": backbone_name,
        "model_revision": model_revision,
        "seed": seed,
        "run_mode": run_mode,
        "target_channels": tuple(M0_TARGET_CHANNELS),
        "channel_policy": M0_CHANNEL_POLICY,
        "length_policy": LENGTH_POLICY,
        "confidence_policy": CONFIDENCE_POLICY,
        "train_loss": metrics.train_loss,
        "validation_loss": metrics.validation_loss,
        "validation_metric": metrics.validation_metric,
        "metric_name": metrics.metric_name,
        "metric_value": metrics.validation_metric,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "target_standardization": None
        if target_standardization is None
        else dict(target_standardization),
    }
    torch.save(payload, path)
    return path


def load_training_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and validate an M0 baseline training checkpoint payload."""

    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a mapping: {path}")

    required_keys = {
        "schema_version",
        "epoch",
        "role",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "config",
        "config_hash",
        "backbone_name",
        "model_revision",
        "seed",
        "run_mode",
        "target_channels",
        "channel_policy",
        "length_policy",
        "confidence_policy",
        "train_loss",
        "validation_loss",
        "validation_metric",
        "metric_name",
        "metric_value",
        "best_metric",
        "best_epoch",
        "target_standardization",
    }
    missing_keys = sorted(required_keys.difference(payload))
    if missing_keys:
        formatted = ", ".join(missing_keys)
        raise ValueError(f"Checkpoint is missing required field(s): {formatted}.")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Checkpoint schema version mismatch: "
            f"expected {CHECKPOINT_SCHEMA_VERSION!r}, got {payload['schema_version']!r}."
        )
    return payload


def write_run_summary(path: Path, summary: Mapping[str, Any]) -> Path:
    """Write a minimal runtime-side provenance summary."""

    if not path.parent.is_dir():
        raise FileNotFoundError(f"Run summary parent directory does not exist: {path.parent}")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(summary), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path
