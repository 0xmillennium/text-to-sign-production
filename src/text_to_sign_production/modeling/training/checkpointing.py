"""Checkpointing and run-summary helpers for Sprint 3 baseline training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from text_to_sign_production.data.utils import write_json
from text_to_sign_production.modeling.data import SPRINT3_TARGET_CHANNELS

CHECKPOINT_SCHEMA_VERSION = "t2sp-baseline-checkpoint-v1"
RUN_SUMMARY_FILENAME = "run_summary.json"


@dataclass(frozen=True, slots=True)
class CheckpointMetrics:
    """Scalar metrics stored in a baseline training checkpoint."""

    train_loss: float
    validation_loss: float
    validation_metric: float


def ensure_checkpoint_output_dir(output_dir: Path) -> Path:
    """Create a checkpoint directory or fail if the path is impossible."""

    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Checkpoint output path exists and is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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
    epoch: int,
    role: str,
    config_summary: Mapping[str, Any],
    backbone_name: str,
    seed: int | None,
    metrics: CheckpointMetrics,
) -> Path:
    """Save an explicit Sprint 3 baseline training checkpoint payload."""

    if epoch <= 0:
        raise ValueError("checkpoint epoch must be positive.")
    if not role:
        raise ValueError("checkpoint role must not be blank.")
    if path.exists() and path.is_dir():
        raise ValueError(f"Checkpoint path is a directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch,
        "role": role,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": dict(config_summary),
        "backbone_name": backbone_name,
        "seed": seed,
        "target_channels": tuple(SPRINT3_TARGET_CHANNELS),
        "train_loss": metrics.train_loss,
        "validation_loss": metrics.validation_loss,
        "validation_metric": metrics.validation_metric,
    }
    torch.save(payload, path)
    return path


def load_training_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and validate a Sprint 3 baseline training checkpoint payload."""

    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a mapping: {path}")

    required_keys = {
        "schema_version",
        "epoch",
        "role",
        "model_state_dict",
        "optimizer_state_dict",
        "config",
        "backbone_name",
        "seed",
        "target_channels",
        "train_loss",
        "validation_loss",
        "validation_metric",
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

    write_json(path, dict(summary))
    return path
