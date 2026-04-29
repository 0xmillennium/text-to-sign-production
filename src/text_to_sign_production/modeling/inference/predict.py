"""Baseline checkpoint loading and prediction helpers for M0 export."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from text_to_sign_production.modeling.data import M0_TARGET_CHANNELS, ProcessedPoseBatch
from text_to_sign_production.modeling.models import BaselinePoseOutput
from text_to_sign_production.modeling.training.checkpointing import load_training_checkpoint
from text_to_sign_production.modeling.training.config import BaselineTrainingConfig
from text_to_sign_production.modeling.training.evaluate import move_batch_to_device
from text_to_sign_production.modeling.training.standardization import (
    inverse_standardize_predictions,
)
from text_to_sign_production.modeling.training.train import (
    build_baseline_model,
    resolve_training_device,
)


class BaselinePredictionError(ValueError):
    """Raised when a baseline checkpoint cannot be used for prediction export."""


@dataclass(frozen=True, slots=True)
class LoadedBaselinePredictor:
    """A baseline model loaded from a checkpoint for inference-only use."""

    model: nn.Module
    device: torch.device
    checkpoint_path: Path
    checkpoint_payload: Mapping[str, Any]


def resolve_baseline_checkpoint_path(
    config: BaselineTrainingConfig,
    checkpoint_path: Path | str | None,
    *,
    repo_root: Path | str | None = None,
) -> Path:
    """Resolve the checkpoint path used for qualitative export.

    An explicit checkpoint path wins. If omitted, inference uses the best-checkpoint convention:
    ``checkpoint.output_dir / "best.pt"``.
    """

    resolved = (
        _resolve_path(checkpoint_path, repo_root=repo_root)
        if checkpoint_path is not None
        else (config.checkpoint.output_dir / "best.pt").resolve()
    )
    if not resolved.is_file():
        raise FileNotFoundError(f"Baseline checkpoint does not exist or is not a file: {resolved}")
    return resolved


def load_baseline_predictor(
    config: BaselineTrainingConfig,
    *,
    checkpoint_path: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> LoadedBaselinePredictor:
    """Build the configured baseline model and load a strict Phase 5 checkpoint."""

    resolved_checkpoint_path = resolve_baseline_checkpoint_path(
        config,
        checkpoint_path,
        repo_root=repo_root,
    )
    device = resolve_training_device(config.training.device)
    try:
        checkpoint_payload = load_training_checkpoint(
            resolved_checkpoint_path,
            map_location="cpu",
        )
        _validate_prediction_checkpoint(
            checkpoint_payload,
            checkpoint_path=resolved_checkpoint_path,
        )
    except BaselinePredictionError:
        raise
    except (KeyError, ValueError) as exc:
        raise BaselinePredictionError(
            "Checkpoint is invalid for baseline prediction export: "
            f"{resolved_checkpoint_path}: {exc}"
        ) from exc

    model = build_baseline_model(config)
    model_state_dict = cast(Mapping[str, Any], checkpoint_payload["model_state_dict"])
    try:
        load_result = model.load_state_dict(model_state_dict, strict=True)
    except RuntimeError as exc:
        raise BaselinePredictionError(
            "Checkpoint model_state_dict is incompatible with the configured baseline model."
        ) from exc

    missing_keys = tuple(getattr(load_result, "missing_keys", ()))
    unexpected_keys = tuple(getattr(load_result, "unexpected_keys", ()))
    if missing_keys or unexpected_keys:
        raise BaselinePredictionError(
            "Checkpoint model_state_dict is incompatible with the configured baseline model: "
            f"missing_keys={list(missing_keys)}, unexpected_keys={list(unexpected_keys)}."
        )

    model.to(device=device)
    target_standardization = checkpoint_payload.get("target_standardization")
    if isinstance(target_standardization, Mapping):
        model._target_standardization = dict(target_standardization)  # type: ignore[assignment]
    model.eval()
    return LoadedBaselinePredictor(
        model=model,
        device=device,
        checkpoint_path=resolved_checkpoint_path,
        checkpoint_payload=checkpoint_payload,
    )


def predict_baseline_batch(
    model: nn.Module,
    batch: ProcessedPoseBatch,
    *,
    device: torch.device,
) -> BaselinePoseOutput:
    """Run baseline prediction for one processed-pose batch and return CPU tensors."""

    model.eval()
    device_batch = move_batch_to_device(batch, device)
    with torch.no_grad():
        raw_predictions = model(device_batch)
    predictions = inverse_standardize_predictions(
        raw_predictions,
        checkpoint_transform(model),
    )

    return BaselinePoseOutput(
        body=_prediction_tensor(predictions, "body"),
        left_hand=_prediction_tensor(predictions, "left_hand"),
        right_hand=_prediction_tensor(predictions, "right_hand"),
        face=_prediction_tensor(predictions, "face"),
    )


def checkpoint_transform(model: nn.Module) -> Mapping[str, Any] | None:
    value = getattr(model, "_target_standardization", None)
    return value if isinstance(value, Mapping) else None


def _validate_prediction_checkpoint(
    checkpoint_payload: Mapping[str, Any],
    *,
    checkpoint_path: Path,
) -> None:
    target_channels = checkpoint_payload["target_channels"]
    if not isinstance(target_channels, tuple | list):
        raise BaselinePredictionError(
            f"Checkpoint target_channels must be a sequence: {checkpoint_path}"
        )
    if tuple(target_channels) != M0_TARGET_CHANNELS:
        raise BaselinePredictionError(
            "Checkpoint target_channels are incompatible with M0 full-BFH export: "
            f"expected {M0_TARGET_CHANNELS!r}, got {tuple(target_channels)!r}."
        )

    model_state_dict = checkpoint_payload["model_state_dict"]
    if not isinstance(model_state_dict, Mapping):
        raise BaselinePredictionError(
            f"Checkpoint model_state_dict must be a mapping: {checkpoint_path}"
        )


def _prediction_tensor(predictions: object, channel: str) -> torch.Tensor:
    if not hasattr(predictions, channel):
        raise BaselinePredictionError(f"Baseline predictions are missing channel {channel!r}.")
    value = getattr(predictions, channel)
    if not isinstance(value, torch.Tensor):
        raise BaselinePredictionError(f"Baseline prediction channel {channel!r} must be a tensor.")
    return value.detach().cpu()


def _resolve_path(path: Path | str, *, repo_root: Path | str | None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    if repo_root is not None:
        root = Path(repo_root).expanduser().resolve()
        resolved = (root / candidate).resolve()
        if not resolved.is_relative_to(root):
            raise BaselinePredictionError(f"Checkpoint path must stay under {root}: {path}")
        return resolved

    raise BaselinePredictionError(f"Relative checkpoint path requires explicit repo_root: {path}")
