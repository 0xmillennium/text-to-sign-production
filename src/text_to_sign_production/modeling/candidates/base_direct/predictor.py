"""Checkpoint selection and existing-predictor loading for base_direct."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.modeling.candidates import (
    ModelStageExecutionContext,
    ModelStageExecutionError,
)
from text_to_sign_production.modeling.candidates.base_direct.config import BaseDirectRunConfig


@dataclass(frozen=True, slots=True)
class BaseDirectPredictorBundle:
    compatibility_config_path: Path
    checkpoint_path: Path
    model: object
    device: object


def select_base_direct_checkpoint(context: ModelStageExecutionContext) -> Path:
    """Select the best materialized checkpoint, falling back to the last epoch."""

    checkpoints = context.topology.models
    best = checkpoints.model_checkpoint_file(
        "base_direct",
        context.request.run_name,
        "best.pt",
    ).path
    if best.is_file():
        return best
    last = checkpoints.model_checkpoint_file(
        "base_direct",
        context.request.run_name,
        "last.pt",
    ).path
    if last.is_file():
        return last
    raise ModelStageExecutionError(
        "base_direct cannot generate pose without best.pt or last.pt checkpoint."
    )


def load_base_direct_predictor(
    *,
    config: BaseDirectRunConfig,
    compatibility_config_path: Path,
    checkpoint_path: Path,
    validate_config_paths: bool = True,
) -> BaseDirectPredictorBundle:
    """Load the existing M0 predictor without using its legacy output writer."""

    if not compatibility_config_path.is_file():
        raise ModelStageExecutionError(
            f"base_direct compatibility config does not exist: {compatibility_config_path}"
        )
    try:
        from text_to_sign_production.modeling.inference.predict import load_baseline_predictor
        from text_to_sign_production.modeling.training.config import (
            load_baseline_training_config,
        )
    except ModuleNotFoundError as exc:
        raise ModelStageExecutionError(
            "base_direct generation requires the modeling torch/transformers dependencies."
        ) from exc
    try:
        legacy_config = load_baseline_training_config(
            compatibility_config_path,
            checkpoint_output_dir=checkpoint_path.parent,
            validate_paths=validate_config_paths,
        )
        predictor = load_baseline_predictor(
            legacy_config,
            checkpoint_path=checkpoint_path,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise ModelStageExecutionError(f"base_direct predictor load failed: {exc}") from exc
    return BaseDirectPredictorBundle(
        compatibility_config_path=compatibility_config_path,
        checkpoint_path=checkpoint_path,
        model=predictor.model,
        device=predictor.device,
    )


__all__ = [
    "BaseDirectPredictorBundle",
    "load_base_direct_predictor",
    "select_base_direct_checkpoint",
]
