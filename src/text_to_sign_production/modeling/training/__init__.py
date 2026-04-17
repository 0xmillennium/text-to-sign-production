"""Lazy exports for Sprint 3 baseline training utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .checkpointing import (
        CheckpointMetrics,
        ensure_checkpoint_output_dir,
        load_training_checkpoint,
        save_training_checkpoint,
        should_replace_best_checkpoint,
        write_run_summary,
    )
    from .config import (
        BaselineTrainingConfig,
        BaselineTrainingConfigError,
        baseline_config_to_dict,
        load_baseline_training_config,
    )
    from .evaluate import (
        ValidationEpochResult,
        ValidationStepResult,
        count_valid_contributing_frames,
        move_batch_to_device,
        run_validation_epoch,
        validation_step,
    )
    from .losses import masked_pose_mse_loss
    from .masking import build_effective_frame_mask
    from .metrics import masked_average_keypoint_l2_error
    from .train import (
        BaselineTrainingRunResult,
        TrainingEpochResult,
        TrainingStepResult,
        build_baseline_model,
        build_optimizer,
        resolve_training_device,
        run_baseline_training,
        run_training_epoch,
        set_reproducibility_seed,
        train_step,
    )

__all__ = [
    "BaselineTrainingConfig",
    "BaselineTrainingConfigError",
    "BaselineTrainingRunResult",
    "CheckpointMetrics",
    "TrainingEpochResult",
    "TrainingStepResult",
    "ValidationEpochResult",
    "ValidationStepResult",
    "baseline_config_to_dict",
    "build_baseline_model",
    "build_optimizer",
    "build_effective_frame_mask",
    "count_valid_contributing_frames",
    "ensure_checkpoint_output_dir",
    "load_baseline_training_config",
    "load_training_checkpoint",
    "masked_average_keypoint_l2_error",
    "masked_pose_mse_loss",
    "move_batch_to_device",
    "resolve_training_device",
    "run_baseline_training",
    "run_training_epoch",
    "run_validation_epoch",
    "save_training_checkpoint",
    "set_reproducibility_seed",
    "should_replace_best_checkpoint",
    "train_step",
    "validation_step",
    "write_run_summary",
]


def __getattr__(name: str) -> Any:
    if name in {
        "CheckpointMetrics",
        "ensure_checkpoint_output_dir",
        "load_training_checkpoint",
        "save_training_checkpoint",
        "should_replace_best_checkpoint",
        "write_run_summary",
    }:
        from . import checkpointing

        return getattr(checkpointing, name)
    if name in {
        "BaselineTrainingConfig",
        "BaselineTrainingConfigError",
        "baseline_config_to_dict",
        "load_baseline_training_config",
    }:
        from . import config

        return getattr(config, name)
    if name in {
        "ValidationEpochResult",
        "ValidationStepResult",
        "count_valid_contributing_frames",
        "move_batch_to_device",
        "run_validation_epoch",
        "validation_step",
    }:
        from . import evaluate

        return getattr(evaluate, name)
    if name == "build_effective_frame_mask":
        from .masking import build_effective_frame_mask

        return build_effective_frame_mask
    if name == "masked_pose_mse_loss":
        from .losses import masked_pose_mse_loss

        return masked_pose_mse_loss
    if name == "masked_average_keypoint_l2_error":
        from .metrics import masked_average_keypoint_l2_error

        return masked_average_keypoint_l2_error
    if name in {
        "BaselineTrainingRunResult",
        "TrainingEpochResult",
        "TrainingStepResult",
        "build_baseline_model",
        "build_optimizer",
        "resolve_training_device",
        "run_baseline_training",
        "run_training_epoch",
        "set_reproducibility_seed",
        "train_step",
    }:
        from . import train

        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
