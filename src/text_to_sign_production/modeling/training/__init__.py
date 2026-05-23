"""Lazy exports for M0 baseline training utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .checkpointing import (
        CheckpointMetrics,
        load_training_checkpoint,
        require_checkpoint_output_dir,
        save_training_checkpoint,
        should_replace_best_checkpoint,
        write_run_summary,
    )
    from .checkpoints import (
        CHECKPOINT_METADATA_SCHEMA_VERSION,
        BestCheckpointSelection,
        CheckpointRole,
        ModelCheckpointMetadata,
        read_checkpoint_metadata_json,
        select_best_checkpoint,
        write_checkpoint_metadata_json,
    )
    from .config import (
        BaselineIdentityConfig,
        BaselineLossConfig,
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
    from .losses import (
        ChannelBalancedPoseLoss,
        channel_balanced_masked_pose_mse_loss,
        masked_pose_mse_loss,
    )
    from .masking import build_effective_frame_mask
    from .metrics import (
        TRAINING_METRIC_SCHEMA_VERSION,
        TrainingMetricRecord,
        read_training_metric_records_jsonl,
        write_training_metric_records_jsonl,
    )
    from .pose_tensor_metrics import masked_average_keypoint_l2_error
    from .results import TRAINING_RESULT_SCHEMA_VERSION, TrainingRunSummary
    from .stages import ProviderStageArtifactRole
    from .train import (
        BaselineTrainingRunResult,
        TrainingEpochArtifacts,
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
    "BaselineIdentityConfig",
    "BaselineLossConfig",
    "BaselineTrainingRunResult",
    "BestCheckpointSelection",
    "CHECKPOINT_METADATA_SCHEMA_VERSION",
    "ChannelBalancedPoseLoss",
    "CheckpointMetrics",
    "CheckpointRole",
    "ModelCheckpointMetadata",
    "ProviderStageArtifactRole",
    "TRAINING_METRIC_SCHEMA_VERSION",
    "TRAINING_RESULT_SCHEMA_VERSION",
    "TrainingEpochArtifacts",
    "TrainingEpochResult",
    "TrainingStepResult",
    "TrainingMetricRecord",
    "TrainingRunSummary",
    "ValidationEpochResult",
    "ValidationStepResult",
    "baseline_config_to_dict",
    "build_baseline_model",
    "build_optimizer",
    "build_effective_frame_mask",
    "count_valid_contributing_frames",
    "load_baseline_training_config",
    "load_training_checkpoint",
    "channel_balanced_masked_pose_mse_loss",
    "masked_average_keypoint_l2_error",
    "masked_pose_mse_loss",
    "move_batch_to_device",
    "require_checkpoint_output_dir",
    "read_checkpoint_metadata_json",
    "read_training_metric_records_jsonl",
    "resolve_training_device",
    "run_baseline_training",
    "run_training_epoch",
    "run_validation_epoch",
    "save_training_checkpoint",
    "select_best_checkpoint",
    "set_reproducibility_seed",
    "should_replace_best_checkpoint",
    "train_step",
    "validation_step",
    "write_checkpoint_metadata_json",
    "write_run_summary",
    "write_training_metric_records_jsonl",
]


def __getattr__(name: str) -> Any:
    if name in {
        "CHECKPOINT_METADATA_SCHEMA_VERSION",
        "BestCheckpointSelection",
        "CheckpointRole",
        "ModelCheckpointMetadata",
        "read_checkpoint_metadata_json",
        "select_best_checkpoint",
        "write_checkpoint_metadata_json",
    }:
        from . import checkpoints

        return getattr(checkpoints, name)
    if name in {
        "CheckpointMetrics",
        "load_training_checkpoint",
        "require_checkpoint_output_dir",
        "save_training_checkpoint",
        "should_replace_best_checkpoint",
        "write_run_summary",
    }:
        from . import checkpointing

        return getattr(checkpointing, name)
    if name in {
        "BaselineTrainingConfig",
        "BaselineTrainingConfigError",
        "BaselineIdentityConfig",
        "BaselineLossConfig",
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
    if name in {
        "ChannelBalancedPoseLoss",
        "channel_balanced_masked_pose_mse_loss",
        "masked_pose_mse_loss",
    }:
        from . import losses

        return getattr(losses, name)
    if name in {
        "TRAINING_METRIC_SCHEMA_VERSION",
        "TrainingMetricRecord",
        "read_training_metric_records_jsonl",
        "write_training_metric_records_jsonl",
    }:
        from . import metrics

        return getattr(metrics, name)
    if name == "masked_average_keypoint_l2_error":
        from .pose_tensor_metrics import masked_average_keypoint_l2_error

        return masked_average_keypoint_l2_error
    if name in {"TRAINING_RESULT_SCHEMA_VERSION", "TrainingRunSummary"}:
        from . import results

        return getattr(results, name)
    if name == "ProviderStageArtifactRole":
        from .stages import ProviderStageArtifactRole

        return ProviderStageArtifactRole
    if name in {
        "BaselineTrainingRunResult",
        "TrainingEpochArtifacts",
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
