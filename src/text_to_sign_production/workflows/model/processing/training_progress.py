"""Adapt baseline training progress events to workflow progress sessions."""

from __future__ import annotations

from text_to_sign_production.core.progress import ProgressSession, ProgressTaskHandle
from text_to_sign_production.modeling.training.events import (
    CheckpointSaved,
    EarlyStoppingEvaluated,
    EpochCompleted,
    MetricsWritten,
    ResumeLoaded,
    StandardizationItemProcessed,
    SummaryWritten,
    TrainingBatchProcessed,
    TrainingProgressEvent,
    ValidationBatchProcessed,
)
from text_to_sign_production.workflows.model.constants import (
    MODEL_STAGE_TRAINING_STANDARDIZATION,
    MODEL_STAGE_TRAINING_TRAIN_EPOCH,
    MODEL_STAGE_TRAINING_VAL_EPOCH,
)
from text_to_sign_production.workflows.model.progress import model_progress_stage


class ModelWorkflowTrainingProgressSink:
    """Bridge training-domain progress events to model workflow progress."""

    def __init__(
        self,
        *,
        progress_session: ProgressSession | None,
        provider_label: str,
    ) -> None:
        self._progress_session = progress_session
        self._provider_label = provider_label
        self._standardization_task: ProgressTaskHandle | None = None
        self._train_tasks: dict[int, ProgressTaskHandle] = {}
        self._val_tasks: dict[int, ProgressTaskHandle] = {}

    def emit(self, event: TrainingProgressEvent) -> None:
        if self._progress_session is None:
            return
        if isinstance(event, StandardizationItemProcessed):
            self._emit_standardization(event)
            return
        if isinstance(event, TrainingBatchProcessed):
            self._emit_training_batch(event)
            return
        if isinstance(event, ValidationBatchProcessed):
            self._emit_validation_batch(event)
            return
        if isinstance(event, EpochCompleted):
            self._emit_epoch_completed(event)
            return
        if isinstance(event, (CheckpointSaved, MetricsWritten)):
            return
        if isinstance(event, SummaryWritten):
            self._progress_session.status(
                f"{self._provider_label} training summary written",
                summary_path=event.summary_path,
            )
            return
        if isinstance(event, ResumeLoaded):
            self._progress_session.status(
                f"{self._provider_label} training resume",
                completed_epoch=event.completed_epoch,
                best_epoch=event.best_epoch,
                best_metric=event.best_metric,
            )
            return
        if isinstance(event, EarlyStoppingEvaluated) and event.should_stop:
            self._progress_session.status(
                f"{self._provider_label} early stopping",
                epoch=event.epoch_index,
                patience_epochs=event.patience_epochs,
                reason=event.reason,
            )

    def _emit_standardization(self, event: StandardizationItemProcessed) -> None:
        if self._standardization_task is None:
            self._standardization_task = self._progress_session.task(
                model_progress_stage(
                    stage_id=MODEL_STAGE_TRAINING_STANDARDIZATION,
                    label=f"{self._provider_label} target standardization",
                    unit="sample",
                    owner_module=__name__,
                    operation_kind="target_standardization",
                    total_semantics="training samples standardized",
                ),
                total=event.total_items,
            )
        self._standardization_task.advance()
        if event.item_index >= event.total_items:
            self._standardization_task.close()
            self._standardization_task = None

    def _emit_training_batch(self, event: TrainingBatchProcessed) -> None:
        task = self._train_tasks.get(event.epoch_index)
        if task is None:
            task = self._progress_session.task(
                model_progress_stage(
                    stage_id=f"{MODEL_STAGE_TRAINING_TRAIN_EPOCH}.{event.epoch_index}",
                    label=(
                        f"{self._provider_label} train epoch "
                        f"{event.epoch_index:02d}/{event.epoch_count:02d}"
                    ),
                    unit="batch",
                    owner_module=__name__,
                    operation_kind="train_epoch",
                    total_semantics="training batches in epoch",
                    allowed_counters=("loss", "running_loss", "frames", "skipped"),
                ),
                total=event.batch_count,
            )
            self._train_tasks[event.epoch_index] = task
        task.advance(1, counters=_training_batch_counters(event))
        if event.batch_count is not None and event.batch_index >= event.batch_count:
            task.close()
            self._train_tasks.pop(event.epoch_index, None)

    def _emit_validation_batch(self, event: ValidationBatchProcessed) -> None:
        task = self._val_tasks.get(event.epoch_index)
        if task is None:
            task = self._progress_session.task(
                model_progress_stage(
                    stage_id=f"{MODEL_STAGE_TRAINING_VAL_EPOCH}.{event.epoch_index}",
                    label=(
                        f"{self._provider_label} val epoch "
                        f"{event.epoch_index:02d}/{event.epoch_count:02d}"
                    ),
                    unit="batch",
                    owner_module=__name__,
                    operation_kind="val_epoch",
                    total_semantics="validation batches in epoch",
                    allowed_counters=(
                        "loss",
                        "metric",
                        "running_loss",
                        "frames",
                        "points",
                        "skipped",
                    ),
                ),
                total=event.batch_count,
            )
            self._val_tasks[event.epoch_index] = task
        task.advance(1, counters=_validation_batch_counters(event))
        if event.batch_count is not None and event.batch_index >= event.batch_count:
            task.close()
            self._val_tasks.pop(event.epoch_index, None)

    def _emit_epoch_completed(self, event: EpochCompleted) -> None:
        self._progress_session.status(
            f"{self._provider_label} epoch summary",
            epoch=event.epoch_index,
            epoch_count=event.epoch_count,
            train_loss=f"{event.train_loss:.6g}",
            validation_loss=f"{event.validation_loss:.6g}",
            validation_metric=f"{event.validation_metric:.6g}",
            best="yes" if event.best_checkpoint_updated else "no",
            elapsed_seconds=f"{event.elapsed_seconds:.1f}",
        )


def _training_batch_counters(event: TrainingBatchProcessed) -> dict[str, object]:
    return {
        "loss": _format_optional_float(getattr(event, "batch_loss", None)),
        "running_loss": _format_optional_float(getattr(event, "running_loss", None)),
        "frames": getattr(event, "valid_frame_count", None),
        "skipped": 1 if getattr(event, "skipped", False) else 0,
    }


def _validation_batch_counters(event: ValidationBatchProcessed) -> dict[str, object]:
    return {
        "loss": _format_optional_float(getattr(event, "batch_loss", None)),
        "metric": _format_optional_float(getattr(event, "batch_metric", None)),
        "running_loss": _format_optional_float(getattr(event, "running_loss", None)),
        "frames": getattr(event, "valid_frame_count", None),
        "points": getattr(event, "valid_point_count", None),
        "skipped": 1 if getattr(event, "skipped", False) else 0,
    }


def _format_optional_float(value: object) -> str | None:
    if value is None:
        return None
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return None


__all__ = ["ModelWorkflowTrainingProgressSink"]
