"""Typed domain progress events for baseline training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias


@dataclass(frozen=True, slots=True)
class StandardizationItemProcessed:
    item_index: int
    total_items: int


@dataclass(frozen=True, slots=True)
class TrainingBatchProcessed:
    epoch_index: int
    epoch_count: int
    batch_index: int
    batch_count: int | None
    valid_frame_count: int
    batch_loss: float | None
    running_loss: float | None
    skipped: bool = False


@dataclass(frozen=True, slots=True)
class ValidationBatchProcessed:
    epoch_index: int
    epoch_count: int
    batch_index: int
    batch_count: int | None
    valid_frame_count: int
    valid_point_count: int
    batch_loss: float | None
    batch_metric: float | None
    running_loss: float | None
    skipped: bool = False


@dataclass(frozen=True, slots=True)
class EpochCompleted:
    epoch_index: int
    epoch_count: int
    train_loss: float
    validation_loss: float
    validation_metric: float
    elapsed_seconds: float
    best_checkpoint_updated: bool


@dataclass(frozen=True, slots=True)
class CheckpointSaved:
    checkpoint_path: Path
    role: str
    epoch: int


@dataclass(frozen=True, slots=True)
class MetricsWritten:
    metrics_path: Path
    epoch: int


@dataclass(frozen=True, slots=True)
class SummaryWritten:
    summary_path: Path


@dataclass(frozen=True, slots=True)
class ResumeLoaded:
    checkpoint_path: Path
    completed_epoch: int
    best_epoch: int | None
    best_metric: float | None


@dataclass(frozen=True, slots=True)
class EarlyStoppingEvaluated:
    epoch_index: int
    min_epochs: int
    patience: int
    patience_epochs: int
    should_stop: bool
    reason: str | None


TrainingProgressEvent: TypeAlias = (
    StandardizationItemProcessed
    | TrainingBatchProcessed
    | ValidationBatchProcessed
    | EpochCompleted
    | CheckpointSaved
    | MetricsWritten
    | SummaryWritten
    | ResumeLoaded
    | EarlyStoppingEvaluated
)


class TrainingProgressSink(Protocol):
    def emit(self, event: TrainingProgressEvent) -> None: ...


@dataclass(frozen=True, slots=True)
class NoOpTrainingProgressSink:
    def emit(self, event: TrainingProgressEvent) -> None:
        return


__all__ = [
    "CheckpointSaved",
    "EarlyStoppingEvaluated",
    "EpochCompleted",
    "MetricsWritten",
    "NoOpTrainingProgressSink",
    "ResumeLoaded",
    "StandardizationItemProcessed",
    "SummaryWritten",
    "TrainingBatchProcessed",
    "TrainingProgressEvent",
    "TrainingProgressSink",
    "ValidationBatchProcessed",
]
