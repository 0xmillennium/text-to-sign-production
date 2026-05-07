"""Run-log events and sinks for baseline training."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeAlias


@dataclass(frozen=True, slots=True)
class RunStarted:
    run_mode: str | None
    training_surface: str
    validation_surface: str


@dataclass(frozen=True, slots=True)
class EpochStarted:
    epoch_index: int
    epoch_count: int


@dataclass(frozen=True, slots=True)
class EpochSummaryLogged:
    epoch_index: int
    epoch_count: int
    train_loss: float
    validation_loss: float
    validation_metric: float
    elapsed_seconds: float
    best_checkpoint_updated: bool


@dataclass(frozen=True, slots=True)
class ResumeStateLogged:
    checkpoint_path: Path
    completed_epoch: int


@dataclass(frozen=True, slots=True)
class EarlyStoppingDecisionLogged:
    epoch_index: int
    patience_epochs: int
    should_stop: bool


@dataclass(frozen=True, slots=True)
class CheckpointSavedLog:
    checkpoint_path: Path
    role: str
    epoch: int


TrainingRunLogEvent: TypeAlias = (
    RunStarted
    | EpochStarted
    | EpochSummaryLogged
    | ResumeStateLogged
    | EarlyStoppingDecisionLogged
    | CheckpointSavedLog
)


class TrainingRunLogSink(Protocol):
    def emit(self, event: TrainingRunLogEvent) -> None: ...


@dataclass(frozen=True, slots=True)
class NoOpTrainingRunLogSink:
    def emit(self, event: TrainingRunLogEvent) -> None:
        return


@dataclass(frozen=True, slots=True)
class TextTrainingRunLogSink:
    prefix: str = ""
    log_path: Path | None = None
    stream: object = field(default_factory=lambda: sys.stdout)

    def emit(self, event: TrainingRunLogEvent) -> None:
        message, fields = _event_line_parts(event)
        line = _render_line(self.prefix, message, fields)
        print(line, file=self.stream)
        flush = getattr(self.stream, "flush", None)
        if callable(flush):
            flush()
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")


def _event_line_parts(event: TrainingRunLogEvent) -> tuple[str, Mapping[str, object]]:
    if isinstance(event, RunStarted):
        return (
            "run start",
            {
                "run_mode": event.run_mode,
                "training_surface": event.training_surface,
                "validation_surface": event.validation_surface,
            },
        )
    if isinstance(event, EpochStarted):
        return "epoch start", {"epoch": event.epoch_index, "epoch_count": event.epoch_count}
    if isinstance(event, EpochSummaryLogged):
        return (
            "epoch summary",
            {
                "epoch": event.epoch_index,
                "epoch_count": event.epoch_count,
                "train_loss": f"{event.train_loss:.6g}",
                "validation_loss": f"{event.validation_loss:.6g}",
                "validation_metric": f"{event.validation_metric:.6g}",
                "elapsed_seconds": f"{event.elapsed_seconds:.1f}",
                "best_checkpoint_updated": "yes" if event.best_checkpoint_updated else "no",
            },
        )
    if isinstance(event, ResumeStateLogged):
        return (
            "resume",
            {
                "checkpoint": event.checkpoint_path,
                "completed_epoch": event.completed_epoch,
            },
        )
    if isinstance(event, EarlyStoppingDecisionLogged):
        return (
            "early stopping",
            {
                "epoch": event.epoch_index,
                "patience": event.patience_epochs,
                "should_stop": event.should_stop,
            },
        )
    if isinstance(event, CheckpointSavedLog):
        return (
            "checkpoint saved",
            {
                "path": event.checkpoint_path,
                "role": event.role,
                "epoch": event.epoch,
            },
        )
    raise TypeError(f"Unsupported training log event: {type(event).__name__}")


def _render_line(prefix: str, message: str, fields: Mapping[str, object]) -> str:
    field_parts = [
        f"{key}={value}"
        for key, value in fields.items()
        if value is not None and str(value) != ""
    ]
    body = message if not field_parts else f"{message} " + " ".join(field_parts)
    stripped_prefix = prefix.strip()
    return body if not stripped_prefix else f"{stripped_prefix} {body}"


__all__ = [
    "CheckpointSavedLog",
    "EarlyStoppingDecisionLogged",
    "EpochStarted",
    "EpochSummaryLogged",
    "NoOpTrainingRunLogSink",
    "ResumeStateLogged",
    "RunStarted",
    "TextTrainingRunLogSink",
    "TrainingRunLogEvent",
    "TrainingRunLogSink",
]
