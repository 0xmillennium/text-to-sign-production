"""Provider-neutral immutable training run result summary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TRAINING_RESULT_SCHEMA_VERSION = "t2sp-training-result-v1"


@dataclass(frozen=True, slots=True)
class TrainingRunSummary:
    """Small serialized-ready summary of a provider training run."""

    schema_version: str
    model_key: str
    run_name: str
    stage_name: str
    status: str
    started_at_utc: str | None
    completed_at_utc: str | None
    epoch_count: int | None
    global_step_count: int | None
    metric_record_count: int
    best_checkpoint_path: Path | None
    last_checkpoint_path: Path | None
    issues: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != TRAINING_RESULT_SCHEMA_VERSION:
            raise ValueError("training result schema_version is unsupported.")
        for value, name in (
            (self.model_key, "model_key"),
            (self.run_name, "run_name"),
            (self.stage_name, "stage_name"),
            (self.status, "status"),
        ):
            _require_text(value, name)
        for value, name in (
            (self.started_at_utc, "started_at_utc"),
            (self.completed_at_utc, "completed_at_utc"),
        ):
            if value is not None:
                _require_text(value, name)
        _optional_non_negative_int(self.epoch_count, "epoch_count")
        _optional_non_negative_int(self.global_step_count, "global_step_count")
        if (
            not isinstance(self.metric_record_count, int)
            or isinstance(self.metric_record_count, bool)
            or self.metric_record_count < 0
        ):
            raise ValueError("metric_record_count must be a non-negative integer.")
        for name in ("best_checkpoint_path", "last_checkpoint_path"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str | Path) or not str(value).strip():
                    raise ValueError(f"{name} must be a non-empty path when provided.")
                object.__setattr__(self, name, Path(value))
        issues = tuple(self.issues)
        for issue in issues:
            _require_text(issue, "issue")
        object.__setattr__(self, "issues", issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_key": self.model_key,
            "run_name": self.run_name,
            "stage_name": self.stage_name,
            "status": self.status,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "epoch_count": self.epoch_count,
            "global_step_count": self.global_step_count,
            "metric_record_count": self.metric_record_count,
            "best_checkpoint_path": (
                None if self.best_checkpoint_path is None else str(self.best_checkpoint_path)
            ),
            "last_checkpoint_path": (
                None if self.last_checkpoint_path is None else str(self.last_checkpoint_path)
            ),
            "issues": list(self.issues),
        }


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty.")


def _optional_non_negative_int(value: int | None, name: str) -> None:
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer when provided.")


__all__ = ["TRAINING_RESULT_SCHEMA_VERSION", "TrainingRunSummary"]
