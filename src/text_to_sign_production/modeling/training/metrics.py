"""Provider-neutral training metric records and deterministic JSONL IO."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

TRAINING_METRIC_SCHEMA_VERSION = "t2sp-training-metric-v1"


@dataclass(frozen=True, slots=True)
class TrainingMetricRecord:
    """One finite provider training/validation metric observation."""

    schema_version: str
    split: str
    metric_name: str
    value: float
    epoch: int | None
    global_step: int | None
    stage_name: str
    created_at_utc: str
    extra: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.schema_version != TRAINING_METRIC_SCHEMA_VERSION:
            raise ValueError("training metric schema_version is unsupported.")
        for value, name in (
            (self.split, "split"),
            (self.metric_name, "metric_name"),
            (self.stage_name, "stage_name"),
            (self.created_at_utc, "created_at_utc"),
        ):
            _require_text(value, name)
        if not isinstance(self.value, int | float) or isinstance(self.value, bool) or not math.isfinite(self.value):
            raise ValueError("training metric value must be finite.")
        _optional_non_negative_int(self.epoch, "epoch")
        _optional_non_negative_int(self.global_step, "global_step")
        object.__setattr__(self, "extra", _json_mapping(self.extra, "extra"))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "split": self.split,
            "metric_name": self.metric_name,
            "value": float(self.value),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "stage_name": self.stage_name,
            "created_at_utc": self.created_at_utc,
            "extra": dict(self.extra),
        }


def write_training_metric_records_jsonl(
    path: Path,
    records: Iterable[TrainingMetricRecord],
) -> None:
    """Write deterministic schema-validated training metric JSONL."""

    materialized = tuple(records)
    if any(not isinstance(record, TrainingMetricRecord) for record in materialized):
        raise ValueError("records must contain TrainingMetricRecord values.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in materialized:
            handle.write(json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def read_training_metric_records_jsonl(path: Path) -> tuple[TrainingMetricRecord, ...]:
    """Read strict schema-validated training metric JSONL."""

    records: list[TrainingMetricRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed training metric JSONL at line {line_number}: {exc}") from exc
            if not isinstance(document, Mapping):
                raise ValueError(f"training metric JSONL record at line {line_number} must be an object.")
            expected = {
                "schema_version",
                "split",
                "metric_name",
                "value",
                "epoch",
                "global_step",
                "stage_name",
                "created_at_utc",
                "extra",
            }
            if set(document) != expected:
                raise ValueError(f"training metric JSONL record keys mismatch at line {line_number}.")
            records.append(
                TrainingMetricRecord(
                    schema_version=_record_text(document["schema_version"], "schema_version"),
                    split=_record_text(document["split"], "split"),
                    metric_name=_record_text(document["metric_name"], "metric_name"),
                    value=_record_float(document["value"], "value"),
                    epoch=_record_optional_int(document["epoch"], "epoch"),
                    global_step=_record_optional_int(document["global_step"], "global_step"),
                    stage_name=_record_text(document["stage_name"], "stage_name"),
                    created_at_utc=_record_text(document["created_at_utc"], "created_at_utc"),
                    extra=_record_mapping(document["extra"], "extra"),
                )
            )
    return tuple(records)


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty.")


def _optional_non_negative_int(value: int | None, name: str) -> None:
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer when provided.")


def _json_mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys.")
    resolved = dict(value)
    try:
        json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain JSON-serializable values.") from exc
    return MappingProxyType(resolved)


def _record_text(value: object, name: str) -> str:
    _require_text(value, name)
    return str(value)


def _record_float(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")
    return float(value)


def _record_optional_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer or None.")
    return value


def _record_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    return value


__all__ = [
    "TRAINING_METRIC_SCHEMA_VERSION",
    "TrainingMetricRecord",
    "read_training_metric_records_jsonl",
    "write_training_metric_records_jsonl",
]
