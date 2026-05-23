"""Provider-neutral checkpoint metadata and deterministic selection contracts."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

CHECKPOINT_METADATA_SCHEMA_VERSION = "t2sp-checkpoint-metadata-v1"


class CheckpointRole(StrEnum):
    BEST = "best"
    LAST = "last"
    INTERMEDIATE = "intermediate"


@dataclass(frozen=True, slots=True)
class ModelCheckpointMetadata:
    """Identity and optional metric evidence for one saved checkpoint."""

    schema_version: str
    model_key: str
    run_name: str
    role: CheckpointRole
    stage_name: str
    checkpoint_path: Path
    epoch: int | None
    global_step: int | None
    selection_metric: str | None
    selection_metric_value: float | None
    lower_is_better: bool | None
    created_at_utc: str
    extra: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_METADATA_SCHEMA_VERSION:
            raise ValueError("checkpoint metadata schema_version is unsupported.")
        for value, name in (
            (self.model_key, "model_key"),
            (self.run_name, "run_name"),
            (self.stage_name, "stage_name"),
            (self.created_at_utc, "created_at_utc"),
        ):
            _require_text(value, name)
        try:
            object.__setattr__(self, "role", CheckpointRole(self.role))
        except ValueError as exc:
            raise ValueError(f"unsupported checkpoint role: {self.role!r}.") from exc
        object.__setattr__(self, "checkpoint_path", _path(self.checkpoint_path, "checkpoint_path"))
        _optional_non_negative_int(self.epoch, "epoch")
        _optional_non_negative_int(self.global_step, "global_step")
        if self.selection_metric is None:
            if self.selection_metric_value is not None or self.lower_is_better is not None:
                raise ValueError(
                    "selection_metric is required when selection metric value or direction is provided."
                )
        else:
            _require_text(self.selection_metric, "selection_metric")
            if self.selection_metric_value is None or self.lower_is_better is None:
                raise ValueError(
                    "selection_metric_value and lower_is_better are required for a selection metric."
                )
        if self.selection_metric_value is not None and not _finite(self.selection_metric_value):
            raise ValueError("selection_metric_value must be finite when provided.")
        if self.lower_is_better is not None and not isinstance(self.lower_is_better, bool):
            raise ValueError("lower_is_better must be boolean when provided.")
        object.__setattr__(self, "extra", _json_mapping(self.extra, "extra"))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_key": self.model_key,
            "run_name": self.run_name,
            "role": self.role.value,
            "stage_name": self.stage_name,
            "checkpoint_path": str(self.checkpoint_path),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "selection_metric": self.selection_metric,
            "selection_metric_value": self.selection_metric_value,
            "lower_is_better": self.lower_is_better,
            "created_at_utc": self.created_at_utc,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True, slots=True)
class BestCheckpointSelection:
    """Result of selecting among checkpoint metadata carrying one metric."""

    metric_name: str
    lower_is_better: bool
    selected_checkpoint_path: Path | None
    selected_metric_value: float | None
    candidate_count: int
    reason: str

    def __post_init__(self) -> None:
        _require_text(self.metric_name, "metric_name")
        if not isinstance(self.lower_is_better, bool):
            raise ValueError("lower_is_better must be boolean.")
        if self.selected_checkpoint_path is not None:
            object.__setattr__(
                self,
                "selected_checkpoint_path",
                _path(self.selected_checkpoint_path, "selected_checkpoint_path"),
            )
        if self.selected_metric_value is not None and not _finite(self.selected_metric_value):
            raise ValueError("selected_metric_value must be finite when provided.")
        if not isinstance(self.candidate_count, int) or isinstance(self.candidate_count, bool) or self.candidate_count < 0:
            raise ValueError("candidate_count must be a non-negative integer.")
        _require_text(self.reason, "reason")
        absent = self.selected_checkpoint_path is None and self.selected_metric_value is None
        present = self.selected_checkpoint_path is not None and self.selected_metric_value is not None
        if not (absent or present):
            raise ValueError("selected checkpoint path and metric value must be present together.")
        if self.candidate_count == 0 and not absent:
            raise ValueError("zero checkpoint candidates cannot yield a selected checkpoint.")

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_name": self.metric_name,
            "lower_is_better": self.lower_is_better,
            "selected_checkpoint_path": (
                None if self.selected_checkpoint_path is None else str(self.selected_checkpoint_path)
            ),
            "selected_metric_value": self.selected_metric_value,
            "candidate_count": self.candidate_count,
            "reason": self.reason,
        }


def write_checkpoint_metadata_json(path: Path, metadata: ModelCheckpointMetadata) -> None:
    """Write deterministic UTF-8 checkpoint metadata JSON."""

    if not isinstance(metadata, ModelCheckpointMetadata):
        raise ValueError("metadata must be ModelCheckpointMetadata.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def read_checkpoint_metadata_json(path: Path) -> ModelCheckpointMetadata:
    """Read strict checkpoint metadata JSON."""

    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed checkpoint metadata JSON: {exc}") from exc
    expected = {
        "schema_version",
        "model_key",
        "run_name",
        "role",
        "stage_name",
        "checkpoint_path",
        "epoch",
        "global_step",
        "selection_metric",
        "selection_metric_value",
        "lower_is_better",
        "created_at_utc",
        "extra",
    }
    if not isinstance(record, Mapping) or set(record) != expected:
        raise ValueError("checkpoint metadata JSON keys do not match the schema.")
    return ModelCheckpointMetadata(
        schema_version=_value_text(record["schema_version"], "schema_version"),
        model_key=_value_text(record["model_key"], "model_key"),
        run_name=_value_text(record["run_name"], "run_name"),
        role=CheckpointRole(_value_text(record["role"], "role")),
        stage_name=_value_text(record["stage_name"], "stage_name"),
        checkpoint_path=Path(_value_text(record["checkpoint_path"], "checkpoint_path")),
        epoch=_value_optional_int(record["epoch"], "epoch"),
        global_step=_value_optional_int(record["global_step"], "global_step"),
        selection_metric=_value_optional_text(record["selection_metric"], "selection_metric"),
        selection_metric_value=_value_optional_float(
            record["selection_metric_value"], "selection_metric_value"
        ),
        lower_is_better=_value_optional_bool(record["lower_is_better"], "lower_is_better"),
        created_at_utc=_value_text(record["created_at_utc"], "created_at_utc"),
        extra=_value_mapping(record["extra"], "extra"),
    )


def select_best_checkpoint(
    checkpoint_metadata: Iterable[ModelCheckpointMetadata],
    *,
    metric_name: str,
    lower_is_better: bool,
) -> BestCheckpointSelection:
    """Select the best eligible metric-bearing checkpoint, or honestly select none."""

    _require_text(metric_name, "metric_name")
    if not isinstance(lower_is_better, bool):
        raise ValueError("lower_is_better must be boolean.")
    materialized = tuple(checkpoint_metadata)
    if any(not isinstance(item, ModelCheckpointMetadata) for item in materialized):
        raise ValueError("checkpoint_metadata must contain ModelCheckpointMetadata values.")
    candidates = tuple(
        item
        for item in materialized
        if item.selection_metric == metric_name
        and item.selection_metric_value is not None
        and item.lower_is_better is lower_is_better
    )
    if not candidates:
        return BestCheckpointSelection(
            metric_name=metric_name,
            lower_is_better=lower_is_better,
            selected_checkpoint_path=None,
            selected_metric_value=None,
            candidate_count=0,
            reason="No checkpoint candidates carry the requested selection metric and direction.",
        )
    selected = min(
        candidates,
        key=lambda item: (
            item.selection_metric_value
            if lower_is_better
            else -float(item.selection_metric_value),
            str(item.checkpoint_path),
        ),
    )
    return BestCheckpointSelection(
        metric_name=metric_name,
        lower_is_better=lower_is_better,
        selected_checkpoint_path=selected.checkpoint_path,
        selected_metric_value=selected.selection_metric_value,
        candidate_count=len(candidates),
        reason="Selected the best finite metric value among matching checkpoint candidates.",
    )


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty.")


def _path(value: object, name: str) -> Path:
    if not isinstance(value, str | Path) or not str(value).strip():
        raise ValueError(f"{name} must be a non-empty path.")
    return Path(value)


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


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


def _value_text(value: object, name: str) -> str:
    _require_text(value, name)
    return str(value)


def _value_optional_text(value: object, name: str) -> str | None:
    return None if value is None else _value_text(value, name)


def _value_optional_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer or None.")
    return value


def _value_optional_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric or None.")
    return float(value)


def _value_optional_bool(value: object, name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean or None.")
    return value


def _value_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    return value


__all__ = [
    "CHECKPOINT_METADATA_SCHEMA_VERSION",
    "BestCheckpointSelection",
    "CheckpointRole",
    "ModelCheckpointMetadata",
    "read_checkpoint_metadata_json",
    "select_best_checkpoint",
    "write_checkpoint_metadata_json",
]
