"""Immutable validation artifact records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.validation.errors import ModelValidationError
from text_to_sign_production.modeling.validation.identifiers import (
    ValidationChannelMetricKey,
    ValidationMetricKey,
)

VALIDATION_SCHEMA_VERSION = "t2sp-model-validation-v2"
_PAIRING_STATUSES = frozenset(
    {"paired", "missing_generated", "missing_reference", "generated_failed", "identity_mismatch"}
)


@dataclass(frozen=True, slots=True)
class ValidationPairingKey:
    split: SampleSplit
    sample_id: str
    generation_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", _validation_split(self.split))
        _require_text(self.sample_id, "sample_id")
        _require_non_negative_int(self.generation_index, "generation_index")

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split.value,
            "sample_id": self.sample_id,
            "generation_index": self.generation_index,
        }


@dataclass(frozen=True, slots=True)
class ValidationPairingEntry:
    schema_version: str
    key: ValidationPairingKey
    status: str
    reference_sample_id: str | None
    generated_sample_id: str | None
    reference_payload_path: Path | None
    generated_payload_path: Path | None
    issues: tuple[str, ...]

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        if not isinstance(self.key, ValidationPairingKey):
            raise ModelValidationError("validation pairing key must be a ValidationPairingKey.")
        if self.status not in _PAIRING_STATUSES:
            raise ModelValidationError(f"unsupported validation pairing status: {self.status!r}.")
        for value, label in (
            (self.reference_sample_id, "reference_sample_id"),
            (self.generated_sample_id, "generated_sample_id"),
        ):
            if value is not None:
                _require_text(value, label)
        if self.reference_payload_path is not None:
            object.__setattr__(self, "reference_payload_path", Path(self.reference_payload_path))
        if self.generated_payload_path is not None:
            object.__setattr__(self, "generated_payload_path", Path(self.generated_payload_path))
        object.__setattr__(self, "issues", _issues(self.issues))
        if self.status == "paired" and (
            self.reference_payload_path is None or self.generated_payload_path is None
        ):
            raise ModelValidationError("paired validation entry must include both payload paths.")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            **self.key.to_dict(),
            "status": self.status,
            "reference_sample_id": self.reference_sample_id,
            "generated_sample_id": self.generated_sample_id,
            "reference_payload_path": (
                None if self.reference_payload_path is None else str(self.reference_payload_path)
            ),
            "generated_payload_path": (
                None if self.generated_payload_path is None else str(self.generated_payload_path)
            ),
            "issues": list(self.issues),
        }


@dataclass(frozen=True, slots=True)
class ValidationMetricResult:
    schema_version: str
    metric_key: ValidationMetricKey
    split: SampleSplit
    sample_id: str
    generation_index: int
    value: float
    frame_count: int
    joint_count: int
    valid_value_count: int
    issues: tuple[str, ...]

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        object.__setattr__(self, "metric_key", ValidationMetricKey(self.metric_key))
        object.__setattr__(self, "split", _validation_split(self.split))
        _require_text(self.sample_id, "sample_id")
        _require_non_negative_int(self.generation_index, "generation_index")
        _require_finite(self.value, "value")
        for name in ("frame_count", "joint_count", "valid_value_count"):
            _require_non_negative_int(getattr(self, name), name)
        object.__setattr__(self, "issues", _issues(self.issues))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "metric_key": self.metric_key.value,
            "split": self.split.value,
            "sample_id": self.sample_id,
            "generation_index": self.generation_index,
            "value": self.value,
            "frame_count": self.frame_count,
            "joint_count": self.joint_count,
            "valid_value_count": self.valid_value_count,
            "issues": list(self.issues),
        }


@dataclass(frozen=True, slots=True)
class ValidationChannelMetricResult:
    schema_version: str
    metric_key: ValidationChannelMetricKey
    channel: PoseChannel
    split: SampleSplit
    sample_id: str
    generation_index: int
    value: float
    frame_count: int
    joint_count: int
    valid_value_count: int
    issues: tuple[str, ...]

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        object.__setattr__(self, "metric_key", ValidationChannelMetricKey(self.metric_key))
        object.__setattr__(self, "channel", PoseChannel(self.channel))
        object.__setattr__(self, "split", _validation_split(self.split))
        _require_text(self.sample_id, "sample_id")
        _require_non_negative_int(self.generation_index, "generation_index")
        _require_finite(self.value, "value")
        for name in ("frame_count", "joint_count", "valid_value_count"):
            _require_non_negative_int(getattr(self, name), name)
        object.__setattr__(self, "issues", _issues(self.issues))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "metric_key": self.metric_key.value,
            "channel": self.channel.value,
            "split": self.split.value,
            "sample_id": self.sample_id,
            "generation_index": self.generation_index,
            "value": self.value,
            "frame_count": self.frame_count,
            "joint_count": self.joint_count,
            "valid_value_count": self.valid_value_count,
            "issues": list(self.issues),
        }


@dataclass(frozen=True, slots=True)
class ValidationAggregateMetric:
    schema_version: str
    metric_key: str
    split: SampleSplit
    count: int
    mean: float | None
    median: float | None
    min: float | None
    max: float | None
    std: float | None
    missing_count: int
    issue_count: int

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        _require_text(self.metric_key, "metric_key")
        object.__setattr__(self, "split", _validation_split(self.split))
        for name in ("count", "missing_count", "issue_count"):
            _require_non_negative_int(getattr(self, name), name)
        numeric = (self.mean, self.median, self.min, self.max, self.std)
        if self.count == 0 and any(value is not None for value in numeric):
            raise ModelValidationError("zero-count aggregate numeric values must be None.")
        if self.count > 0 and any(value is None for value in numeric):
            raise ModelValidationError("observed aggregate numeric values must all be present.")
        for name, value in zip(("mean", "median", "min", "max", "std"), numeric, strict=True):
            if value is not None:
                _require_finite(value, name)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "metric_key": self.metric_key,
            "split": self.split.value,
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "min": self.min,
            "max": self.max,
            "std": self.std,
            "missing_count": self.missing_count,
            "issue_count": self.issue_count,
        }


@dataclass(frozen=True, slots=True)
class ValidationLimitations:
    schema_version: str
    split: SampleSplit
    automatic_metrics_are_not_intelligibility: bool
    validation_is_not_test_performance: bool
    simple_prefix_alignment: bool
    generated_missing_count: int
    reference_missing_count: int
    paired_count: int
    failed_generated_count: int
    identity_mismatch_count: int
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        object.__setattr__(self, "split", _validation_split(self.split))
        for name in (
            "generated_missing_count",
            "reference_missing_count",
            "paired_count",
            "failed_generated_count",
            "identity_mismatch_count",
        ):
            _require_non_negative_int(getattr(self, name), name)
        object.__setattr__(self, "notes", _issues(self.notes))
        if not all(
            (
                self.automatic_metrics_are_not_intelligibility,
                self.validation_is_not_test_performance,
                self.simple_prefix_alignment,
            )
        ):
            raise ModelValidationError("validation limitations must retain required interpretation flags.")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "split": self.split.value,
            "automatic_metrics_are_not_intelligibility": self.automatic_metrics_are_not_intelligibility,
            "validation_is_not_test_performance": self.validation_is_not_test_performance,
            "simple_prefix_alignment": self.simple_prefix_alignment,
            "generated_missing_count": self.generated_missing_count,
            "reference_missing_count": self.reference_missing_count,
            "paired_count": self.paired_count,
            "failed_generated_count": self.failed_generated_count,
            "identity_mismatch_count": self.identity_mismatch_count,
            "notes": list(self.notes),
        }


def _schema(value: str) -> None:
    if value != VALIDATION_SCHEMA_VERSION:
        raise ModelValidationError("model validation schema_version is unsupported.")


def _validation_split(value: SampleSplit | str) -> SampleSplit:
    split = SampleSplit(value)
    if split is not SampleSplit.VAL:
        raise ModelValidationError("model validation records must use the validation split 'val'.")
    return split


def _require_text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ModelValidationError(f"{label} must be non-empty.")


def _require_non_negative_int(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ModelValidationError(f"{label} must be a non-negative integer.")


def _require_finite(value: object, label: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ModelValidationError(f"{label} must be finite.")


def _issues(values: tuple[str, ...]) -> tuple[str, ...]:
    resolved = tuple(values)
    for value in resolved:
        _require_text(value, "issue")
    return resolved


__all__ = [
    "VALIDATION_SCHEMA_VERSION",
    "ValidationAggregateMetric",
    "ValidationChannelMetricResult",
    "ValidationLimitations",
    "ValidationMetricResult",
    "ValidationPairingEntry",
    "ValidationPairingKey",
]
