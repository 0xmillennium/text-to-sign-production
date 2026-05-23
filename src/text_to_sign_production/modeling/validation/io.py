"""Deterministic JSON and JSONL IO for model validation artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.validation.errors import ModelValidationError
from text_to_sign_production.modeling.validation.identifiers import (
    ValidationChannelMetricKey,
    ValidationMetricKey,
)
from text_to_sign_production.modeling.validation.records import (
    VALIDATION_SCHEMA_VERSION,
    ValidationAggregateMetric,
    ValidationChannelMetricResult,
    ValidationLimitations,
    ValidationMetricResult,
    ValidationPairingEntry,
    ValidationPairingKey,
)

_PAIRING_KEYS = frozenset(
    {
        "schema_version",
        "split",
        "sample_id",
        "generation_index",
        "status",
        "reference_sample_id",
        "generated_sample_id",
        "reference_payload_path",
        "generated_payload_path",
        "issues",
    }
)
_METRIC_KEYS = frozenset(
    {
        "schema_version",
        "metric_key",
        "split",
        "sample_id",
        "generation_index",
        "value",
        "frame_count",
        "joint_count",
        "valid_value_count",
        "issues",
    }
)
_CHANNEL_METRIC_KEYS = _METRIC_KEYS | {"channel"}
_AGGREGATE_KEYS = frozenset(
    {
        "schema_version",
        "metric_key",
        "split",
        "count",
        "mean",
        "median",
        "min",
        "max",
        "std",
        "missing_count",
        "issue_count",
    }
)
_LIMITATION_KEYS = frozenset(
    {
        "schema_version",
        "split",
        "automatic_metrics_are_not_intelligibility",
        "validation_is_not_test_performance",
        "simple_prefix_alignment",
        "generated_missing_count",
        "reference_missing_count",
        "paired_count",
        "failed_generated_count",
        "identity_mismatch_count",
        "notes",
    }
)


def write_validation_pairing_jsonl(path: str | Path, entries: Iterable[ValidationPairingEntry]) -> None:
    _write_jsonl(path, (entry.to_dict() for entry in entries))


def read_validation_pairing_jsonl(path: str | Path) -> tuple[ValidationPairingEntry, ...]:
    return tuple(_pairing_from_record(record) for record in _read_jsonl(path))


def write_validation_metric_results_jsonl(
    path: str | Path,
    results: Iterable[ValidationMetricResult],
) -> None:
    _write_jsonl(path, (result.to_dict() for result in results))


def read_validation_metric_results_jsonl(path: str | Path) -> tuple[ValidationMetricResult, ...]:
    return tuple(_metric_from_record(record) for record in _read_jsonl(path))


def write_validation_channel_metric_results_jsonl(
    path: str | Path,
    results: Iterable[ValidationChannelMetricResult],
) -> None:
    _write_jsonl(path, (result.to_dict() for result in results))


def read_validation_channel_metric_results_jsonl(
    path: str | Path,
) -> tuple[ValidationChannelMetricResult, ...]:
    return tuple(_channel_metric_from_record(record) for record in _read_jsonl(path))


def write_validation_aggregate_metrics_json(
    path: str | Path,
    aggregates: Iterable[ValidationAggregateMetric],
) -> None:
    values = tuple(aggregates)
    _write_json(
        path,
        {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "aggregates": [value.to_dict() for value in values],
        },
    )


def read_validation_aggregate_metrics_json(path: str | Path) -> tuple[ValidationAggregateMetric, ...]:
    document = _json_object(path)
    _exact_keys(document, frozenset({"schema_version", "aggregates"}), "aggregate document")
    _schema(document["schema_version"])
    values = document["aggregates"]
    if not isinstance(values, list):
        raise ModelValidationError("validation aggregate document aggregates must be a list.")
    return tuple(_aggregate_from_record(_mapping(value, "aggregate")) for value in values)


def write_validation_limitations_json(path: str | Path, limitations: ValidationLimitations) -> None:
    _write_json(path, limitations.to_dict())


def read_validation_limitations_json(path: str | Path) -> ValidationLimitations:
    return _limitations_from_record(_json_object(path))


def _write_jsonl(path: str | Path, records: Iterable[Mapping[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _read_jsonl(path: str | Path) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ModelValidationError(
                    f"malformed validation JSONL at line {line_number}: {exc}"
                ) from exc
            records.append(_mapping(record, f"JSONL record at line {line_number}"))
    return tuple(records)


def _write_json(path: str | Path, document: Mapping[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(document), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _json_object(path: str | Path) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelValidationError(f"malformed validation JSON: {exc}") from exc
    return _mapping(value, "validation JSON document")


def _pairing_from_record(record: Mapping[str, object]) -> ValidationPairingEntry:
    _exact_keys(record, _PAIRING_KEYS, "pairing record")
    return ValidationPairingEntry(
        _text(record["schema_version"], "schema_version"),
        ValidationPairingKey(
            SampleSplit(_text(record["split"], "split")),
            _text(record["sample_id"], "sample_id"),
            _int(record["generation_index"], "generation_index"),
        ),
        _text(record["status"], "status"),
        _optional_text(record["reference_sample_id"], "reference_sample_id"),
        _optional_text(record["generated_sample_id"], "generated_sample_id"),
        _optional_path(record["reference_payload_path"], "reference_payload_path"),
        _optional_path(record["generated_payload_path"], "generated_payload_path"),
        _text_tuple(record["issues"], "issues"),
    )


def _metric_from_record(record: Mapping[str, object]) -> ValidationMetricResult:
    _exact_keys(record, _METRIC_KEYS, "metric record")
    return ValidationMetricResult(
        _text(record["schema_version"], "schema_version"),
        ValidationMetricKey(_text(record["metric_key"], "metric_key")),
        SampleSplit(_text(record["split"], "split")),
        _text(record["sample_id"], "sample_id"),
        _int(record["generation_index"], "generation_index"),
        _float(record["value"], "value"),
        _int(record["frame_count"], "frame_count"),
        _int(record["joint_count"], "joint_count"),
        _int(record["valid_value_count"], "valid_value_count"),
        _text_tuple(record["issues"], "issues"),
    )


def _channel_metric_from_record(record: Mapping[str, object]) -> ValidationChannelMetricResult:
    _exact_keys(record, _CHANNEL_METRIC_KEYS, "channel metric record")
    return ValidationChannelMetricResult(
        _text(record["schema_version"], "schema_version"),
        ValidationChannelMetricKey(_text(record["metric_key"], "metric_key")),
        PoseChannel(_text(record["channel"], "channel")),
        SampleSplit(_text(record["split"], "split")),
        _text(record["sample_id"], "sample_id"),
        _int(record["generation_index"], "generation_index"),
        _float(record["value"], "value"),
        _int(record["frame_count"], "frame_count"),
        _int(record["joint_count"], "joint_count"),
        _int(record["valid_value_count"], "valid_value_count"),
        _text_tuple(record["issues"], "issues"),
    )


def _aggregate_from_record(record: Mapping[str, object]) -> ValidationAggregateMetric:
    _exact_keys(record, _AGGREGATE_KEYS, "aggregate record")
    return ValidationAggregateMetric(
        _text(record["schema_version"], "schema_version"),
        _text(record["metric_key"], "metric_key"),
        SampleSplit(_text(record["split"], "split")),
        _int(record["count"], "count"),
        _optional_float(record["mean"], "mean"),
        _optional_float(record["median"], "median"),
        _optional_float(record["min"], "min"),
        _optional_float(record["max"], "max"),
        _optional_float(record["std"], "std"),
        _int(record["missing_count"], "missing_count"),
        _int(record["issue_count"], "issue_count"),
    )


def _limitations_from_record(record: Mapping[str, object]) -> ValidationLimitations:
    _exact_keys(record, _LIMITATION_KEYS, "limitations record")
    return ValidationLimitations(
        _text(record["schema_version"], "schema_version"),
        SampleSplit(_text(record["split"], "split")),
        _bool(record["automatic_metrics_are_not_intelligibility"], "automatic_metrics_are_not_intelligibility"),
        _bool(record["validation_is_not_test_performance"], "validation_is_not_test_performance"),
        _bool(record["simple_prefix_alignment"], "simple_prefix_alignment"),
        _int(record["generated_missing_count"], "generated_missing_count"),
        _int(record["reference_missing_count"], "reference_missing_count"),
        _int(record["paired_count"], "paired_count"),
        _int(record["failed_generated_count"], "failed_generated_count"),
        _int(record["identity_mismatch_count"], "identity_mismatch_count"),
        _text_tuple(record["notes"], "notes"),
    )


def _schema(value: object) -> None:
    if _text(value, "schema_version") != VALIDATION_SCHEMA_VERSION:
        raise ModelValidationError("model validation schema_version is unsupported.")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ModelValidationError(f"{label} must be a JSON object with string keys.")
    return value


def _exact_keys(record: Mapping[str, object], keys: frozenset[str], label: str) -> None:
    observed = frozenset(record)
    if observed != keys:
        raise ModelValidationError(
            f"{label} keys mismatch: missing={sorted(keys - observed)}, extra={sorted(observed - keys)}."
        )


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelValidationError(f"{label} must be non-empty text.")
    return value


def _optional_text(value: object, label: str) -> str | None:
    return None if value is None else _text(value, label)


def _optional_path(value: object, label: str) -> Path | None:
    return None if value is None else Path(_text(value, label))


def _int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ModelValidationError(f"{label} must be an integer.")
    return value


def _float(value: object, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ModelValidationError(f"{label} must be numeric.")
    return float(value)


def _optional_float(value: object, label: str) -> float | None:
    return None if value is None else _float(value, label)


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ModelValidationError(f"{label} must be a boolean.")
    return value


def _text_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ModelValidationError(f"{label} must be a JSON list.")
    return tuple(_text(item, label) for item in value)


__all__ = [
    "read_validation_aggregate_metrics_json",
    "read_validation_channel_metric_results_jsonl",
    "read_validation_limitations_json",
    "read_validation_metric_results_jsonl",
    "read_validation_pairing_jsonl",
    "write_validation_aggregate_metrics_json",
    "write_validation_channel_metric_results_jsonl",
    "write_validation_limitations_json",
    "write_validation_metric_results_jsonl",
    "write_validation_pairing_jsonl",
]
