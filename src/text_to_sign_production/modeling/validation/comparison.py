"""Comparison tables for candidate-agnostic validation aggregates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.validation.aggregation import (
    aggregate_validation_metric_results,
)
from text_to_sign_production.modeling.validation.config import ValidationComparisonConfig
from text_to_sign_production.modeling.validation.records import (
    VALIDATION_SCHEMA_VERSION,
    ValidationAggregateMetric,
    ValidationChannelMetricResult,
    ValidationMetricResult,
    ValidationPairingEntry,
)


@dataclass(frozen=True, slots=True)
class ValidationSharedPairingKey:
    sample_id: str
    generation_index: int

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "generation_index": self.generation_index,
        }


@dataclass(frozen=True, slots=True)
class ValidationSurfaceAggregateSummary:
    surface_label: str
    producer_key: str
    producer_type: str
    run_name: str
    paired_count: int
    issue_count: int
    missing_count: int
    shared_paired_count: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "surface_label": self.surface_label,
            "producer_key": self.producer_key,
            "producer_type": self.producer_type,
            "run_name": self.run_name,
            "paired_count": self.paired_count,
            "issue_count": self.issue_count,
            "missing_count": self.missing_count,
            "shared_paired_count": self.shared_paired_count,
        }


@dataclass(frozen=True, slots=True)
class ValidationMetricComparisonRow:
    metric_key: str
    surface_label: str
    producer_key: str
    producer_type: str
    run_name: str
    count: int
    mean: float | None
    median: float | None
    missing_count: int
    issue_count: int
    direction: str | None
    rank: int | None
    is_best_by_metric: bool
    availability_status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_key": self.metric_key,
            "surface_label": self.surface_label,
            "producer_key": self.producer_key,
            "producer_type": self.producer_type,
            "run_name": self.run_name,
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "missing_count": self.missing_count,
            "issue_count": self.issue_count,
            "direction": self.direction,
            "rank": self.rank,
            "is_best_by_metric": self.is_best_by_metric,
            "availability_status": self.availability_status,
        }


@dataclass(frozen=True, slots=True)
class ValidationComparisonResult:
    schema_version: str
    status: str
    require_shared_pairing_subset: bool
    shared_pairing_subset_sample_ids: tuple[str, ...]
    shared_pairing_subset_keys: tuple[ValidationSharedPairingKey, ...]
    summaries: tuple[ValidationSurfaceAggregateSummary, ...]
    rows: tuple[ValidationMetricComparisonRow, ...]
    message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "require_shared_pairing_subset": self.require_shared_pairing_subset,
            "shared_pairing_subset_sample_ids": list(self.shared_pairing_subset_sample_ids),
            "shared_pairing_subset_keys": [
                key.to_dict() for key in self.shared_pairing_subset_keys
            ],
            "summaries": [summary.to_dict() for summary in self.summaries],
            "rows": [row.to_dict() for row in self.rows],
            "message": self.message,
        }


def build_validation_comparison(
    *,
    surface_results: Iterable[object],
    config: ValidationComparisonConfig,
) -> ValidationComparisonResult:
    materialized = tuple(surface_results)
    summaries = tuple(_summary(result, None) for result in materialized)
    if len(materialized) < 2:
        return ValidationComparisonResult(
            VALIDATION_SCHEMA_VERSION,
            "skipped",
            config.require_shared_pairing_subset,
            (),
            (),
            summaries,
            (),
            "validation comparison requires at least two validated surfaces.",
        )
    shared_keys = _shared_paired_keys(materialized) if config.require_shared_pairing_subset else None
    shared_sample_ids = (
        tuple(key[0] for key in sorted(shared_keys))
        if shared_keys is not None
        else ()
    )
    shared_key_records = (
        tuple(
            ValidationSharedPairingKey(sample_id=sample_id, generation_index=generation_index)
            for sample_id, generation_index in sorted(shared_keys)
        )
        if shared_keys is not None
        else ()
    )
    summaries = tuple(_summary(result, len(shared_keys) if shared_keys is not None else None) for result in materialized)
    row_inputs: list[tuple[object, ValidationAggregateMetric]] = []
    for result in materialized:
        aggregates = _comparison_aggregates(result, shared_keys=shared_keys)
        row_inputs.extend((result, aggregate) for aggregate in aggregates)
    rows = _rank_rows(
        tuple(_row(result, aggregate, config) for result, aggregate in row_inputs),
        config=config,
    )
    return ValidationComparisonResult(
        VALIDATION_SCHEMA_VERSION,
        "ok",
        config.require_shared_pairing_subset,
        shared_sample_ids,
        shared_key_records,
        summaries,
        rows,
        None,
    )


def _comparison_aggregates(
    result: object,
    *,
    shared_keys: set[tuple[str, int]] | None,
) -> tuple[ValidationAggregateMetric, ...]:
    if shared_keys is None:
        return tuple(getattr(result, "aggregates")) + tuple(getattr(result, "channel_aggregates"))
    metric_results = tuple(
        value
        for value in getattr(result, "metric_results")
        if (value.sample_id, value.generation_index) in shared_keys
    )
    channel_metric_results = tuple(
        value
        for value in getattr(result, "channel_metric_results")
        if (value.sample_id, value.generation_index) in shared_keys
    )
    issue_count = (
        sum(len(value.issues) for value in metric_results)
        + sum(len(value.issues) for value in channel_metric_results)
    )
    missing_count = max(0, len(shared_keys) - len({(value.sample_id, value.generation_index) for value in metric_results}))
    return (
        *aggregate_validation_metric_results(
            metric_results,
            split=SampleSplit.VAL,
            missing_count=missing_count,
            issue_count=issue_count,
        ),
        *aggregate_validation_metric_results(
            channel_metric_results,
            split=SampleSplit.VAL,
            missing_count=missing_count,
            issue_count=issue_count,
        ),
    )


def _shared_paired_keys(results: tuple[object, ...]) -> set[tuple[str, int]]:
    shared: set[tuple[str, int]] | None = None
    for result in results:
        keys = {
            (pair.key.sample_id, pair.key.generation_index)
            for pair in getattr(result, "pairing")
            if isinstance(pair, ValidationPairingEntry) and pair.status == "paired"
        }
        shared = keys if shared is None else shared & keys
    return shared or set()


def _row(
    result: object,
    aggregate: ValidationAggregateMetric,
    config: ValidationComparisonConfig,
) -> ValidationMetricComparisonRow:
    direction = _direction(aggregate.metric_key, config)
    availability_status = "available" if aggregate.count > 0 and aggregate.mean is not None else "unavailable"
    surface = getattr(result, "surface")
    return ValidationMetricComparisonRow(
        metric_key=aggregate.metric_key,
        surface_label=surface.label,
        producer_key=surface.producer_key,
        producer_type=surface.producer_type,
        run_name=surface.run_name,
        count=aggregate.count,
        mean=aggregate.mean,
        median=aggregate.median,
        missing_count=aggregate.missing_count,
        issue_count=aggregate.issue_count,
        direction=direction,
        rank=None,
        is_best_by_metric=False,
        availability_status=availability_status,
    )


def _rank_rows(
    rows: tuple[ValidationMetricComparisonRow, ...],
    *,
    config: ValidationComparisonConfig,
) -> tuple[ValidationMetricComparisonRow, ...]:
    ranked: list[ValidationMetricComparisonRow] = list(rows)
    for metric_key in sorted({row.metric_key for row in rows}):
        metric_indexes = [index for index, row in enumerate(ranked) if row.metric_key == metric_key]
        available = [
            ranked[index]
            for index in metric_indexes
            if ranked[index].availability_status == "available" and ranked[index].direction is not None
        ]
        if not available:
            continue
        reverse = available[0].direction == "higher_is_better"
        ordered = sorted(
            available,
            key=lambda row: (
                -float(row.mean) if reverse else float(row.mean),
                row.surface_label,
            ),
        )
        rank_by_surface = {row.surface_label: rank for rank, row in enumerate(ordered, start=1)}
        for index in metric_indexes:
            row = ranked[index]
            rank = rank_by_surface.get(row.surface_label)
            ranked[index] = ValidationMetricComparisonRow(
                row.metric_key,
                row.surface_label,
                row.producer_key,
                row.producer_type,
                row.run_name,
                row.count,
                row.mean,
                row.median,
                row.missing_count,
                row.issue_count,
                row.direction,
                rank,
                rank == 1,
                row.availability_status,
            )
    return tuple(ranked)


def _direction(metric_key: str, config: ValidationComparisonConfig) -> str | None:
    if _matches(metric_key, config.lower_is_better):
        return "lower_is_better"
    if _matches(metric_key, config.higher_is_better):
        return "higher_is_better"
    return None


def _matches(metric_key: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if pattern.endswith(":*") and metric_key.startswith(pattern[:-1]):
            return True
        if metric_key == pattern:
            return True
    return False


def _summary(result: object, shared_paired_count: int | None) -> ValidationSurfaceAggregateSummary:
    surface = getattr(result, "surface")
    limitations = getattr(result, "limitations")
    missing_count = (
        limitations.generated_missing_count
        + limitations.reference_missing_count
        + limitations.failed_generated_count
        + limitations.identity_mismatch_count
    )
    issue_count = sum(len(pair.issues) for pair in getattr(result, "pairing"))
    return ValidationSurfaceAggregateSummary(
        surface_label=surface.label,
        producer_key=surface.producer_key,
        producer_type=surface.producer_type,
        run_name=surface.run_name,
        paired_count=limitations.paired_count,
        issue_count=issue_count,
        missing_count=missing_count,
        shared_paired_count=shared_paired_count,
    )


__all__ = [
    "ValidationComparisonResult",
    "ValidationMetricComparisonRow",
    "ValidationSharedPairingKey",
    "ValidationSurfaceAggregateSummary",
    "build_validation_comparison",
]
