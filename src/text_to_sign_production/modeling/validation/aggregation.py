"""Deterministic aggregation for validation metric records."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.data.bfh_schema import FULL_BFH_CHANNELS
from text_to_sign_production.modeling.validation.errors import ModelValidationError
from text_to_sign_production.modeling.validation.identifiers import (
    ValidationChannelMetricKey,
    ValidationMetricKey,
)
from text_to_sign_production.modeling.validation.records import (
    VALIDATION_SCHEMA_VERSION,
    ValidationAggregateMetric,
    ValidationChannelMetricResult,
    ValidationMetricResult,
)


def aggregate_validation_metric_results(
    results: Iterable[ValidationMetricResult | ValidationChannelMetricResult],
    *,
    split: SampleSplit,
    missing_count: int,
    issue_count: int,
) -> tuple[ValidationAggregateMetric, ...]:
    """Aggregate finite observations while preserving unavailable metrics as None."""

    resolved_split = SampleSplit(split)
    if resolved_split is not SampleSplit.VAL:
        raise ModelValidationError("validation aggregate metrics must use split='val'.")
    if not isinstance(missing_count, int) or isinstance(missing_count, bool) or missing_count < 0:
        raise ModelValidationError("missing_count must be a non-negative integer.")
    if not isinstance(issue_count, int) or isinstance(issue_count, bool) or issue_count < 0:
        raise ModelValidationError("issue_count must be a non-negative integer.")
    materialized = tuple(results)
    if any(result.split is not resolved_split for result in materialized):
        raise ModelValidationError("aggregate result split does not match validation split.")
    has_regular = any(isinstance(result, ValidationMetricResult) for result in materialized)
    has_channel = any(isinstance(result, ValidationChannelMetricResult) for result in materialized)
    keys: list[str] = []
    if has_regular:
        keys.extend(metric.value for metric in ValidationMetricKey)
    if has_channel:
        keys.extend(
            f"{metric.value}:{channel.value}"
            for metric in ValidationChannelMetricKey
            for channel in FULL_BFH_CHANNELS
        )
    values: dict[str, list[float]] = {key: [] for key in keys}
    for result in materialized:
        if isinstance(result, ValidationMetricResult):
            metric_key = result.metric_key.value
        elif isinstance(result, ValidationChannelMetricResult):
            metric_key = f"{result.metric_key.value}:{result.channel.value}"
        else:
            raise ModelValidationError("unsupported validation metric result type.")
        if not np.isfinite(result.value):
            raise ModelValidationError(f"non-finite metric value cannot be aggregated: {metric_key}.")
        values.setdefault(metric_key, []).append(float(result.value))
    return tuple(
        _aggregate(metric_key, values[metric_key], resolved_split, missing_count, issue_count)
        for metric_key in keys
    )


def _aggregate(
    metric_key: str,
    values: list[float],
    split: SampleSplit,
    missing_count: int,
    issue_count: int,
) -> ValidationAggregateMetric:
    if not values:
        return ValidationAggregateMetric(
            VALIDATION_SCHEMA_VERSION,
            metric_key,
            split,
            0,
            None,
            None,
            None,
            None,
            None,
            missing_count,
            issue_count,
        )
    array = np.asarray(values, dtype=np.float64)
    return ValidationAggregateMetric(
        VALIDATION_SCHEMA_VERSION,
        metric_key,
        split,
        int(array.size),
        float(np.mean(array)),
        float(np.median(array)),
        float(np.min(array)),
        float(np.max(array)),
        float(np.std(array)),
        missing_count,
        issue_count,
    )


__all__ = ["aggregate_validation_metric_results"]
