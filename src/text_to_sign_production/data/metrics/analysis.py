"""Package-local analysis surfaces for metric bundles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from typing import Any

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.data._shared.analysis import numeric_summary, safe_ratio
from text_to_sign_production.data.metrics.types import (
    MetricBundle,
    MetricConfidenceSummaryRecord,
    MetricCoverageSummaryRecord,
    MetricDistributionRecord,
    MetricFamily,
    MetricFamilyCountRecord,
    MetricMissingnessRecord,
    MetricObservationRecord,
    MetricSaturationRecord,
)

_NEAR_CONSTANT_UNIQUE_RATIO_THRESHOLD = 0.01

_COVERAGE_SUMMARY_KEYS: frozenset[tuple[MetricFamily, str]] = frozenset(
    {
        (
            MetricFamily.UPPER_BODY_SUPPORT,
            "active_span_upper_body_support_landmark_coverage_ratio",
        ),
        (
            MetricFamily.UPPER_BODY_SUPPORT,
            "full_clip_upper_body_support_landmark_coverage_ratio",
        ),
        (MetricFamily.COVERAGE, "full_body_landmark_coverage_ratio"),
        (MetricFamily.COVERAGE, "left_hand_landmark_coverage_ratio"),
        (MetricFamily.COVERAGE, "right_hand_landmark_coverage_ratio"),
        (MetricFamily.COVERAGE, "any_hand_landmark_coverage_ratio"),
        (MetricFamily.COVERAGE, "face_landmark_coverage_ratio"),
        (
            MetricFamily.MANUAL_DETAIL,
            "active_span_representative_hand_landmark_coverage_ratio",
        ),
        (
            MetricFamily.MANUAL_DETAIL,
            "active_span_representative_hand_fingertip_coverage_ratio",
        ),
        (
            MetricFamily.MANUAL_DETAIL,
            "active_span_representative_hand_distal_chain_coverage_ratio",
        ),
        (MetricFamily.NON_MANUAL_QUALITY, "active_span_face_landmark_coverage_ratio"),
        (MetricFamily.NON_MANUAL_QUALITY, "active_span_upper_face_landmark_coverage_ratio"),
        (MetricFamily.NON_MANUAL_QUALITY, "active_span_lower_face_landmark_coverage_ratio"),
    }
)


def build_metric_observation_records(
    metric_bundles: Sequence[MetricBundle],
) -> tuple[MetricObservationRecord, ...]:
    """Build the numeric metric inventory observed in metric bundles."""
    records: list[MetricObservationRecord] = []
    for family, metric_key in _numeric_metric_keys(metric_bundles):
        values = tuple(_metric_value(bundle, family, metric_key) for bundle in metric_bundles)
        missing_count = sum(1 for value in values if value is None)
        records.append(
            MetricObservationRecord(
                family=family,
                metric_key=metric_key,
                observed_count=len(values) - missing_count,
                missing_count=missing_count,
            )
        )
    return tuple(records)


def build_metric_family_count_records(
    metric_bundles: Sequence[MetricBundle],
) -> tuple[MetricFamilyCountRecord, ...]:
    """Count metric bundles by split and metric family."""
    records: list[MetricFamilyCountRecord] = []
    splits = sorted({bundle.split for bundle in metric_bundles}, key=lambda item: item.value)
    for split in splits:
        sample_count = sum(1 for bundle in metric_bundles if bundle.split == split)
        records.extend(
            MetricFamilyCountRecord(split=split, family=family, sample_count=sample_count)
            for family in MetricFamily
        )
    return tuple(records)


def build_metric_distribution_records(
    metric_bundles: Sequence[MetricBundle],
) -> tuple[MetricDistributionRecord, ...]:
    """Build split-aware numeric distributions for metric bundle values."""
    records: list[MetricDistributionRecord] = []
    for split in _split_scopes(metric_bundles):
        scoped = _filter_by_split(metric_bundles, split)
        for family, metric_key in _numeric_metric_keys(metric_bundles):
            values = tuple(_metric_value(bundle, family, metric_key) for bundle in scoped)
            summary = numeric_summary(values)
            records.append(
                MetricDistributionRecord(
                    split=split,
                    family=family,
                    metric_key=metric_key,
                    sample_count=summary.sample_count,
                    missing_count=summary.missing_count,
                    unique_value_count=summary.unique_value_count,
                    minimum=summary.minimum,
                    p5=summary.p5,
                    p25=summary.p25,
                    p50=summary.p50,
                    p75=summary.p75,
                    p95=summary.p95,
                    p99=summary.p99,
                    maximum=summary.maximum,
                )
            )
    return tuple(records)


def build_metric_missingness_records(
    distribution_records: Sequence[MetricDistributionRecord],
) -> tuple[MetricMissingnessRecord, ...]:
    """Build missingness summaries from metric distributions."""
    return tuple(
        MetricMissingnessRecord(
            split=record.split,
            family=record.family,
            metric_key=record.metric_key,
            sample_count=record.sample_count,
            missing_count=record.missing_count,
            missing_ratio=safe_ratio(record.missing_count, record.sample_count),
        )
        for record in distribution_records
    )


def build_metric_coverage_summary_records(
    distribution_records: Sequence[MetricDistributionRecord],
) -> tuple[MetricCoverageSummaryRecord, ...]:
    """Build dedicated coverage and density summaries from metric distributions."""
    return tuple(
        MetricCoverageSummaryRecord(
            split=record.split,
            family=record.family,
            metric_key=record.metric_key,
            sample_count=record.sample_count,
            minimum=record.minimum,
            p50=record.p50,
            p95=record.p95,
            maximum=record.maximum,
        )
        for record in distribution_records
        if (record.family, record.metric_key) in _COVERAGE_SUMMARY_KEYS
    )


def build_metric_confidence_summary_records(
    distribution_records: Sequence[MetricDistributionRecord],
) -> tuple[MetricConfidenceSummaryRecord, ...]:
    """Build channel-aware confidence summaries from metric distributions."""
    return tuple(
        MetricConfidenceSummaryRecord(
            split=record.split,
            metric_key=record.metric_key,
            sample_count=record.sample_count,
            minimum=record.minimum,
            p50=record.p50,
            p95=record.p95,
            maximum=record.maximum,
        )
        for record in distribution_records
        if record.family == MetricFamily.CONFIDENCE
    )


def build_metric_saturation_records(
    distribution_records: Sequence[MetricDistributionRecord],
) -> tuple[MetricSaturationRecord, ...]:
    """Detect constant and near-constant metric value distributions."""
    records: list[MetricSaturationRecord] = []
    for record in distribution_records:
        reasons: list[str] = []
        if record.sample_count > 0 and record.unique_value_count == 1:
            reasons.append("constant_values")
        elif (
            record.sample_count > 0
            and record.unique_value_count / record.sample_count
            <= _NEAR_CONSTANT_UNIQUE_RATIO_THRESHOLD
        ):
            reasons.append("near_constant_values")
        if reasons:
            records.append(
                MetricSaturationRecord(
                    split=record.split,
                    family=record.family,
                    metric_key=record.metric_key,
                    sample_count=record.sample_count,
                    unique_value_count=record.unique_value_count,
                    reasons=tuple(reasons),
                )
            )
    return tuple(records)


def _numeric_metric_keys(
    metric_bundles: Sequence[MetricBundle],
) -> tuple[tuple[MetricFamily, str], ...]:
    if not metric_bundles:
        return ()
    first = metric_bundles[0]
    keys: list[tuple[MetricFamily, str]] = []
    for family in MetricFamily:
        section = getattr(first, family.value)
        if not is_dataclass(section):
            continue
        for field in fields(section):
            value = getattr(section, field.name)
            if isinstance(value, bool | str):
                continue
            if value is None or isinstance(value, int | float):
                keys.append((family, field.name))
    return tuple(keys)


def _metric_value(
    bundle: MetricBundle, family: MetricFamily, metric_key: str
) -> float | int | None:
    value: Any = getattr(getattr(bundle, family.value), metric_key)
    if value is None:
        return None
    if isinstance(value, bool | str):
        return None
    if isinstance(value, int | float):
        return value
    return None


def _split_scopes(metric_bundles: Sequence[MetricBundle]) -> tuple[SampleSplit | None, ...]:
    splits = tuple(sorted({bundle.split for bundle in metric_bundles}, key=lambda item: item.value))
    return (None, *splits)


def _filter_by_split(
    metric_bundles: Sequence[MetricBundle],
    split: SampleSplit | None,
) -> tuple[MetricBundle, ...]:
    if split is None:
        return tuple(metric_bundles)
    return tuple(bundle for bundle in metric_bundles if bundle.split == split)
