"""Typed calibration surfaces for tier threshold review."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from text_to_sign_production.data._shared.identities import SampleSplit
from text_to_sign_production.data.metrics.types import MetricBundle
from text_to_sign_production.data.tiers.types import (
    BindingTierFamily,
    BlockerFrequencyRecord,
    CoFailureRecord,
    ConfidenceChannelSummaryRecord,
    CoverageFamilySummaryRecord,
    DiagnosticMetric,
    FilterConfig,
    FilterLevel,
    MetricDistributionRecord,
    MetricPolicyRole,
    NearThresholdSampleRecord,
    NearThresholdSide,
    SaturatedMetricRecord,
    SplitScope,
    TierBundle,
    TierCalibrationSurfaces,
    TierDecision,
    TierDeltaRecord,
    TierDeltaSummaryRecord,
    TierMetricPassFailRecord,
    TierName,
)

MetricValueFn = Callable[[MetricBundle], float | int | None]
ThresholdValueFn = Callable[[FilterConfig, FilterLevel], float | int]

_QUANTILES = {
    "p5": 0.05,
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
    "p95": 0.95,
    "p99": 0.99,
}
_SATURATION_RATE_THRESHOLD = 0.99
_NEAR_CONSTANT_UNIQUE_RATIO_THRESHOLD = 0.01
_NEAR_THRESHOLD_LIMIT = 20
_TIER_DELTAS: tuple[tuple[TierName, TierName], ...] = (
    (TierName.LOOSE, TierName.CLEAN),
    (TierName.CLEAN, TierName.TIGHT),
)


@dataclass(frozen=True, slots=True)
class _CalibrationMetricSpec:
    """Internal metric definition used to build calibration records."""

    role: MetricPolicyRole
    family: str
    metric_key: str
    value_fn: MetricValueFn
    comparison: str | None = None
    threshold_fn: ThresholdValueFn | None = None


_BINDING_METRICS: tuple[_CalibrationMetricSpec, ...] = (
    _CalibrationMetricSpec(
        role=MetricPolicyRole.BINDING,
        family=BindingTierFamily.OOB.value,
        metric_key="out_of_bounds_ratio",
        value_fn=lambda bundle: bundle.oob.out_of_bounds_ratio,
        comparison="<=",
        threshold_fn=lambda config, level: config.oob[level].max_out_of_bounds_ratio,
    ),
    _CalibrationMetricSpec(
        role=MetricPolicyRole.BINDING,
        family=BindingTierFamily.COVERAGE.value,
        metric_key="body_landmark_coverage_ratio",
        value_fn=lambda bundle: bundle.coverage.body_landmark_coverage_ratio,
        comparison=">=",
        threshold_fn=lambda config, level: (
            config.coverage[level].min_body_landmark_coverage_ratio
        ),
    ),
    _CalibrationMetricSpec(
        role=MetricPolicyRole.BINDING,
        family=BindingTierFamily.COVERAGE.value,
        metric_key="any_hand_landmark_coverage_ratio",
        value_fn=lambda bundle: bundle.coverage.any_hand_landmark_coverage_ratio,
        comparison=">=",
        threshold_fn=lambda config, level: (
            config.coverage[level].min_any_hand_landmark_coverage_ratio
        ),
    ),
    _CalibrationMetricSpec(
        role=MetricPolicyRole.BINDING,
        family=BindingTierFamily.COVERAGE.value,
        metric_key="face_landmark_coverage_ratio",
        value_fn=lambda bundle: bundle.coverage.face_landmark_coverage_ratio,
        comparison=">=",
        threshold_fn=lambda config, level: (
            config.coverage[level].min_face_landmark_coverage_ratio
        ),
    ),
    _CalibrationMetricSpec(
        role=MetricPolicyRole.BINDING,
        family=BindingTierFamily.HAND.value,
        metric_key="any_hand_available_frame_ratio",
        value_fn=lambda bundle: bundle.hand.any_hand_available_frame_ratio,
        comparison=">=",
        threshold_fn=lambda config, level: config.hand[level].min_any_hand_available_frame_ratio,
    ),
    _CalibrationMetricSpec(
        role=MetricPolicyRole.BINDING,
        family=BindingTierFamily.HAND.value,
        metric_key="max_any_hand_unavailable_run_ratio",
        value_fn=lambda bundle: bundle.hand.max_any_hand_unavailable_run_ratio,
        comparison="<=",
        threshold_fn=lambda config, level: config.hand[level].max_any_hand_unavailable_run_ratio,
    ),
    _CalibrationMetricSpec(
        role=MetricPolicyRole.BINDING,
        family=BindingTierFamily.FACE.value,
        metric_key="face_available_frame_ratio",
        value_fn=lambda bundle: bundle.face.face_available_frame_ratio,
        comparison=">=",
        threshold_fn=lambda config, level: config.face[level].min_face_available_frame_ratio,
    ),
    _CalibrationMetricSpec(
        role=MetricPolicyRole.BINDING,
        family=BindingTierFamily.CONFIDENCE.value,
        metric_key="body_mean_confidence",
        value_fn=lambda bundle: bundle.confidence.body_mean_confidence,
        comparison=">=",
        threshold_fn=lambda config, level: config.confidence[level].min_body_mean_confidence,
    ),
    _CalibrationMetricSpec(
        role=MetricPolicyRole.BINDING,
        family=BindingTierFamily.CONFIDENCE.value,
        metric_key="left_hand_mean_confidence",
        value_fn=lambda bundle: bundle.confidence.left_hand_mean_confidence,
        comparison=">=",
        threshold_fn=lambda config, level: (
            config.confidence[level].min_left_hand_mean_confidence
        ),
    ),
    _CalibrationMetricSpec(
        role=MetricPolicyRole.BINDING,
        family=BindingTierFamily.CONFIDENCE.value,
        metric_key="right_hand_mean_confidence",
        value_fn=lambda bundle: bundle.confidence.right_hand_mean_confidence,
        comparison=">=",
        threshold_fn=lambda config, level: (
            config.confidence[level].min_right_hand_mean_confidence
        ),
    ),
    _CalibrationMetricSpec(
        role=MetricPolicyRole.BINDING,
        family=BindingTierFamily.CONFIDENCE.value,
        metric_key="face_mean_confidence",
        value_fn=lambda bundle: bundle.confidence.face_mean_confidence,
        comparison=">=",
        threshold_fn=lambda config, level: config.confidence[level].min_face_mean_confidence,
    ),
    _CalibrationMetricSpec(
        role=MetricPolicyRole.BINDING,
        family=BindingTierFamily.TEXT.value,
        metric_key="character_count",
        value_fn=lambda bundle: bundle.text.character_count,
        comparison=">=",
        threshold_fn=lambda config, level: config.text[level].min_character_count,
    ),
    _CalibrationMetricSpec(
        role=MetricPolicyRole.BINDING,
        family=BindingTierFamily.TEXT.value,
        metric_key="token_count",
        value_fn=lambda bundle: bundle.text.token_count,
        comparison=">=",
        threshold_fn=lambda config, level: config.text[level].min_token_count,
    ),
    _CalibrationMetricSpec(
        role=MetricPolicyRole.BINDING,
        family=BindingTierFamily.LENGTH.value,
        metric_key="num_frames",
        value_fn=lambda bundle: bundle.length.num_frames,
        comparison=">=",
        threshold_fn=lambda config, level: config.length[level].min_num_frames,
    ),
    _CalibrationMetricSpec(
        role=MetricPolicyRole.BINDING,
        family=BindingTierFamily.LENGTH.value,
        metric_key="duration_seconds",
        value_fn=lambda bundle: bundle.length.duration_seconds,
        comparison=">=",
        threshold_fn=lambda config, level: config.length[level].min_duration_seconds,
    ),
)

_DIAGNOSTIC_METRICS: tuple[_CalibrationMetricSpec, ...] = (
    _CalibrationMetricSpec(
        role=MetricPolicyRole.DIAGNOSTIC,
        family="valid",
        metric_key=DiagnosticMetric.VALID_FRAME_RATIO.value,
        value_fn=lambda bundle: bundle.valid.valid_frame_ratio,
    ),
    _CalibrationMetricSpec(
        role=MetricPolicyRole.DIAGNOSTIC,
        family="valid",
        metric_key=DiagnosticMetric.INVALID_FRAME_RATIO.value,
        value_fn=lambda bundle: bundle.valid.invalid_frame_ratio,
    ),
    _CalibrationMetricSpec(
        role=MetricPolicyRole.DIAGNOSTIC,
        family="valid",
        metric_key=DiagnosticMetric.ZEROED_CANONICAL_JOINT_FRAME_RATIO.value,
        value_fn=lambda bundle: bundle.valid.zeroed_canonical_joint_frame_ratio,
    ),
    _CalibrationMetricSpec(
        role=MetricPolicyRole.DIAGNOSTIC,
        family=BindingTierFamily.LENGTH.value,
        metric_key=DiagnosticMetric.FRAMES_PER_TOKEN.value,
        value_fn=lambda bundle: bundle.length.frames_per_token,
    ),
    _CalibrationMetricSpec(
        role=MetricPolicyRole.DIAGNOSTIC,
        family=BindingTierFamily.LENGTH.value,
        metric_key=DiagnosticMetric.FRAMES_PER_CHARACTER.value,
        value_fn=lambda bundle: bundle.length.frames_per_character,
    ),
)


def build_tier_calibration_surfaces(
    metric_bundles: Sequence[MetricBundle],
    tier_bundle: TierBundle,
    filter_config: FilterConfig,
) -> TierCalibrationSurfaces:
    """Build reusable calibration surfaces without changing tier decisions."""
    metrics_by_key = {
        (bundle.split, bundle.sample_id): bundle
        for bundle in metric_bundles
    }
    decisions = tuple(tier_bundle.decisions)
    pass_fail_counts = _build_pass_fail_counts(decisions, metrics_by_key, filter_config)
    distributions = build_metric_distribution_records(metric_bundles)
    return TierCalibrationSurfaces(
        metric_distributions=distributions,
        pass_fail_counts=pass_fail_counts,
        primary_blockers=build_primary_blocker_records(tier_bundle),
        all_blockers=build_all_blocker_records(tier_bundle),
        cofailures=build_cofailure_records(tier_bundle),
        saturated_metrics=build_saturated_metric_records(distributions, pass_fail_counts),
        near_threshold_samples=build_near_threshold_sample_records(
            decisions,
            metrics_by_key,
            filter_config,
        ),
        tier_delta_samples=build_tier_delta_records(tier_bundle),
        tier_delta_summaries=build_tier_delta_summary_records(tier_bundle),
        coverage_summaries=build_coverage_family_summary_records(distributions),
        confidence_channel_summaries=build_confidence_channel_summary_records(distributions),
    )


def build_metric_distribution_records(
    metric_bundles: Sequence[MetricBundle],
) -> tuple[MetricDistributionRecord, ...]:
    """Build split and aggregate quantiles for binding and diagnostic metrics."""
    specs = (*_BINDING_METRICS, *_DIAGNOSTIC_METRICS)
    records: list[MetricDistributionRecord] = []
    for split in _split_scopes(metric_bundles):
        scoped = _filter_by_split(metric_bundles, split)
        for spec in specs:
            values = tuple(spec.value_fn(bundle) for bundle in scoped)
            records.append(_distribution_record(split, spec, values))
    return tuple(records)


def build_primary_blocker_records(tier_bundle: TierBundle) -> tuple[BlockerFrequencyRecord, ...]:
    """Aggregate the deterministic first metric failure for each excluded decision."""
    counters: dict[tuple[TierName, SampleSplit, BindingTierFamily, str], int] = defaultdict(int)
    for decision in tier_bundle.decisions:
        if decision.included or not decision.metric_failures:
            continue
        primary = decision.metric_failures[0]
        counters[(decision.tier_name, decision.split, primary.family, primary.metric_key)] += 1
    return _blocker_records(counters)


def build_all_blocker_records(tier_bundle: TierBundle) -> tuple[BlockerFrequencyRecord, ...]:
    """Aggregate every metric failure for each excluded decision."""
    counters: dict[tuple[TierName, SampleSplit, BindingTierFamily, str], int] = defaultdict(int)
    for decision in tier_bundle.decisions:
        if decision.included:
            continue
        for failure in decision.metric_failures:
            counters[(decision.tier_name, decision.split, failure.family, failure.metric_key)] += 1
    return _blocker_records(counters)


def build_cofailure_records(tier_bundle: TierBundle) -> tuple[CoFailureRecord, ...]:
    """Build family-level co-failure counts for excluded decisions."""
    counters: dict[tuple[TierName, SampleSplit, BindingTierFamily, BindingTierFamily], int] = (
        defaultdict(int)
    )
    for decision in tier_bundle.decisions:
        if decision.included:
            continue
        families = tuple(
            family for family in BindingTierFamily
            if any(failure.family == family for failure in decision.metric_failures)
        )
        for left in families:
            for right in families:
                counters[(decision.tier_name, decision.split, left, right)] += 1
    return tuple(
        CoFailureRecord(
            tier_name=tier,
            split=split,
            left_family=left,
            right_family=right,
            decision_count=count,
        )
        for (tier, split, left, right), count in sorted(
            counters.items(),
            key=lambda item: (
                item[0][0].value,
                item[0][1].value,
                item[0][2].value,
                item[0][3].value,
            ),
        )
    )


def build_near_threshold_sample_records(
    decisions: Sequence[TierDecision],
    metrics_by_key: Mapping[tuple[SampleSplit, str], MetricBundle],
    filter_config: FilterConfig,
    *,
    limit_per_metric_side: int = _NEAR_THRESHOLD_LIMIT,
) -> tuple[NearThresholdSampleRecord, ...]:
    """List nearest passing and failing samples by tier/family/metric/split."""
    buckets: dict[
        tuple[TierName, SampleSplit, BindingTierFamily, str, NearThresholdSide],
        list[NearThresholdSampleRecord],
    ] = defaultdict(list)
    for decision in decisions:
        bundle = metrics_by_key[(decision.split, decision.sample_id)]
        for spec in _BINDING_METRICS:
            assert spec.comparison is not None
            assert spec.threshold_fn is not None
            family = BindingTierFamily(spec.family)
            level = decision.applied_family_levels[family]
            actual = spec.value_fn(bundle)
            expected = spec.threshold_fn(filter_config, level)
            passes = _passes(actual, expected, spec.comparison)
            side = NearThresholdSide.PASSING if passes else NearThresholdSide.FAILING
            distance = _threshold_distance(actual, expected)
            buckets[(decision.tier_name, decision.split, family, spec.metric_key, side)].append(
                NearThresholdSampleRecord(
                    tier_name=decision.tier_name,
                    split=decision.split,
                    role=MetricPolicyRole.BINDING,
                    family=family,
                    metric_key=spec.metric_key,
                    sample_id=decision.sample_id,
                    side=side,
                    actual_value=actual,
                    expected_value=expected,
                    comparison=spec.comparison,
                    threshold_distance=distance,
                )
            )

    records: list[NearThresholdSampleRecord] = []
    for key in sorted(
        buckets,
        key=lambda item: (item[0].value, item[1].value, item[2].value, item[3], item[4].value),
    ):
        records.extend(
            sorted(
                buckets[key],
                key=lambda record: (record.threshold_distance, record.sample_id),
            )[:limit_per_metric_side]
        )
    return tuple(records)


def build_tier_delta_records(tier_bundle: TierBundle) -> tuple[TierDeltaRecord, ...]:
    """Build sample-level adjacent-tier deltas by split."""
    included = _included_by_split_tier(tier_bundle)
    records: list[TierDeltaRecord] = []
    splits = sorted(
        {decision.split for decision in tier_bundle.decisions},
        key=lambda item: item.value,
    )
    for split in splits:
        for from_tier, to_tier in _TIER_DELTAS:
            lost = sorted(included[(split, from_tier)] - included[(split, to_tier)])
            records.extend(
                TierDeltaRecord(
                    split=split,
                    from_tier=from_tier,
                    to_tier=to_tier,
                    sample_id=sample_id,
                )
                for sample_id in lost
            )
    return tuple(records)


def build_tier_delta_summary_records(tier_bundle: TierBundle) -> tuple[TierDeltaSummaryRecord, ...]:
    """Build split-level adjacent-tier delta counts."""
    included = _included_by_split_tier(tier_bundle)
    records: list[TierDeltaSummaryRecord] = []
    splits = sorted(
        {decision.split for decision in tier_bundle.decisions},
        key=lambda item: item.value,
    )
    for split in splits:
        for from_tier, to_tier in _TIER_DELTAS:
            records.append(
                TierDeltaSummaryRecord(
                    split=split,
                    from_tier=from_tier,
                    to_tier=to_tier,
                    sample_count=len(included[(split, from_tier)] - included[(split, to_tier)]),
                )
            )
    return tuple(records)


def build_coverage_family_summary_records(
    distribution_records: Sequence[MetricDistributionRecord],
) -> tuple[CoverageFamilySummaryRecord, ...]:
    """Build dedicated coverage-family summaries from distribution records."""
    coverage_metrics = {
        "body_landmark_coverage_ratio",
        "any_hand_landmark_coverage_ratio",
        "face_landmark_coverage_ratio",
    }
    return tuple(
        CoverageFamilySummaryRecord(
            split=record.split,
            metric_key=record.metric_key,
            sample_count=record.sample_count,
            minimum=record.minimum,
            p50=record.p50,
            p95=record.p95,
            maximum=record.maximum,
        )
        for record in distribution_records
        if record.role is MetricPolicyRole.BINDING
        and record.family == BindingTierFamily.COVERAGE.value
        and record.metric_key in coverage_metrics
    )


def build_confidence_channel_summary_records(
    distribution_records: Sequence[MetricDistributionRecord],
) -> tuple[ConfidenceChannelSummaryRecord, ...]:
    """Build dedicated channel-aware confidence summaries from distribution records."""
    confidence_metrics = {
        "body_mean_confidence",
        "left_hand_mean_confidence",
        "right_hand_mean_confidence",
        "face_mean_confidence",
    }
    return tuple(
        ConfidenceChannelSummaryRecord(
            split=record.split,
            metric_key=record.metric_key,
            sample_count=record.sample_count,
            minimum=record.minimum,
            p50=record.p50,
            p95=record.p95,
            maximum=record.maximum,
        )
        for record in distribution_records
        if record.role is MetricPolicyRole.BINDING
        and record.family == BindingTierFamily.CONFIDENCE.value
        and record.metric_key in confidence_metrics
    )


def build_saturated_metric_records(
    distribution_records: Sequence[MetricDistributionRecord],
    pass_fail_records: Sequence[TierMetricPassFailRecord],
) -> tuple[SaturatedMetricRecord, ...]:
    """Detect constant/near-constant metrics and near-total pass/fail rates."""
    records: list[SaturatedMetricRecord] = []
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
                SaturatedMetricRecord(
                    split=record.split,
                    tier_name=None,
                    role=record.role,
                    family=record.family,
                    metric_key=record.metric_key,
                    sample_count=record.sample_count,
                    pass_rate=None,
                    fail_rate=None,
                    unique_value_count=record.unique_value_count,
                    reasons=tuple(reasons),
                )
            )

    for pass_fail_record in pass_fail_records:
        total = pass_fail_record.pass_count + pass_fail_record.fail_count
        if total <= 0:
            continue
        pass_rate = pass_fail_record.pass_count / total
        fail_rate = pass_fail_record.fail_count / total
        reasons = []
        if pass_rate >= _SATURATION_RATE_THRESHOLD:
            reasons.append("near_total_pass_rate")
        if fail_rate >= _SATURATION_RATE_THRESHOLD:
            reasons.append("near_total_fail_rate")
        if not reasons:
            continue
        records.append(
            SaturatedMetricRecord(
                split=pass_fail_record.split,
                tier_name=pass_fail_record.tier_name,
                role=pass_fail_record.role,
                family=pass_fail_record.family.value,
                metric_key=pass_fail_record.metric_key,
                sample_count=total,
                pass_rate=pass_rate,
                fail_rate=fail_rate,
                unique_value_count=None,
                reasons=tuple(reasons),
            )
        )
    return tuple(records)


def _build_pass_fail_counts(
    decisions: Sequence[TierDecision],
    metrics_by_key: Mapping[tuple[SampleSplit, str], MetricBundle],
    filter_config: FilterConfig,
) -> tuple[TierMetricPassFailRecord, ...]:
    counters: dict[
        tuple[TierName, SampleSplit, BindingTierFamily, str],
        dict[str, int],
    ] = defaultdict(lambda: {"pass": 0, "fail": 0})
    for decision in decisions:
        bundle = metrics_by_key[(decision.split, decision.sample_id)]
        for spec in _BINDING_METRICS:
            assert spec.comparison is not None
            assert spec.threshold_fn is not None
            family = BindingTierFamily(spec.family)
            level = decision.applied_family_levels[family]
            actual = spec.value_fn(bundle)
            expected = spec.threshold_fn(filter_config, level)
            bucket = counters[(decision.tier_name, decision.split, family, spec.metric_key)]
            bucket["pass" if _passes(actual, expected, spec.comparison) else "fail"] += 1
    return tuple(
        TierMetricPassFailRecord(
            tier_name=tier,
            split=split,
            role=MetricPolicyRole.BINDING,
            family=family,
            metric_key=metric_key,
            pass_count=counts["pass"],
            fail_count=counts["fail"],
        )
        for (tier, split, family, metric_key), counts in sorted(
            counters.items(),
            key=lambda item: (item[0][0].value, item[0][1].value, item[0][2].value, item[0][3]),
        )
    )


def _distribution_record(
    split: SplitScope,
    spec: _CalibrationMetricSpec,
    raw_values: Sequence[float | int | None],
) -> MetricDistributionRecord:
    values = sorted(float(value) for value in raw_values if value is not None)
    summary = _summary(values)
    return MetricDistributionRecord(
        split=split,
        role=spec.role,
        family=spec.family,
        metric_key=spec.metric_key,
        sample_count=len(raw_values),
        missing_count=len(raw_values) - len(values),
        unique_value_count=len(set(values)),
        minimum=summary["min"],
        p5=summary["p5"],
        p25=summary["p25"],
        p50=summary["p50"],
        p75=summary["p75"],
        p95=summary["p95"],
        p99=summary["p99"],
        maximum=summary["max"],
    )


def _summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "p5": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "min": values[0],
        "p5": _quantile(values, _QUANTILES["p5"]),
        "p25": _quantile(values, _QUANTILES["p25"]),
        "p50": _quantile(values, _QUANTILES["p50"]),
        "p75": _quantile(values, _QUANTILES["p75"]),
        "p95": _quantile(values, _QUANTILES["p95"]),
        "p99": _quantile(values, _QUANTILES["p99"]),
        "max": values[-1],
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _blocker_records(
    counters: Mapping[tuple[TierName, SampleSplit, BindingTierFamily, str], int],
) -> tuple[BlockerFrequencyRecord, ...]:
    return tuple(
        BlockerFrequencyRecord(
            tier_name=tier,
            split=split,
            role=MetricPolicyRole.BINDING,
            family=family,
            metric_key=metric_key,
            blocker_count=count,
        )
        for (tier, split, family, metric_key), count in sorted(
            counters.items(),
            key=lambda item: (item[0][0].value, item[0][1].value, item[0][2].value, item[0][3]),
        )
    )


def _split_scopes(metric_bundles: Sequence[MetricBundle]) -> tuple[SplitScope, ...]:
    splits = tuple(sorted({bundle.split for bundle in metric_bundles}, key=lambda item: item.value))
    return (None, *splits)


def _filter_by_split(
    metric_bundles: Sequence[MetricBundle], split: SplitScope
) -> tuple[MetricBundle, ...]:
    if split is None:
        return tuple(metric_bundles)
    return tuple(bundle for bundle in metric_bundles if bundle.split == split)


def _passes(
    actual_value: float | int | None,
    expected_value: float | int,
    comparison: str,
) -> bool:
    if actual_value is None:
        return False
    if comparison == ">=":
        return actual_value >= expected_value
    if comparison == "<=":
        return actual_value <= expected_value
    raise ValueError(f"Unsupported comparison {comparison!r}")


def _threshold_distance(
    actual_value: float | int | None,
    expected_value: float | int,
) -> float:
    if actual_value is None:
        return math.inf
    return abs(float(actual_value) - float(expected_value))


def _included_by_split_tier(
    tier_bundle: TierBundle,
) -> dict[tuple[SampleSplit, TierName], set[str]]:
    included: dict[tuple[SampleSplit, TierName], set[str]] = defaultdict(set)
    for decision in tier_bundle.decisions:
        if decision.included:
            included[(decision.split, decision.tier_name)].add(decision.sample_id)
    return included
