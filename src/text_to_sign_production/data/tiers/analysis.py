"""Package-local analysis surfaces for tier memberships and decision details."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.data._shared.analysis import comparison_passes, threshold_distance
from text_to_sign_production.data.metrics.types import MetricBundle
from text_to_sign_production.data.tiers.roles import (
    BINDING_TIER_METRICS,
    get_metric_value,
    get_threshold_value,
)
from text_to_sign_production.data.tiers.types import (
    BindingTierFamily,
    BlockerFrequencyRecord,
    CoFailureRecord,
    FilterConfig,
    FilterLevel,
    NearThresholdSampleRecord,
    NearThresholdSide,
    NearThresholdSummaryRecord,
    TierDecisionDetail,
    TierDeltaRecord,
    TierDeltaSummaryRecord,
    TierInclusionCountRecord,
    TierMembership,
    TierMembershipRecord,
    TierMetricPassFailRecord,
    TierName,
)

SampleKey = tuple[SampleSplit, str]

_NEAR_THRESHOLD_LIMIT = 20
_TIER_DELTAS: tuple[tuple[TierName, TierName], ...] = (
    (TierName.LOOSE, TierName.CLEAN),
    (TierName.CLEAN, TierName.TIGHT),
)


def build_tier_inclusion_count_records(
    memberships: Sequence[TierMembershipRecord],
) -> tuple[TierInclusionCountRecord, ...]:
    """Count included and excluded memberships by tier and split."""
    counters: dict[tuple[TierName, SampleSplit], dict[str, int]] = defaultdict(
        lambda: {"included": 0, "excluded": 0}
    )
    for membership in memberships:
        bucket = counters[(membership.tier_name, membership.split)]
        bucket[membership.membership.value] += 1
    return tuple(
        TierInclusionCountRecord(
            tier_name=tier,
            split=split,
            included_count=counts["included"],
            excluded_count=counts["excluded"],
            total_count=counts["included"] + counts["excluded"],
        )
        for (tier, split), counts in sorted(
            counters.items(), key=lambda item: (item[0][0].value, item[0][1].value)
        )
    )


def build_tier_membership_records(
    memberships: Sequence[TierMembershipRecord],
    *,
    tier_name: TierName | str | None = None,
    membership: TierMembership | str | None = None,
) -> tuple[TierMembershipRecord, ...]:
    """Build deterministic file-free tier membership records."""
    selected_tier = TierName(tier_name) if tier_name is not None else None
    selected_membership = TierMembership(membership) if membership is not None else None
    return tuple(
        record
        for record in memberships
        if (selected_tier is None or record.tier_name == selected_tier)
        and (selected_membership is None or record.membership == selected_membership)
    )


def build_included_membership_records(
    memberships: Sequence[TierMembershipRecord],
    *,
    tier_name: TierName | str | None = None,
) -> tuple[TierMembershipRecord, ...]:
    """Build thin membership records for samples included in a tier."""
    return build_tier_membership_records(
        memberships,
        tier_name=tier_name,
        membership=TierMembership.INCLUDED,
    )


def build_excluded_membership_records(
    memberships: Sequence[TierMembershipRecord],
    *,
    tier_name: TierName | str | None = None,
) -> tuple[TierMembershipRecord, ...]:
    """Build thin membership records for samples excluded from a tier."""
    return build_tier_membership_records(
        memberships,
        tier_name=tier_name,
        membership=TierMembership.EXCLUDED,
    )


def build_tier_metric_pass_fail_records(
    decision_details: Sequence[TierDecisionDetail],
    metric_bundles: Sequence[MetricBundle],
    filter_config: FilterConfig,
) -> tuple[TierMetricPassFailRecord, ...]:
    """Count binding metric pass/fail outcomes under applied tier policies."""
    metrics_by_key = _build_metric_lookup(metric_bundles)
    counters: dict[
        tuple[TierName, SampleSplit, BindingTierFamily, str],
        dict[str, int],
    ] = defaultdict(lambda: {"pass": 0, "fail": 0})
    for detail in decision_details:
        bundle = metrics_by_key[(detail.split, detail.sample_id)]
        for spec in BINDING_TIER_METRICS:
            assert spec.comparison is not None
            family = _binding_spec_family(spec)
            level = detail.applied_family_levels[family]
            actual = get_metric_value(bundle, spec)
            expected = get_threshold_value(_thresholds_for_family(filter_config, family, level), spec)
            bucket = counters[(detail.tier_name, detail.split, family, spec.metric_key)]
            bucket["pass" if comparison_passes(actual, expected, spec.comparison) else "fail"] += 1
    return tuple(
        TierMetricPassFailRecord(
            tier_name=tier,
            split=split,
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


def build_primary_blocker_records(
    decision_details: Sequence[TierDecisionDetail],
) -> tuple[BlockerFrequencyRecord, ...]:
    """Aggregate the first metric failure for each excluded decision detail."""
    counters: dict[tuple[TierName, SampleSplit, BindingTierFamily, str], int] = defaultdict(int)
    for detail in decision_details:
        if detail.membership is TierMembership.INCLUDED or not detail.metric_failures:
            continue
        primary = detail.metric_failures[0]
        counters[(detail.tier_name, detail.split, primary.family, primary.metric_key)] += 1
    return _blocker_records(counters)


def build_all_blocker_records(
    decision_details: Sequence[TierDecisionDetail],
) -> tuple[BlockerFrequencyRecord, ...]:
    """Aggregate every metric failure for each excluded decision detail."""
    counters: dict[tuple[TierName, SampleSplit, BindingTierFamily, str], int] = defaultdict(int)
    for detail in decision_details:
        if detail.membership is TierMembership.INCLUDED:
            continue
        for failure in detail.metric_failures:
            counters[(detail.tier_name, detail.split, failure.family, failure.metric_key)] += 1
    return _blocker_records(counters)


def build_cofailure_records(
    decision_details: Sequence[TierDecisionDetail],
) -> tuple[CoFailureRecord, ...]:
    """Build family-level co-failure counts for excluded decision details."""
    counters: dict[tuple[TierName, SampleSplit, BindingTierFamily, BindingTierFamily], int] = (
        defaultdict(int)
    )
    for detail in decision_details:
        if detail.membership is TierMembership.INCLUDED:
            continue
        families = tuple(
            family
            for family in BindingTierFamily
            if any(failure.family == family for failure in detail.metric_failures)
        )
        for left in families:
            for right in families:
                counters[(detail.tier_name, detail.split, left, right)] += 1
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
    decision_details: Sequence[TierDecisionDetail],
    metric_bundles: Sequence[MetricBundle],
    filter_config: FilterConfig,
    *,
    limit_per_metric_side: int = _NEAR_THRESHOLD_LIMIT,
) -> tuple[NearThresholdSampleRecord, ...]:
    """List nearest passing and failing samples by tier/family/metric/split."""
    metrics_by_key = _build_metric_lookup(metric_bundles)
    buckets: dict[
        tuple[TierName, SampleSplit, BindingTierFamily, str, NearThresholdSide],
        list[NearThresholdSampleRecord],
    ] = defaultdict(list)
    for detail in decision_details:
        bundle = metrics_by_key[(detail.split, detail.sample_id)]
        for spec in BINDING_TIER_METRICS:
            assert spec.comparison is not None
            family = _binding_spec_family(spec)
            level = detail.applied_family_levels[family]
            actual = get_metric_value(bundle, spec)
            expected = get_threshold_value(_thresholds_for_family(filter_config, family, level), spec)
            side = (
                NearThresholdSide.PASSING
                if comparison_passes(actual, expected, spec.comparison)
                else NearThresholdSide.FAILING
            )
            buckets[(detail.tier_name, detail.split, family, spec.metric_key, side)].append(
                NearThresholdSampleRecord(
                    tier_name=detail.tier_name,
                    split=detail.split,
                    family=family,
                    metric_key=spec.metric_key,
                    sample_id=detail.sample_id,
                    side=side,
                    actual_value=actual,
                    expected_value=expected,
                    comparison=spec.comparison,
                    threshold_distance=threshold_distance(actual, expected),
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


def build_near_threshold_summary_records(
    near_threshold_samples: Sequence[NearThresholdSampleRecord],
) -> tuple[NearThresholdSummaryRecord, ...]:
    """Count near-threshold samples by tier, split, metric, and side."""
    counters: dict[
        tuple[TierName, SampleSplit, BindingTierFamily, str, NearThresholdSide],
        int,
    ] = defaultdict(int)
    for record in near_threshold_samples:
        counters[
            (
                record.tier_name,
                record.split,
                record.family,
                record.metric_key,
                record.side,
            )
        ] += 1
    return tuple(
        NearThresholdSummaryRecord(
            tier_name=tier,
            split=split,
            family=family,
            metric_key=metric_key,
            side=side,
            sample_count=count,
        )
        for (tier, split, family, metric_key, side), count in sorted(
            counters.items(),
            key=lambda item: (
                item[0][0].value,
                item[0][1].value,
                item[0][2].value,
                item[0][3],
                item[0][4].value,
            ),
        )
    )


def build_tier_delta_records(
    memberships: Sequence[TierMembershipRecord],
) -> tuple[TierDeltaRecord, ...]:
    """Build sample-level adjacent-tier deltas by split."""
    included = _included_by_split_tier(memberships)
    records: list[TierDeltaRecord] = []
    splits = sorted({record.split for record in memberships}, key=lambda item: item.value)
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


def build_tier_delta_summary_records(
    memberships: Sequence[TierMembershipRecord],
) -> tuple[TierDeltaSummaryRecord, ...]:
    """Build split-level adjacent-tier delta counts."""
    included = _included_by_split_tier(memberships)
    records: list[TierDeltaSummaryRecord] = []
    splits = sorted({record.split for record in memberships}, key=lambda item: item.value)
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


def _blocker_records(
    counters: Mapping[tuple[TierName, SampleSplit, BindingTierFamily, str], int],
) -> tuple[BlockerFrequencyRecord, ...]:
    return tuple(
        BlockerFrequencyRecord(
            tier_name=tier,
            split=split,
            family=family,
            metric_key=metric_key,
            blocker_count=count,
        )
        for (tier, split, family, metric_key), count in sorted(
            counters.items(),
            key=lambda item: (item[0][0].value, item[0][1].value, item[0][2].value, item[0][3]),
        )
    )


def _build_metric_lookup(metric_bundles: Sequence[MetricBundle]) -> dict[SampleKey, MetricBundle]:
    lookup: dict[SampleKey, MetricBundle] = {}
    for bundle in metric_bundles:
        lookup[(bundle.split, bundle.sample_id)] = bundle
    return lookup


def _thresholds_for_family(
    filter_config: FilterConfig,
    family: BindingTierFamily,
    level: FilterLevel,
) -> object:
    return getattr(filter_config, family.value)[level]


def _binding_spec_family(spec: object) -> BindingTierFamily:
    family = getattr(spec, "family")
    if not isinstance(family, BindingTierFamily):
        raise TypeError(f"Binding role spec has non-binding family {family!r}.")
    return family


def _included_by_split_tier(
    memberships: Sequence[TierMembershipRecord],
) -> dict[tuple[SampleSplit, TierName], set[str]]:
    included: dict[tuple[SampleSplit, TierName], set[str]] = defaultdict(set)
    for record in memberships:
        if record.membership is TierMembership.INCLUDED:
            included[(record.split, record.tier_name)].add(record.sample_id)
    return included
