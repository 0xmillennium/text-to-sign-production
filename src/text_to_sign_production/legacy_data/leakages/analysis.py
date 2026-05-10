"""Package-local analysis surfaces for leakage facts."""

from __future__ import annotations

from collections import Counter

from text_to_sign_production.legacy_data._shared.analysis import safe_ratio
from text_to_sign_production.legacy_data.leakages.types import (
    LEAKAGE_RELATION_ORDER,
    LeakageAffectedSampleCoverageRecord,
    LeakageBundle,
    LeakageRelation,
    LeakageRelationCooccurrenceRecord,
    LeakageRelationFrequencyRecord,
    LeakageSampleSeverityDistributionRecord,
    LeakageSeverity,
    LeakageSeverityDistributionRecord,
    LeakageSplitPairCountRecord,
)


def build_leakage_relation_frequency_records(
    bundle: LeakageBundle,
) -> tuple[LeakageRelationFrequencyRecord, ...]:
    """Count leakage relations across pair facts."""
    counter = Counter(relation for fact in bundle.pair_facts for relation in fact.relations)
    return tuple(
        LeakageRelationFrequencyRecord(relation=relation, pair_count=counter[relation])
        for relation in LEAKAGE_RELATION_ORDER
        if counter[relation] > 0
    )


def build_leakage_severity_distribution_records(
    bundle: LeakageBundle,
) -> tuple[LeakageSeverityDistributionRecord, ...]:
    """Count leakage pair facts by severity."""
    counter = Counter(fact.severity for fact in bundle.pair_facts)
    return tuple(
        LeakageSeverityDistributionRecord(severity=severity, pair_count=counter[severity])
        for severity in LeakageSeverity
        if counter[severity] > 0
    )


def build_leakage_sample_severity_distribution_records(
    bundle: LeakageBundle,
) -> tuple[LeakageSampleSeverityDistributionRecord, ...]:
    """Count samples by maximum leakage severity within each split."""
    counter = Counter((summary.split, summary.max_severity) for summary in bundle.sample_summaries)
    return tuple(
        LeakageSampleSeverityDistributionRecord(
            split=split,
            severity=severity,
            sample_count=count,
        )
        for (split, severity), count in sorted(
            counter.items(), key=lambda item: (item[0][0].value, item[0][1].value)
        )
    )


def build_leakage_relation_cooccurrence_records(
    bundle: LeakageBundle,
) -> tuple[LeakageRelationCooccurrenceRecord, ...]:
    """Count ordered relation co-occurrences across pair facts."""
    counter: Counter[tuple[LeakageRelation, LeakageRelation]] = Counter()
    for fact in bundle.pair_facts:
        relation_set = set(fact.relations)
        for left in LEAKAGE_RELATION_ORDER:
            if left not in relation_set:
                continue
            for right in LEAKAGE_RELATION_ORDER:
                if right in relation_set:
                    counter[(left, right)] += 1
    return tuple(
        LeakageRelationCooccurrenceRecord(
            left_relation=left,
            right_relation=right,
            pair_count=count,
        )
        for (left, right), count in sorted(
            counter.items(), key=lambda item: (item[0][0].value, item[0][1].value)
        )
    )


def build_leakage_split_pair_count_records(
    bundle: LeakageBundle,
) -> tuple[LeakageSplitPairCountRecord, ...]:
    """Count leakage pair facts by split pair."""
    counter = Counter((fact.left_split, fact.right_split) for fact in bundle.pair_facts)
    return tuple(
        LeakageSplitPairCountRecord(
            left_split=left_split,
            right_split=right_split,
            pair_count=count,
        )
        for (left_split, right_split), count in sorted(
            counter.items(), key=lambda item: (item[0][0].value, item[0][1].value)
        )
    )


def build_leakage_affected_sample_coverage_records(
    bundle: LeakageBundle,
) -> tuple[LeakageAffectedSampleCoverageRecord, ...]:
    """Count samples affected by any leakage by split."""
    splits = sorted(
        {summary.split for summary in bundle.sample_summaries}, key=lambda item: item.value
    )
    records: list[LeakageAffectedSampleCoverageRecord] = []
    for split in splits:
        scoped = tuple(summary for summary in bundle.sample_summaries if summary.split == split)
        affected = sum(1 for summary in scoped if summary.has_leakage)
        records.append(
            LeakageAffectedSampleCoverageRecord(
                split=split,
                sample_count=len(scoped),
                affected_sample_count=affected,
                affected_sample_ratio=safe_ratio(affected, len(scoped)),
            )
        )
    return tuple(records)
