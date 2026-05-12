"""Read-only leakage analysis helpers."""

from __future__ import annotations

from collections import Counter

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.data.tier.leakages.types import (
    LeakageBundle,
    LeakageRelation,
    LeakageRelationFrequencyRecord,
    LeakageSampleSummary,
    LeakageSeverity,
    LeakageSeverityDistributionRecord,
)


def leakage_relation_frequencies(
    bundle: LeakageBundle,
) -> tuple[LeakageRelationFrequencyRecord, ...]:
    """Count leakage relations across pair facts."""
    counter = Counter(relation for fact in bundle.pair_facts for relation in fact.relations)
    return tuple(
        LeakageRelationFrequencyRecord(relation=relation, pair_count=counter[relation])
        for relation in LeakageRelation
        if counter[relation] > 0
    )


def leakage_severity_distribution(
    bundle: LeakageBundle,
) -> tuple[LeakageSeverityDistributionRecord, ...]:
    """Count leakage severities across pair facts."""
    counter = Counter(fact.severity for fact in bundle.pair_facts)
    return tuple(
        LeakageSeverityDistributionRecord(severity=severity, pair_count=counter[severity])
        for severity in LeakageSeverity
        if counter[severity] > 0
    )


def sample_leakage_summary(
    bundle: LeakageBundle,
    *,
    split: SampleSplit | str,
    sample_id: str,
) -> LeakageSampleSummary:
    """Return the sample-local leakage summary for one split/sample key."""
    normalized_split = SampleSplit(split)
    for summary in bundle.sample_summaries:
        if summary.split == normalized_split and summary.sample_id == sample_id:
            return summary
    raise KeyError(f"Missing leakage summary for {split!r}/{sample_id!r}")


__all__ = [
    "leakage_relation_frequencies",
    "leakage_severity_distribution",
    "sample_leakage_summary",
]
