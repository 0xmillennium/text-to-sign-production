"""Policy-free, deterministic leakage-facts layer."""

from __future__ import annotations

from text_to_sign_production.data.leakages.analysis import (
    build_leakage_affected_sample_coverage_records,
    build_leakage_relation_cooccurrence_records,
    build_leakage_relation_frequency_records,
    build_leakage_sample_severity_distribution_records,
    build_leakage_severity_distribution_records,
    build_leakage_split_pair_count_records,
)
from text_to_sign_production.data.leakages.detect import (
    LeakageProgressEvent,
    LeakageProgressSink,
    build_leakage_bundle,
)
from text_to_sign_production.data.leakages.overlap import (
    build_leakage_input,
    detect_pair_relations,
)
from text_to_sign_production.data.leakages.severity import (
    LEAKAGE_SEVERITY_RANK,
    classify_leakage_severity,
    max_leakage_severity,
)
from text_to_sign_production.data.leakages.types import (
    LeakageAffectedSampleCoverageRecord,
    LeakageBundle,
    LeakageInput,
    LeakagePairFact,
    LeakageRelation,
    LeakageRelationCooccurrenceRecord,
    LeakageRelationFrequencyRecord,
    LeakageSampleRef,
    LeakageSampleSeverityDistributionRecord,
    LeakageSampleSummary,
    LeakageSeverity,
    LeakageSeverityDistributionRecord,
    LeakageSplitPairCountRecord,
    LeakageValidationIssue,
)
from text_to_sign_production.data.leakages.validate import validate_leakage_bundle

__all__ = [
    "LeakageAffectedSampleCoverageRecord",
    "LeakageBundle",
    "LeakageInput",
    "LeakagePairFact",
    "LeakageProgressEvent",
    "LeakageProgressSink",
    "LeakageRelation",
    "LeakageRelationCooccurrenceRecord",
    "LeakageRelationFrequencyRecord",
    "LeakageSampleRef",
    "LeakageSampleSummary",
    "LeakageSampleSeverityDistributionRecord",
    "LeakageSeverity",
    "LeakageSeverityDistributionRecord",
    "LeakageSplitPairCountRecord",
    "LeakageValidationIssue",
    "LEAKAGE_SEVERITY_RANK",
    "build_leakage_affected_sample_coverage_records",
    "build_leakage_bundle",
    "build_leakage_input",
    "build_leakage_relation_cooccurrence_records",
    "build_leakage_relation_frequency_records",
    "build_leakage_sample_severity_distribution_records",
    "build_leakage_severity_distribution_records",
    "build_leakage_split_pair_count_records",
    "classify_leakage_severity",
    "detect_pair_relations",
    "max_leakage_severity",
    "validate_leakage_bundle",
]
