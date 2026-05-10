"""Quality-domain leakage detection package."""

from text_to_sign_production.data.tier.leakages.analysis import (
    leakage_relation_frequencies,
    leakage_severity_distribution,
)
from text_to_sign_production.data.tier.leakages.detect import (
    build_leakage_bundle,
    build_leakage_bundle_from_inputs,
)
from text_to_sign_production.data.tier.leakages.overlap import (
    build_leakage_input,
    build_leakage_input_from_manifest,
    detect_pair_relations,
)
from text_to_sign_production.data.tier.leakages.severity import (
    classify_leakage_severity,
    max_leakage_severity,
)
from text_to_sign_production.data.tier.leakages.types import (
    LEAKAGE_RELATION_ORDER,
    LeakageBundle,
    LeakageInput,
    LeakagePairFact,
    LeakageRelation,
    LeakageRelationFrequencyRecord,
    LeakageSampleRef,
    LeakageSampleSummary,
    LeakageSeverity,
    LeakageSeverityDistributionRecord,
    LeakageValidationIssue,
    LeakageValidationIssueCode,
)
from text_to_sign_production.data.tier.leakages.validate import (
    validate_leakage_bundle,
    validate_leakage_input,
)

__all__ = [
    "LEAKAGE_RELATION_ORDER",
    "LeakageBundle",
    "LeakageInput",
    "LeakagePairFact",
    "LeakageRelation",
    "LeakageRelationFrequencyRecord",
    "LeakageSampleRef",
    "LeakageSampleSummary",
    "LeakageSeverity",
    "LeakageSeverityDistributionRecord",
    "LeakageValidationIssue",
    "LeakageValidationIssueCode",
    "build_leakage_bundle",
    "build_leakage_bundle_from_inputs",
    "build_leakage_input",
    "build_leakage_input_from_manifest",
    "classify_leakage_severity",
    "detect_pair_relations",
    "leakage_relation_frequencies",
    "leakage_severity_distribution",
    "max_leakage_severity",
    "validate_leakage_bundle",
    "validate_leakage_input",
]
