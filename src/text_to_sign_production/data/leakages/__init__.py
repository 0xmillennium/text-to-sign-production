"""Policy-free, deterministic leakage-facts layer."""

from __future__ import annotations

from text_to_sign_production.data.leakages.detect import build_leakage_bundle
from text_to_sign_production.data.leakages.overlap import (
    build_leakage_input,
    detect_pair_relations,
)
from text_to_sign_production.data.leakages.severity import classify_leakage_severity
from text_to_sign_production.data.leakages.types import (
    LeakageBundle,
    LeakageInput,
    LeakagePairFact,
    LeakageRelation,
    LeakageSampleRef,
    LeakageSampleSummary,
    LeakageSeverity,
)
from text_to_sign_production.data.leakages.validate import validate_leakage_bundle

__all__ = [
    "LeakageBundle",
    "LeakageInput",
    "LeakagePairFact",
    "LeakageRelation",
    "LeakageSampleRef",
    "LeakageSampleSummary",
    "LeakageSeverity",
    "build_leakage_bundle",
    "build_leakage_input",
    "classify_leakage_severity",
    "detect_pair_relations",
    "validate_leakage_bundle",
]
