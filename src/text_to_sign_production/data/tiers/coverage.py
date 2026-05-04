"""Coverage tier threshold parsing and evaluation."""

from __future__ import annotations

from text_to_sign_production.data.metrics.types import MetricBundle
from text_to_sign_production.data.tiers._shared.parsing import (
    require_exact_keys,
    require_mapping,
    require_ratio,
)
from text_to_sign_production.data.tiers.roles import (
    BINDING_TIER_METRICS_BY_FAMILY,
    evaluate_binding_metric,
)
from text_to_sign_production.data.tiers.types import (
    BindingTierFamily,
    CoverageThresholds,
    FilterLevel,
    TierMetricFailure,
)

_BINDING_SPECS = BINDING_TIER_METRICS_BY_FAMILY[BindingTierFamily.COVERAGE]
_THRESHOLD_KEYS = tuple(spec.threshold_attr for spec in _BINDING_SPECS)
_LEVEL_KEYS = tuple(level.value for level in FilterLevel)


def parse_coverage_thresholds(payload: object) -> dict[FilterLevel, CoverageThresholds]:
    """Parse strict coverage thresholds for every filter level."""
    levels = require_mapping(payload, "coverage")
    require_exact_keys(levels, _LEVEL_KEYS, "coverage")

    parsed: dict[FilterLevel, CoverageThresholds] = {}
    for level in FilterLevel:
        level_payload = require_mapping(levels[level.value], f"coverage.{level.value}")
        require_exact_keys(level_payload, _THRESHOLD_KEYS, f"coverage.{level.value}")
        parsed[level] = CoverageThresholds(
            min_signing_relevant_body_landmark_coverage_ratio=require_ratio(
                level_payload["min_signing_relevant_body_landmark_coverage_ratio"],
                "coverage.min_signing_relevant_body_landmark_coverage_ratio",
            )
        )
    return parsed


def evaluate_coverage_family(
    bundle: MetricBundle,
    thresholds: CoverageThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    """Evaluate coverage metrics against the applied threshold level."""
    return tuple(
        failure
        for spec in _BINDING_SPECS
        if (failure := evaluate_binding_metric(bundle, thresholds, spec, applied_level))
        is not None
    )
