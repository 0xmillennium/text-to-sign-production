"""Upper-body support tier threshold parsing and evaluation."""

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
    FilterLevel,
    TierMetricFailure,
    UpperBodySupportThresholds,
)

_BINDING_SPECS = BINDING_TIER_METRICS_BY_FAMILY[BindingTierFamily.UPPER_BODY_SUPPORT]
_THRESHOLD_KEYS = tuple(spec.threshold_attr for spec in _BINDING_SPECS)
_LEVEL_KEYS = tuple(level.value for level in FilterLevel)


def parse_upper_body_support_thresholds(
    payload: object,
) -> dict[FilterLevel, UpperBodySupportThresholds]:
    """Parse strict upper-body support thresholds for every filter level."""
    levels = require_mapping(payload, "upper_body_support")
    require_exact_keys(levels, _LEVEL_KEYS, "upper_body_support")

    parsed: dict[FilterLevel, UpperBodySupportThresholds] = {}
    for level in FilterLevel:
        level_payload = require_mapping(
            levels[level.value],
            f"upper_body_support.{level.value}",
        )
        require_exact_keys(
            level_payload,
            _THRESHOLD_KEYS,
            f"upper_body_support.{level.value}",
        )
        parsed[level] = UpperBodySupportThresholds(
            min_upper_body_support_landmark_coverage_ratio=require_ratio(
                level_payload["min_upper_body_support_landmark_coverage_ratio"],
                "upper_body_support.min_upper_body_support_landmark_coverage_ratio",
            )
        )
    return parsed


def evaluate_upper_body_support_family(
    bundle: MetricBundle,
    thresholds: UpperBodySupportThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    """Evaluate upper-body support metrics against the applied threshold level."""
    return tuple(
        failure
        for spec in _BINDING_SPECS
        if (failure := evaluate_binding_metric(bundle, thresholds, spec, applied_level))
        is not None
    )
