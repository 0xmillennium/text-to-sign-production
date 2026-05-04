"""Length tier threshold parsing and evaluation."""

from __future__ import annotations

from text_to_sign_production.data.metrics.types import MetricBundle
from text_to_sign_production.data.tiers._shared.parsing import (
    require_exact_keys,
    require_mapping,
    require_positive_float,
    require_positive_int,
)
from text_to_sign_production.data.tiers.roles import (
    BINDING_TIER_METRICS_BY_FAMILY,
    evaluate_binding_metric,
)
from text_to_sign_production.data.tiers.types import (
    BindingTierFamily,
    FilterLevel,
    LengthThresholds,
    TierMetricFailure,
)

_BINDING_SPECS = BINDING_TIER_METRICS_BY_FAMILY[BindingTierFamily.LENGTH]
_THRESHOLD_KEYS = tuple(spec.threshold_attr for spec in _BINDING_SPECS)
_LEVEL_KEYS = tuple(level.value for level in FilterLevel)


def parse_length_thresholds(payload: object) -> dict[FilterLevel, LengthThresholds]:
    """Parse strict length thresholds for every filter level."""
    levels = require_mapping(payload, "length")
    require_exact_keys(levels, _LEVEL_KEYS, "length")

    parsed: dict[FilterLevel, LengthThresholds] = {}
    for level in FilterLevel:
        level_payload = require_mapping(levels[level.value], f"length.{level.value}")
        require_exact_keys(level_payload, _THRESHOLD_KEYS, f"length.{level.value}")
        parsed[level] = LengthThresholds(
            min_num_frames=require_positive_int(
                level_payload["min_num_frames"],
                "length.min_num_frames",
            ),
            min_duration_seconds=require_positive_float(
                level_payload["min_duration_seconds"],
                "length.min_duration_seconds",
            ),
        )
    return parsed


def evaluate_length_family(
    bundle: MetricBundle,
    thresholds: LengthThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    """Evaluate length metrics against the applied threshold level."""
    return tuple(
        failure
        for spec in _BINDING_SPECS
        if (failure := evaluate_binding_metric(bundle, thresholds, spec, applied_level))
        is not None
    )
