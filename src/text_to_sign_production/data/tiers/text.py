"""Text tier threshold parsing and evaluation."""

from __future__ import annotations

from text_to_sign_production.data.metrics.types import MetricBundle
from text_to_sign_production.data.tiers._shared.parsing import (
    require_exact_keys,
    require_mapping,
    require_nonnegative_int,
)
from text_to_sign_production.data.tiers.roles import (
    BINDING_TIER_METRICS_BY_FAMILY,
    evaluate_binding_metric,
)
from text_to_sign_production.data.tiers.types import (
    BindingTierFamily,
    FilterLevel,
    TextThresholds,
    TierMetricFailure,
)

_BINDING_SPECS = BINDING_TIER_METRICS_BY_FAMILY[BindingTierFamily.TEXT]
_THRESHOLD_KEYS = tuple(spec.threshold_attr for spec in _BINDING_SPECS)
_LEVEL_KEYS = tuple(level.value for level in FilterLevel)


def parse_text_thresholds(payload: object) -> dict[FilterLevel, TextThresholds]:
    """Parse strict text thresholds for every filter level."""
    levels = require_mapping(payload, "text")
    require_exact_keys(levels, _LEVEL_KEYS, "text")

    parsed: dict[FilterLevel, TextThresholds] = {}
    for level in FilterLevel:
        level_payload = require_mapping(levels[level.value], f"text.{level.value}")
        require_exact_keys(level_payload, _THRESHOLD_KEYS, f"text.{level.value}")
        parsed[level] = TextThresholds(
            min_character_count=require_nonnegative_int(
                level_payload["min_character_count"],
                "text.min_character_count",
            ),
            min_token_count=require_nonnegative_int(
                level_payload["min_token_count"],
                "text.min_token_count",
            ),
        )
    return parsed


def evaluate_text_family(
    bundle: MetricBundle,
    thresholds: TextThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    """Evaluate text metrics against the applied threshold level."""
    return tuple(
        failure
        for spec in _BINDING_SPECS
        if (failure := evaluate_binding_metric(bundle, thresholds, spec, applied_level))
        is not None
    )
