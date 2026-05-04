"""Hand tier threshold parsing and evaluation."""

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
    HandThresholds,
    TierMetricFailure,
)

_BINDING_SPECS = BINDING_TIER_METRICS_BY_FAMILY[BindingTierFamily.HAND]
_THRESHOLD_KEYS = tuple(spec.threshold_attr for spec in _BINDING_SPECS)
_LEVEL_KEYS = tuple(level.value for level in FilterLevel)


def parse_hand_thresholds(payload: object) -> dict[FilterLevel, HandThresholds]:
    """Parse strict hand thresholds for every filter level."""
    levels = require_mapping(payload, "hand")
    require_exact_keys(levels, _LEVEL_KEYS, "hand")

    parsed: dict[FilterLevel, HandThresholds] = {}
    for level in FilterLevel:
        level_payload = require_mapping(levels[level.value], f"hand.{level.value}")
        require_exact_keys(level_payload, _THRESHOLD_KEYS, f"hand.{level.value}")
        parsed[level] = HandThresholds(
            min_active_window_any_hand_available_frame_ratio=require_ratio(
                level_payload["min_active_window_any_hand_available_frame_ratio"],
                "hand.min_active_window_any_hand_available_frame_ratio",
            ),
            max_active_window_any_hand_unavailable_run_ratio=require_ratio(
                level_payload["max_active_window_any_hand_unavailable_run_ratio"],
                "hand.max_active_window_any_hand_unavailable_run_ratio",
            ),
        )
    return parsed


def evaluate_hand_family(
    bundle: MetricBundle,
    thresholds: HandThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    """Evaluate hand metrics against the applied threshold level."""
    return tuple(
        failure
        for spec in _BINDING_SPECS
        if (failure := evaluate_binding_metric(bundle, thresholds, spec, applied_level))
        is not None
    )
