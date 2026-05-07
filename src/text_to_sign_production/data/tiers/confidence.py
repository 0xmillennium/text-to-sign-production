"""Confidence tier threshold parsing and evaluation."""

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
    ConfidenceThresholds,
    FilterLevel,
    TierMetricFailure,
)

_BINDING_SPECS = BINDING_TIER_METRICS_BY_FAMILY[BindingTierFamily.CONFIDENCE]
_THRESHOLD_KEYS = tuple(spec.threshold_attr for spec in _BINDING_SPECS)
_LEVEL_KEYS = tuple(level.value for level in FilterLevel)


def parse_confidence_thresholds(payload: object) -> dict[FilterLevel, ConfidenceThresholds]:
    """Parse strict confidence thresholds for every filter level."""
    levels = require_mapping(payload, "confidence")
    require_exact_keys(levels, _LEVEL_KEYS, "confidence")

    parsed: dict[FilterLevel, ConfidenceThresholds] = {}
    for level in FilterLevel:
        level_payload = require_mapping(levels[level.value], f"confidence.{level.value}")
        require_exact_keys(level_payload, _THRESHOLD_KEYS, f"confidence.{level.value}")
        parsed[level] = ConfidenceThresholds(
            min_body_available_mean_confidence=require_ratio(
                level_payload["min_body_available_mean_confidence"],
                "confidence.min_body_available_mean_confidence",
            ),
            min_active_span_any_hand_available_mean_confidence=require_ratio(
                level_payload["min_active_span_any_hand_available_mean_confidence"],
                "confidence.min_active_span_any_hand_available_mean_confidence",
            ),
        )
    return parsed


def evaluate_confidence_family(
    bundle: MetricBundle,
    thresholds: ConfidenceThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    """Evaluate confidence metrics against the applied threshold level."""
    return tuple(
        failure
        for spec in _BINDING_SPECS
        if (failure := evaluate_binding_metric(bundle, thresholds, spec, applied_level)) is not None
    )
