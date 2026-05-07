"""Temporal coherence tier threshold parsing and evaluation."""

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
    TemporalCoherenceThresholds,
    TierMetricFailure,
)

_BINDING_SPECS = BINDING_TIER_METRICS_BY_FAMILY[BindingTierFamily.TEMPORAL_COHERENCE]
_THRESHOLD_KEYS = tuple(spec.threshold_attr for spec in _BINDING_SPECS)
_LEVEL_KEYS = tuple(level.value for level in FilterLevel)


def parse_temporal_coherence_thresholds(
    payload: object,
) -> dict[FilterLevel, TemporalCoherenceThresholds]:
    """Parse strict temporal coherence thresholds for every filter level."""
    levels = require_mapping(payload, "temporal_coherence")
    require_exact_keys(levels, _LEVEL_KEYS, "temporal_coherence")

    parsed: dict[FilterLevel, TemporalCoherenceThresholds] = {}
    for level in FilterLevel:
        level_payload = require_mapping(
            levels[level.value],
            f"temporal_coherence.{level.value}",
        )
        require_exact_keys(
            level_payload,
            _THRESHOLD_KEYS,
            f"temporal_coherence.{level.value}",
        )
        parsed[level] = TemporalCoherenceThresholds(
            max_active_span_abrupt_motion_frame_ratio=require_ratio(
                level_payload["max_active_span_abrupt_motion_frame_ratio"],
                "temporal_coherence.max_active_span_abrupt_motion_frame_ratio",
            ),
            max_active_span_discontinuity_frame_ratio=require_ratio(
                level_payload["max_active_span_discontinuity_frame_ratio"],
                "temporal_coherence.max_active_span_discontinuity_frame_ratio",
            ),
            max_active_span_frozen_run_ratio=require_ratio(
                level_payload["max_active_span_frozen_run_ratio"],
                "temporal_coherence.max_active_span_frozen_run_ratio",
            ),
        )
    return parsed


def evaluate_temporal_coherence_family(
    bundle: MetricBundle,
    thresholds: TemporalCoherenceThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    """Evaluate temporal coherence metrics against the applied threshold level."""
    return tuple(
        failure
        for spec in _BINDING_SPECS
        if (failure := evaluate_binding_metric(bundle, thresholds, spec, applied_level)) is not None
    )
