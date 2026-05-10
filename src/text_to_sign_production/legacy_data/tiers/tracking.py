"""Tracking-quality tier threshold parsing and evaluation."""

from __future__ import annotations

from typing import cast

from text_to_sign_production.legacy_data.metrics.types import MetricBundle
from text_to_sign_production.legacy_data.tiers._shared.parsing import (
    require_exact_keys,
    require_mapping,
    require_ratio,
)
from text_to_sign_production.legacy_data.tiers.roles import (
    BINDING_TIER_METRICS_BY_FAMILY,
    evaluate_binding_metric,
)
from text_to_sign_production.legacy_data.tiers.types import (
    BindingTierFamily,
    FilterLevel,
    TierMetricFailure,
    TrackingQualityThresholds,
)

_BINDING_SPECS = BINDING_TIER_METRICS_BY_FAMILY[BindingTierFamily.TRACKING_QUALITY]
_THRESHOLD_KEYS = tuple(cast(str, spec.threshold_attr) for spec in _BINDING_SPECS)
_LEVEL_KEYS = tuple(level.value for level in FilterLevel)


def parse_tracking_quality_thresholds(
    payload: object,
) -> dict[FilterLevel, TrackingQualityThresholds]:
    """Parse strict tracking-quality thresholds for every filter level."""
    levels = require_mapping(payload, "tracking_quality")
    require_exact_keys(levels, _LEVEL_KEYS, "tracking_quality")

    parsed: dict[FilterLevel, TrackingQualityThresholds] = {}
    for level in FilterLevel:
        level_payload = require_mapping(levels[level.value], f"tracking_quality.{level.value}")
        require_exact_keys(level_payload, _THRESHOLD_KEYS, f"tracking_quality.{level.value}")
        parsed[level] = TrackingQualityThresholds(
            max_tracked_target_missing_frame_ratio=require_ratio(
                level_payload["max_tracked_target_missing_frame_ratio"],
                "tracking_quality.max_tracked_target_missing_frame_ratio",
            ),
            max_person_tracking_continuity_break_ratio=require_ratio(
                level_payload["max_person_tracking_continuity_break_ratio"],
                "tracking_quality.max_person_tracking_continuity_break_ratio",
            ),
            max_person_tracking_reanchor_ratio=require_ratio(
                level_payload["max_person_tracking_reanchor_ratio"],
                "tracking_quality.max_person_tracking_reanchor_ratio",
            ),
        )
    return parsed


def evaluate_tracking_quality_family(
    bundle: MetricBundle,
    thresholds: TrackingQualityThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    """Evaluate tracking-quality metrics against the applied threshold level."""
    return tuple(
        failure
        for spec in _BINDING_SPECS
        if (failure := evaluate_binding_metric(bundle, thresholds, spec, applied_level)) is not None
    )
