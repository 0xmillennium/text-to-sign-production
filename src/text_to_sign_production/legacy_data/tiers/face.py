"""Face-availability tier threshold parsing and evaluation."""

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
    NonManualVisibilityThresholds,
    TierMetricFailure,
)

_BINDING_SPECS = BINDING_TIER_METRICS_BY_FAMILY[BindingTierFamily.NON_MANUAL_VISIBILITY]
_THRESHOLD_KEYS = tuple(cast(str, spec.threshold_attr) for spec in _BINDING_SPECS)
_LEVEL_KEYS = tuple(level.value for level in FilterLevel)


def parse_face_thresholds(payload: object) -> dict[FilterLevel, NonManualVisibilityThresholds]:
    """Parse strict non-manual visibility thresholds for every filter level."""
    levels = require_mapping(payload, "non_manual_visibility")
    require_exact_keys(levels, _LEVEL_KEYS, "non_manual_visibility")

    parsed: dict[FilterLevel, NonManualVisibilityThresholds] = {}
    for level in FilterLevel:
        level_payload = require_mapping(
            levels[level.value],
            f"non_manual_visibility.{level.value}",
        )
        require_exact_keys(level_payload, _THRESHOLD_KEYS, f"non_manual_visibility.{level.value}")
        parsed[level] = NonManualVisibilityThresholds(
            min_active_span_face_available_frame_ratio=require_ratio(
                level_payload["min_active_span_face_available_frame_ratio"],
                "non_manual_visibility.min_active_span_face_available_frame_ratio",
            ),
            max_active_span_face_unavailable_run_ratio=require_ratio(
                level_payload["max_active_span_face_unavailable_run_ratio"],
                "non_manual_visibility.max_active_span_face_unavailable_run_ratio",
            ),
        )
    return parsed


def evaluate_face_family(
    bundle: MetricBundle,
    thresholds: NonManualVisibilityThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    """Evaluate active-span face-availability metrics against the applied level."""
    return tuple(
        failure
        for spec in _BINDING_SPECS
        if (failure := evaluate_binding_metric(bundle, thresholds, spec, applied_level)) is not None
    )
