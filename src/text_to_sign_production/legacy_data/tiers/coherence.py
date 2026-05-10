"""Temporal motion-pathology tier threshold parsing and evaluation."""

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
    KinematicNaturalnessThresholds,
    TierMetricFailure,
)

_BINDING_SPECS = BINDING_TIER_METRICS_BY_FAMILY[BindingTierFamily.KINEMATIC_NATURALNESS]
_THRESHOLD_KEYS = tuple(cast(str, spec.threshold_attr) for spec in _BINDING_SPECS)
_LEVEL_KEYS = tuple(level.value for level in FilterLevel)


def parse_temporal_coherence_thresholds(
    payload: object,
) -> dict[FilterLevel, KinematicNaturalnessThresholds]:
    """Parse strict kinematic-naturalness thresholds for every filter level."""
    levels = require_mapping(payload, "kinematic_naturalness")
    require_exact_keys(levels, _LEVEL_KEYS, "kinematic_naturalness")

    parsed: dict[FilterLevel, KinematicNaturalnessThresholds] = {}
    for level in FilterLevel:
        level_payload = require_mapping(
            levels[level.value],
            f"kinematic_naturalness.{level.value}",
        )
        require_exact_keys(
            level_payload,
            _THRESHOLD_KEYS,
            f"kinematic_naturalness.{level.value}",
        )
        parsed[level] = KinematicNaturalnessThresholds(
            max_active_span_abrupt_motion_frame_ratio=require_ratio(
                level_payload["max_active_span_abrupt_motion_frame_ratio"],
                "kinematic_naturalness.max_active_span_abrupt_motion_frame_ratio",
            ),
            max_active_span_discontinuity_frame_ratio=require_ratio(
                level_payload["max_active_span_discontinuity_frame_ratio"],
                "kinematic_naturalness.max_active_span_discontinuity_frame_ratio",
            ),
            max_active_span_frozen_run_ratio=require_ratio(
                level_payload["max_active_span_frozen_run_ratio"],
                "kinematic_naturalness.max_active_span_frozen_run_ratio",
            ),
        )
    return parsed


def evaluate_temporal_coherence_family(
    bundle: MetricBundle,
    thresholds: KinematicNaturalnessThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    """Evaluate temporal motion-pathology metrics against the applied threshold level."""
    return tuple(
        failure
        for spec in _BINDING_SPECS
        if (failure := evaluate_binding_metric(bundle, thresholds, spec, applied_level)) is not None
    )
