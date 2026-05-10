"""Geometry tier threshold parsing and evaluation."""

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
    GeometryThresholds,
    TierMetricFailure,
)

_BINDING_SPECS = BINDING_TIER_METRICS_BY_FAMILY[BindingTierFamily.GEOMETRY]
_THRESHOLD_KEYS = tuple(cast(str, spec.threshold_attr) for spec in _BINDING_SPECS)
_LEVEL_KEYS = tuple(level.value for level in FilterLevel)


def parse_geometry_thresholds(payload: object) -> dict[FilterLevel, GeometryThresholds]:
    """Parse strict geometry thresholds for every filter level."""
    levels = require_mapping(payload, "geometry")
    require_exact_keys(levels, _LEVEL_KEYS, "geometry")

    parsed: dict[FilterLevel, GeometryThresholds] = {}
    for level in FilterLevel:
        level_payload = require_mapping(levels[level.value], f"geometry.{level.value}")
        require_exact_keys(level_payload, _THRESHOLD_KEYS, f"geometry.{level.value}")
        parsed[level] = GeometryThresholds(
            max_active_span_upper_body_bone_length_outlier_frame_ratio=require_ratio(
                level_payload["max_active_span_upper_body_bone_length_outlier_frame_ratio"],
                "geometry.max_active_span_upper_body_bone_length_outlier_frame_ratio",
            ),
            max_active_span_representative_hand_bone_length_outlier_frame_ratio=require_ratio(
                level_payload[
                    "max_active_span_representative_hand_bone_length_outlier_frame_ratio"
                ],
                "geometry.max_active_span_representative_hand_bone_length_outlier_frame_ratio",
            ),
            max_active_span_cross_channel_scale_outlier_frame_ratio=require_ratio(
                level_payload["max_active_span_cross_channel_scale_outlier_frame_ratio"],
                "geometry.max_active_span_cross_channel_scale_outlier_frame_ratio",
            ),
        )
    return parsed


def evaluate_geometry_family(
    bundle: MetricBundle,
    thresholds: GeometryThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    """Evaluate geometry metrics against the applied threshold level."""
    return tuple(
        failure
        for spec in _BINDING_SPECS
        if (failure := evaluate_binding_metric(bundle, thresholds, spec, applied_level)) is not None
    )
