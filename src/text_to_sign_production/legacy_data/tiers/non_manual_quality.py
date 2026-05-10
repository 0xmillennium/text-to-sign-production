"""Non-manual-quality tier threshold parsing and evaluation."""

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
    NonManualQualityThresholds,
    TierMetricFailure,
)

_BINDING_SPECS = BINDING_TIER_METRICS_BY_FAMILY[BindingTierFamily.NON_MANUAL_QUALITY]
_THRESHOLD_KEYS = tuple(cast(str, spec.threshold_attr) for spec in _BINDING_SPECS)
_LEVEL_KEYS = tuple(level.value for level in FilterLevel)


def parse_non_manual_quality_thresholds(
    payload: object,
) -> dict[FilterLevel, NonManualQualityThresholds]:
    """Parse strict non-manual-quality thresholds for every filter level."""
    levels = require_mapping(payload, "non_manual_quality")
    require_exact_keys(levels, _LEVEL_KEYS, "non_manual_quality")

    parsed: dict[FilterLevel, NonManualQualityThresholds] = {}
    for level in FilterLevel:
        level_payload = require_mapping(levels[level.value], f"non_manual_quality.{level.value}")
        require_exact_keys(level_payload, _THRESHOLD_KEYS, f"non_manual_quality.{level.value}")
        parsed[level] = NonManualQualityThresholds(
            min_active_span_face_landmark_coverage_ratio=require_ratio(
                level_payload["min_active_span_face_landmark_coverage_ratio"],
                "non_manual_quality.min_active_span_face_landmark_coverage_ratio",
            ),
            min_active_span_upper_face_landmark_coverage_ratio=require_ratio(
                level_payload["min_active_span_upper_face_landmark_coverage_ratio"],
                "non_manual_quality.min_active_span_upper_face_landmark_coverage_ratio",
            ),
            min_active_span_lower_face_landmark_coverage_ratio=require_ratio(
                level_payload["min_active_span_lower_face_landmark_coverage_ratio"],
                "non_manual_quality.min_active_span_lower_face_landmark_coverage_ratio",
            ),
            max_active_span_face_detail_dropout_run_ratio=require_ratio(
                level_payload["max_active_span_face_detail_dropout_run_ratio"],
                "non_manual_quality.max_active_span_face_detail_dropout_run_ratio",
            ),
            min_active_span_manual_face_overlap_frame_ratio=require_ratio(
                level_payload["min_active_span_manual_face_overlap_frame_ratio"],
                "non_manual_quality.min_active_span_manual_face_overlap_frame_ratio",
            ),
        )
    return parsed


def evaluate_non_manual_quality_family(
    bundle: MetricBundle,
    thresholds: NonManualQualityThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    """Evaluate non-manual-quality metrics against the applied threshold level."""
    return tuple(
        failure
        for spec in _BINDING_SPECS
        if (failure := evaluate_binding_metric(bundle, thresholds, spec, applied_level)) is not None
    )
