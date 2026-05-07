"""Manual-detail tier threshold parsing and evaluation."""

from __future__ import annotations

from typing import cast

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
    ManualDetailThresholds,
    TierMetricFailure,
)

_BINDING_SPECS = BINDING_TIER_METRICS_BY_FAMILY[BindingTierFamily.MANUAL_DETAIL]
_THRESHOLD_KEYS = tuple(cast(str, spec.threshold_attr) for spec in _BINDING_SPECS)
_LEVEL_KEYS = tuple(level.value for level in FilterLevel)


def parse_manual_detail_thresholds(payload: object) -> dict[FilterLevel, ManualDetailThresholds]:
    """Parse strict manual-detail thresholds for every filter level."""
    levels = require_mapping(payload, "manual_detail")
    require_exact_keys(levels, _LEVEL_KEYS, "manual_detail")

    parsed: dict[FilterLevel, ManualDetailThresholds] = {}
    for level in FilterLevel:
        level_payload = require_mapping(levels[level.value], f"manual_detail.{level.value}")
        require_exact_keys(level_payload, _THRESHOLD_KEYS, f"manual_detail.{level.value}")
        parsed[level] = ManualDetailThresholds(
            min_active_span_representative_hand_landmark_coverage_ratio=require_ratio(
                level_payload["min_active_span_representative_hand_landmark_coverage_ratio"],
                "manual_detail.min_active_span_representative_hand_landmark_coverage_ratio",
            ),
            min_active_span_representative_hand_fingertip_coverage_ratio=require_ratio(
                level_payload["min_active_span_representative_hand_fingertip_coverage_ratio"],
                "manual_detail.min_active_span_representative_hand_fingertip_coverage_ratio",
            ),
            min_active_span_representative_hand_distal_chain_coverage_ratio=require_ratio(
                level_payload["min_active_span_representative_hand_distal_chain_coverage_ratio"],
                "manual_detail.min_active_span_representative_hand_distal_chain_coverage_ratio",
            ),
            max_active_span_representative_hand_detail_dropout_run_ratio=require_ratio(
                level_payload["max_active_span_representative_hand_detail_dropout_run_ratio"],
                "manual_detail.max_active_span_representative_hand_detail_dropout_run_ratio",
            ),
        )
    return parsed


def evaluate_manual_detail_family(
    bundle: MetricBundle,
    thresholds: ManualDetailThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    """Evaluate manual-detail metrics against the applied threshold level."""
    return tuple(
        failure
        for spec in _BINDING_SPECS
        if (failure := evaluate_binding_metric(bundle, thresholds, spec, applied_level)) is not None
    )
