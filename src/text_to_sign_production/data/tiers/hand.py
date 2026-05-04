"""Hand tier threshold parsing and evaluation."""

from __future__ import annotations

from text_to_sign_production.data.metrics.types import MetricBundle
from text_to_sign_production.data.tiers._shared.parsing import (
    require_exact_keys,
    require_mapping,
    require_ratio,
)
from text_to_sign_production.data.tiers.types import (
    BindingTierFamily,
    FilterLevel,
    HandThresholds,
    TierMetricFailure,
)

_THRESHOLD_KEYS = (
    "min_any_hand_available_frame_ratio",
    "max_any_hand_unavailable_run_ratio",
)
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
            min_any_hand_available_frame_ratio=require_ratio(
                level_payload["min_any_hand_available_frame_ratio"],
                "hand.min_any_hand_available_frame_ratio",
            ),
            max_any_hand_unavailable_run_ratio=require_ratio(
                level_payload["max_any_hand_unavailable_run_ratio"],
                "hand.max_any_hand_unavailable_run_ratio",
            ),
        )
    return parsed


def evaluate_hand_family(
    bundle: MetricBundle,
    thresholds: HandThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    """Evaluate hand metrics against the applied threshold level."""
    failures: list[TierMetricFailure] = []
    if bundle.hand.any_hand_available_frame_ratio < thresholds.min_any_hand_available_frame_ratio:
        failures.append(
            TierMetricFailure(
                family=BindingTierFamily.HAND,
                metric_key="any_hand_available_frame_ratio",
                reason_code="min_any_hand_available_frame_ratio_not_met",
                actual_value=bundle.hand.any_hand_available_frame_ratio,
                expected_value=thresholds.min_any_hand_available_frame_ratio,
                comparison=">=",
                applied_level=applied_level,
            )
        )
    if (
        bundle.hand.max_any_hand_unavailable_run_ratio
        > thresholds.max_any_hand_unavailable_run_ratio
    ):
        failures.append(
            TierMetricFailure(
                family=BindingTierFamily.HAND,
                metric_key="max_any_hand_unavailable_run_ratio",
                reason_code="max_any_hand_unavailable_run_ratio_exceeded",
                actual_value=bundle.hand.max_any_hand_unavailable_run_ratio,
                expected_value=thresholds.max_any_hand_unavailable_run_ratio,
                comparison="<=",
                applied_level=applied_level,
            )
        )
    return tuple(failures)
