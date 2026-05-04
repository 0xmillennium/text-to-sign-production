"""Length tier threshold parsing and evaluation."""

from __future__ import annotations

from text_to_sign_production.data.metrics.types import MetricBundle
from text_to_sign_production.data.tiers._shared.parsing import (
    require_exact_keys,
    require_mapping,
    require_positive_float,
    require_positive_int,
)
from text_to_sign_production.data.tiers.types import (
    BindingTierFamily,
    FilterLevel,
    LengthThresholds,
    TierMetricFailure,
)

_THRESHOLD_KEYS = ("min_num_frames", "min_duration_seconds")
_LEVEL_KEYS = tuple(level.value for level in FilterLevel)


def parse_length_thresholds(payload: object) -> dict[FilterLevel, LengthThresholds]:
    """Parse strict length thresholds for every filter level."""
    levels = require_mapping(payload, "length")
    require_exact_keys(levels, _LEVEL_KEYS, "length")

    parsed: dict[FilterLevel, LengthThresholds] = {}
    for level in FilterLevel:
        level_payload = require_mapping(levels[level.value], f"length.{level.value}")
        require_exact_keys(level_payload, _THRESHOLD_KEYS, f"length.{level.value}")
        parsed[level] = LengthThresholds(
            min_num_frames=require_positive_int(
                level_payload["min_num_frames"],
                "length.min_num_frames",
            ),
            min_duration_seconds=require_positive_float(
                level_payload["min_duration_seconds"],
                "length.min_duration_seconds",
            ),
        )
    return parsed


def evaluate_length_family(
    bundle: MetricBundle,
    thresholds: LengthThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    """Evaluate length metrics against the applied threshold level."""
    failures: list[TierMetricFailure] = []
    if bundle.length.num_frames < thresholds.min_num_frames:
        failures.append(
            TierMetricFailure(
                family=BindingTierFamily.LENGTH,
                metric_key="num_frames",
                reason_code="min_num_frames_not_met",
                actual_value=bundle.length.num_frames,
                expected_value=thresholds.min_num_frames,
                comparison=">=",
                applied_level=applied_level,
            )
        )
    if bundle.length.duration_seconds is None:
        failures.append(
            TierMetricFailure(
                family=BindingTierFamily.LENGTH,
                metric_key="duration_seconds",
                reason_code="duration_seconds_missing",
                actual_value=None,
                expected_value=thresholds.min_duration_seconds,
                comparison=">=",
                applied_level=applied_level,
            )
        )
    elif bundle.length.duration_seconds < thresholds.min_duration_seconds:
        failures.append(
            TierMetricFailure(
                family=BindingTierFamily.LENGTH,
                metric_key="duration_seconds",
                reason_code="min_duration_seconds_not_met",
                actual_value=bundle.length.duration_seconds,
                expected_value=thresholds.min_duration_seconds,
                comparison=">=",
                applied_level=applied_level,
            )
        )
    return tuple(failures)
