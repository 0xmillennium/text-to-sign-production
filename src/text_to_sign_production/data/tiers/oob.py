"""OOB tier threshold parsing and evaluation."""

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
    OobThresholds,
    TierMetricFailure,
)

_THRESHOLD_KEYS = ("max_out_of_bounds_ratio",)
_LEVEL_KEYS = tuple(level.value for level in FilterLevel)


def parse_oob_thresholds(payload: object) -> dict[FilterLevel, OobThresholds]:
    """Parse strict OOB thresholds for every filter level."""
    levels = require_mapping(payload, "oob")
    require_exact_keys(levels, _LEVEL_KEYS, "oob")

    parsed: dict[FilterLevel, OobThresholds] = {}
    for level in FilterLevel:
        level_payload = require_mapping(levels[level.value], f"oob.{level.value}")
        require_exact_keys(level_payload, _THRESHOLD_KEYS, f"oob.{level.value}")
        parsed[level] = OobThresholds(
            max_out_of_bounds_ratio=require_ratio(
                level_payload["max_out_of_bounds_ratio"],
                "oob.max_out_of_bounds_ratio",
            )
        )
    return parsed


def evaluate_oob_family(
    bundle: MetricBundle,
    thresholds: OobThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    """Evaluate OOB metrics against the applied threshold level."""
    if bundle.oob.out_of_bounds_ratio <= thresholds.max_out_of_bounds_ratio:
        return ()
    return (
        TierMetricFailure(
            family=BindingTierFamily.OOB,
            metric_key="out_of_bounds_ratio",
            reason_code="max_out_of_bounds_ratio_exceeded",
            actual_value=bundle.oob.out_of_bounds_ratio,
            expected_value=thresholds.max_out_of_bounds_ratio,
            comparison="<=",
            applied_level=applied_level,
        ),
    )
