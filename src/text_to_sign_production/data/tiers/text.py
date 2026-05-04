"""Text tier threshold parsing and evaluation."""

from __future__ import annotations

from text_to_sign_production.data.metrics.types import MetricBundle
from text_to_sign_production.data.tiers._shared.parsing import (
    require_exact_keys,
    require_mapping,
    require_nonnegative_int,
)
from text_to_sign_production.data.tiers.types import (
    BindingTierFamily,
    FilterLevel,
    TextThresholds,
    TierMetricFailure,
)

_THRESHOLD_KEYS = ("min_character_count", "min_token_count")
_LEVEL_KEYS = tuple(level.value for level in FilterLevel)


def parse_text_thresholds(payload: object) -> dict[FilterLevel, TextThresholds]:
    """Parse strict text thresholds for every filter level."""
    levels = require_mapping(payload, "text")
    require_exact_keys(levels, _LEVEL_KEYS, "text")

    parsed: dict[FilterLevel, TextThresholds] = {}
    for level in FilterLevel:
        level_payload = require_mapping(levels[level.value], f"text.{level.value}")
        require_exact_keys(level_payload, _THRESHOLD_KEYS, f"text.{level.value}")
        parsed[level] = TextThresholds(
            min_character_count=require_nonnegative_int(
                level_payload["min_character_count"],
                "text.min_character_count",
            ),
            min_token_count=require_nonnegative_int(
                level_payload["min_token_count"],
                "text.min_token_count",
            ),
        )
    return parsed


def evaluate_text_family(
    bundle: MetricBundle,
    thresholds: TextThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    """Evaluate text metrics against the applied threshold level."""
    failures: list[TierMetricFailure] = []
    if bundle.text.character_count < thresholds.min_character_count:
        failures.append(
            TierMetricFailure(
                family=BindingTierFamily.TEXT,
                metric_key="character_count",
                reason_code="min_character_count_not_met",
                actual_value=bundle.text.character_count,
                expected_value=thresholds.min_character_count,
                comparison=">=",
                applied_level=applied_level,
            )
        )
    if bundle.text.token_count < thresholds.min_token_count:
        failures.append(
            TierMetricFailure(
                family=BindingTierFamily.TEXT,
                metric_key="token_count",
                reason_code="min_token_count_not_met",
                actual_value=bundle.text.token_count,
                expected_value=thresholds.min_token_count,
                comparison=">=",
                applied_level=applied_level,
            )
        )
    return tuple(failures)
