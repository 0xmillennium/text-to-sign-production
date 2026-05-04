"""Confidence tier threshold parsing and evaluation."""

from __future__ import annotations

from text_to_sign_production.data.metrics.types import MetricBundle
from text_to_sign_production.data.tiers._shared.parsing import (
    require_exact_keys,
    require_mapping,
    require_ratio,
)
from text_to_sign_production.data.tiers.types import (
    BindingTierFamily,
    ConfidenceThresholds,
    FilterLevel,
    TierMetricFailure,
)

_THRESHOLD_KEYS = (
    "min_body_mean_confidence",
    "min_left_hand_mean_confidence",
    "min_right_hand_mean_confidence",
    "min_face_mean_confidence",
)
_LEVEL_KEYS = tuple(level.value for level in FilterLevel)


def parse_confidence_thresholds(payload: object) -> dict[FilterLevel, ConfidenceThresholds]:
    """Parse strict confidence thresholds for every filter level."""
    levels = require_mapping(payload, "confidence")
    require_exact_keys(levels, _LEVEL_KEYS, "confidence")

    parsed: dict[FilterLevel, ConfidenceThresholds] = {}
    for level in FilterLevel:
        level_payload = require_mapping(levels[level.value], f"confidence.{level.value}")
        require_exact_keys(level_payload, _THRESHOLD_KEYS, f"confidence.{level.value}")
        parsed[level] = ConfidenceThresholds(
            min_body_mean_confidence=require_ratio(
                level_payload["min_body_mean_confidence"],
                "confidence.min_body_mean_confidence",
            ),
            min_left_hand_mean_confidence=require_ratio(
                level_payload["min_left_hand_mean_confidence"],
                "confidence.min_left_hand_mean_confidence",
            ),
            min_right_hand_mean_confidence=require_ratio(
                level_payload["min_right_hand_mean_confidence"],
                "confidence.min_right_hand_mean_confidence",
            ),
            min_face_mean_confidence=require_ratio(
                level_payload["min_face_mean_confidence"],
                "confidence.min_face_mean_confidence",
            ),
        )
    return parsed


def evaluate_confidence_family(
    bundle: MetricBundle,
    thresholds: ConfidenceThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    """Evaluate confidence metrics against the applied threshold level."""
    failures: list[TierMetricFailure] = []
    for metric_key, actual_value, expected_value in (
        (
            "body_mean_confidence",
            bundle.confidence.body_mean_confidence,
            thresholds.min_body_mean_confidence,
        ),
        (
            "left_hand_mean_confidence",
            bundle.confidence.left_hand_mean_confidence,
            thresholds.min_left_hand_mean_confidence,
        ),
        (
            "right_hand_mean_confidence",
            bundle.confidence.right_hand_mean_confidence,
            thresholds.min_right_hand_mean_confidence,
        ),
        (
            "face_mean_confidence",
            bundle.confidence.face_mean_confidence,
            thresholds.min_face_mean_confidence,
        ),
    ):
        if actual_value < expected_value:
            failures.append(
                TierMetricFailure(
                    family=BindingTierFamily.CONFIDENCE,
                    metric_key=metric_key,
                    reason_code=f"min_{metric_key}_not_met",
                    actual_value=actual_value,
                    expected_value=expected_value,
                    comparison=">=",
                    applied_level=applied_level,
                )
            )
    return tuple(failures)
