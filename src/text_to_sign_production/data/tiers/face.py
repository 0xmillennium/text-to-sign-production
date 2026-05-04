"""Face tier threshold parsing and evaluation."""

from __future__ import annotations

from text_to_sign_production.data.metrics.types import MetricBundle
from text_to_sign_production.data.tiers._shared.parsing import (
    require_exact_keys,
    require_mapping,
    require_ratio,
)
from text_to_sign_production.data.tiers.types import (
    BindingTierFamily,
    FaceThresholds,
    FilterLevel,
    TierMetricFailure,
)

_THRESHOLD_KEYS = ("min_face_available_frame_ratio",)
_LEVEL_KEYS = tuple(level.value for level in FilterLevel)


def parse_face_thresholds(payload: object) -> dict[FilterLevel, FaceThresholds]:
    """Parse strict face thresholds for every filter level."""
    levels = require_mapping(payload, "face")
    require_exact_keys(levels, _LEVEL_KEYS, "face")

    parsed: dict[FilterLevel, FaceThresholds] = {}
    for level in FilterLevel:
        level_payload = require_mapping(levels[level.value], f"face.{level.value}")
        require_exact_keys(level_payload, _THRESHOLD_KEYS, f"face.{level.value}")
        parsed[level] = FaceThresholds(
            min_face_available_frame_ratio=require_ratio(
                level_payload["min_face_available_frame_ratio"],
                "face.min_face_available_frame_ratio",
            ),
        )
    return parsed


def evaluate_face_family(
    bundle: MetricBundle,
    thresholds: FaceThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    """Evaluate face metrics against the applied threshold level."""
    if bundle.face.face_available_frame_ratio >= thresholds.min_face_available_frame_ratio:
        return ()
    return (
        TierMetricFailure(
            family=BindingTierFamily.FACE,
            metric_key="face_available_frame_ratio",
            reason_code="min_face_available_frame_ratio_not_met",
            actual_value=bundle.face.face_available_frame_ratio,
            expected_value=thresholds.min_face_available_frame_ratio,
            comparison=">=",
            applied_level=applied_level,
        ),
    )
