"""Coverage tier threshold parsing and evaluation."""

from __future__ import annotations

from text_to_sign_production.data.metrics.types import MetricBundle
from text_to_sign_production.data.tiers._shared.parsing import (
    require_exact_keys,
    require_mapping,
    require_ratio,
)
from text_to_sign_production.data.tiers.types import (
    BindingTierFamily,
    CoverageThresholds,
    FilterLevel,
    TierMetricFailure,
)

_THRESHOLD_KEYS = (
    "min_body_landmark_coverage_ratio",
    "min_any_hand_landmark_coverage_ratio",
    "min_face_landmark_coverage_ratio",
)
_LEVEL_KEYS = tuple(level.value for level in FilterLevel)


def parse_coverage_thresholds(payload: object) -> dict[FilterLevel, CoverageThresholds]:
    """Parse strict coverage thresholds for every filter level."""
    levels = require_mapping(payload, "coverage")
    require_exact_keys(levels, _LEVEL_KEYS, "coverage")

    parsed: dict[FilterLevel, CoverageThresholds] = {}
    for level in FilterLevel:
        level_payload = require_mapping(levels[level.value], f"coverage.{level.value}")
        require_exact_keys(level_payload, _THRESHOLD_KEYS, f"coverage.{level.value}")
        parsed[level] = CoverageThresholds(
            min_body_landmark_coverage_ratio=require_ratio(
                level_payload["min_body_landmark_coverage_ratio"],
                "coverage.min_body_landmark_coverage_ratio",
            ),
            min_any_hand_landmark_coverage_ratio=require_ratio(
                level_payload["min_any_hand_landmark_coverage_ratio"],
                "coverage.min_any_hand_landmark_coverage_ratio",
            ),
            min_face_landmark_coverage_ratio=require_ratio(
                level_payload["min_face_landmark_coverage_ratio"],
                "coverage.min_face_landmark_coverage_ratio",
            ),
        )
    return parsed


def evaluate_coverage_family(
    bundle: MetricBundle,
    thresholds: CoverageThresholds,
    applied_level: FilterLevel,
) -> tuple[TierMetricFailure, ...]:
    """Evaluate coverage metrics against the applied threshold level."""
    failures: list[TierMetricFailure] = []
    _append_min_failure(
        failures,
        metric_key="body_landmark_coverage_ratio",
        reason_code="min_body_landmark_coverage_ratio_not_met",
        actual_value=bundle.coverage.body_landmark_coverage_ratio,
        expected_value=thresholds.min_body_landmark_coverage_ratio,
        applied_level=applied_level,
    )
    _append_min_failure(
        failures,
        metric_key="any_hand_landmark_coverage_ratio",
        reason_code="min_any_hand_landmark_coverage_ratio_not_met",
        actual_value=bundle.coverage.any_hand_landmark_coverage_ratio,
        expected_value=thresholds.min_any_hand_landmark_coverage_ratio,
        applied_level=applied_level,
    )
    _append_min_failure(
        failures,
        metric_key="face_landmark_coverage_ratio",
        reason_code="min_face_landmark_coverage_ratio_not_met",
        actual_value=bundle.coverage.face_landmark_coverage_ratio,
        expected_value=thresholds.min_face_landmark_coverage_ratio,
        applied_level=applied_level,
    )
    return tuple(failures)


def _append_min_failure(
    failures: list[TierMetricFailure],
    *,
    metric_key: str,
    reason_code: str,
    actual_value: float,
    expected_value: float,
    applied_level: FilterLevel,
) -> None:
    if actual_value >= expected_value:
        return
    failures.append(
        TierMetricFailure(
            family=BindingTierFamily.COVERAGE,
            metric_key=metric_key,
            reason_code=reason_code,
            actual_value=actual_value,
            expected_value=expected_value,
            comparison=">=",
            applied_level=applied_level,
        )
    )
