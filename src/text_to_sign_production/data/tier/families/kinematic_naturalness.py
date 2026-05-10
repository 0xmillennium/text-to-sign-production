"""Kinematic naturalness family metrics."""

from __future__ import annotations

import math

from text_to_sign_production.data.tier.context import TransitionKind
from text_to_sign_production.data.tier.families.types import (
    KinematicNaturalnessMetrics,
    QualityMetricBuildInput,
)

_ABRUPT_RELATIVE_DISTANCE_LIMIT = 0.12
_DISCONTINUITY_RELATIVE_DISTANCE_LIMIT = 0.24
_FROZEN_DISTANCE_LIMIT = 1e-3


def compute_kinematic_naturalness_metrics(
    input: QualityMetricBuildInput,
) -> KinematicNaturalnessMetrics:
    """Measure representative-articulator motion over comparable active transitions."""
    frames = input.quality_context.representative_articulator.frames
    active = input.quality_context.active_span.active_frame_mask
    comparable_count = 0
    abrupt_count = 0
    discontinuity_count = 0
    frozen_count = 0
    frozen_denominator_count = 0
    for index in range(1, len(frames)):
        current = frames[index]
        previous = frames[index - 1]
        if (
            current.transition_from_previous is not TransitionKind.COMPARABLE
            or current.point is None
            or previous.point is None
        ):
            continue
        comparable_count += 1
        distance = math.dist(previous.point, current.point)
        if distance > _ABRUPT_RELATIVE_DISTANCE_LIMIT:
            abrupt_count += 1
        if distance > _DISCONTINUITY_RELATIVE_DISTANCE_LIMIT:
            discontinuity_count += 1
        if active[index]:
            frozen_denominator_count += 1
            if distance <= _FROZEN_DISTANCE_LIMIT:
                frozen_count += 1
    return KinematicNaturalnessMetrics(
        comparable_transition_abrupt_ratio=_ratio(abrupt_count, comparable_count),
        comparable_transition_discontinuity_ratio=_ratio(discontinuity_count, comparable_count),
        active_span_frozen_frame_ratio=_ratio(frozen_count, frozen_denominator_count),
    )


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


__all__ = ["compute_kinematic_naturalness_metrics"]
