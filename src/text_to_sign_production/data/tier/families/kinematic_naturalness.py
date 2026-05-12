"""Kinematic naturalness family metrics."""

from __future__ import annotations

import math

import numpy as np

from text_to_sign_production.core.models import PreparedSample
from text_to_sign_production.data.tier.context import ArticulatorSource, TransitionKind
from text_to_sign_production.data.tier.context.span import UPPER_BODY_TRACKING_LANDMARK_INDICES
from text_to_sign_production.data.tier.context.types import RepresentativeArticulatorFrame
from text_to_sign_production.data.tier.families.types import (
    KinematicNaturalnessMetrics,
    QualityMetricBuildInput,
)

_ABRUPT_RELATIVE_DISTANCE_LIMIT = 0.12
_DISCONTINUITY_RELATIVE_DISTANCE_LIMIT = 0.24
_FROZEN_RELATIVE_DISTANCE_LIMIT = 1e-3
_SCALE_EPSILON = 1e-6


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
    frozen_mask = [False] * len(frames)
    for index in range(1, len(frames)):
        current = frames[index]
        previous = frames[index - 1]
        if current.transition_from_previous is not TransitionKind.COMPARABLE:
            continue
        distance = _normalized_transition_distance(input.sample, previous, current)
        if distance is None:
            continue
        comparable_count += 1
        if distance > _ABRUPT_RELATIVE_DISTANCE_LIMIT:
            abrupt_count += 1
        if distance > _DISCONTINUITY_RELATIVE_DISTANCE_LIMIT:
            discontinuity_count += 1
        if active[index]:
            if distance <= _FROZEN_RELATIVE_DISTANCE_LIMIT:
                frozen_count += 1
                frozen_mask[index] = True
    active_count = sum(1 for value in active if value)
    return KinematicNaturalnessMetrics(
        comparable_transition_abrupt_ratio=_ratio(abrupt_count, comparable_count),
        comparable_transition_discontinuity_ratio=_ratio(discontinuity_count, comparable_count),
        active_span_frozen_frame_ratio=_ratio(frozen_count, active_count),
        max_active_span_frozen_run_ratio=_ratio(_max_run(tuple(frozen_mask)), active_count),
    )


def _source_points(
    sample: PreparedSample,
    source: ArticulatorSource,
    frame_index: int,
) -> np.ndarray:
    if source == ArticulatorSource.LEFT_HAND:
        xyc = sample.pose.left_hand_xyc[frame_index]
    elif source == ArticulatorSource.RIGHT_HAND:
        xyc = sample.pose.right_hand_xyc[frame_index]
    else:
        xyc = sample.pose.body_xyc[frame_index, list(UPPER_BODY_TRACKING_LANDMARK_INDICES), :]
    observed = xyc[:, 2] > 0.0
    return np.asarray(xyc[observed, :2], dtype=float)


def _source_scale(
    sample: PreparedSample,
    source: ArticulatorSource,
    frame_index: int,
) -> float | None:
    points = _source_points(sample, source, frame_index)
    if points.shape[0] < 2:
        return None
    minima = np.min(points, axis=0)
    maxima = np.max(points, axis=0)
    scale = float(np.linalg.norm(maxima - minima))
    if not math.isfinite(scale) or scale <= _SCALE_EPSILON:
        return None
    return scale


def _normalized_transition_distance(
    sample: PreparedSample,
    previous_frame: RepresentativeArticulatorFrame,
    current_frame: RepresentativeArticulatorFrame,
) -> float | None:
    if (
        previous_frame.source is None
        or current_frame.source is None
        or previous_frame.source != current_frame.source
        or previous_frame.point is None
        or current_frame.point is None
    ):
        return None
    previous_scale = _source_scale(sample, previous_frame.source, previous_frame.frame_index)
    current_scale = _source_scale(sample, current_frame.source, current_frame.frame_index)
    if previous_scale is None or current_scale is None:
        return None
    scale = max((previous_scale + current_scale) / 2.0, _SCALE_EPSILON)
    return math.dist(previous_frame.point, current_frame.point) / scale


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _max_run(mask: tuple[bool, ...]) -> int:
    longest = 0
    current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


__all__ = ["compute_kinematic_naturalness_metrics"]
