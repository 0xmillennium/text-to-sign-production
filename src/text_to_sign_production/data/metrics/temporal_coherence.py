"""Temporal coherence metrics for active signing motion."""

from __future__ import annotations

import math

import numpy as np

from text_to_sign_production.data.metrics.types import (
    ActiveSigningSpanMetrics,
    TemporalCoherenceMetrics,
)
from text_to_sign_production.data.metrics.upper_body_support import (
    UPPER_BODY_SUPPORT_LANDMARK_INDICES,
)
from text_to_sign_production.data.pose.schema import CANVAS_HEIGHT, CANVAS_WIDTH
from text_to_sign_production.data.samples.types import ProcessedSamplePayload

_ABRUPT_MOTION_DISTANCE_RATIO = 0.12
_DISCONTINUITY_DISTANCE_RATIO = 0.22
_FROZEN_MOTION_DISTANCE_RATIO = 0.001


def compute_temporal_coherence_metrics(
    payload: ProcessedSamplePayload,
    active_signing_span: ActiveSigningSpanMetrics,
) -> TemporalCoherenceMetrics:
    """Compute deterministic active-span motion pathology ratios."""
    start = active_signing_span.start_frame_index
    end = active_signing_span.end_frame_index_exclusive

    body_coords = np.asarray(payload.pose.body.coordinates)[
        start:end,
        UPPER_BODY_SUPPORT_LANDMARK_INDICES,
    ]
    body_conf = np.asarray(payload.pose.body.confidence)[
        start:end,
        UPPER_BODY_SUPPORT_LANDMARK_INDICES,
    ]
    left_coords = np.asarray(payload.pose.left_hand.coordinates)[start:end]
    left_conf = np.asarray(payload.pose.left_hand.confidence)[start:end]
    right_coords = np.asarray(payload.pose.right_hand.coordinates)[start:end]
    right_conf = np.asarray(payload.pose.right_hand.confidence)[start:end]

    points, available = _representative_articulator_points(
        body_coords,
        body_conf,
        left_coords,
        left_conf,
        right_coords,
        right_conf,
    )
    transition_count = max(0, active_signing_span.frame_count - 1)
    if transition_count == 0:
        return TemporalCoherenceMetrics(
            active_span_abrupt_motion_frame_ratio=0.0,
            active_span_discontinuity_frame_ratio=0.0,
            max_active_span_frozen_run_ratio=0.0,
        )

    distances = _transition_distances(points, available)
    abrupt_count = int(np.count_nonzero(distances > _ABRUPT_MOTION_DISTANCE_RATIO))
    discontinuity_count = int(
        np.count_nonzero(np.isnan(distances) | (distances > _DISCONTINUITY_DISTANCE_RATIO))
    )
    max_frozen_run = _longest_frozen_transition_run(distances)

    return TemporalCoherenceMetrics(
        active_span_abrupt_motion_frame_ratio=abrupt_count / transition_count,
        active_span_discontinuity_frame_ratio=discontinuity_count / transition_count,
        max_active_span_frozen_run_ratio=max_frozen_run / transition_count,
    )


def _representative_articulator_points(
    body_coords: np.ndarray,
    body_conf: np.ndarray,
    left_coords: np.ndarray,
    left_conf: np.ndarray,
    right_coords: np.ndarray,
    right_conf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    body_points, body_available, body_quality = _centroids_with_quality(body_coords, body_conf)
    left_points, left_available, left_quality = _centroids_with_quality(left_coords, left_conf)
    right_points, right_available, right_quality = _centroids_with_quality(right_coords, right_conf)

    points = np.zeros_like(body_points)
    available = np.zeros((body_points.shape[0],), dtype=np.bool_)
    for index in range(body_points.shape[0]):
        if left_available[index] or right_available[index]:
            if left_quality[index] >= right_quality[index]:
                points[index] = left_points[index]
                available[index] = left_available[index]
            else:
                points[index] = right_points[index]
                available[index] = right_available[index]
        elif body_available[index]:
            points[index] = body_points[index]
            available[index] = True
    return points, available


def _centroids_with_quality(
    coordinates: np.ndarray,
    confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if coordinates.ndim != 3 or confidence.ndim != 2:
        raise ValueError(
            f"Expected coordinates [frame, landmark, xy] and confidence [frame, landmark], "
            f"got {coordinates.shape} and {confidence.shape}."
        )
    points = np.zeros((confidence.shape[0], 2), dtype=float)
    available = np.any(confidence > 0.0, axis=1)
    quality = np.zeros((confidence.shape[0],), dtype=float)
    for index in np.flatnonzero(available):
        landmark_mask = confidence[index] > 0.0
        points[index] = np.mean(coordinates[index, landmark_mask], axis=0)
        quality[index] = float(np.mean(confidence[index, landmark_mask]))
    return points, available, quality


def _transition_distances(points: np.ndarray, available: np.ndarray) -> np.ndarray:
    distances = np.full((max(0, points.shape[0] - 1),), np.nan, dtype=float)
    normalizer = math.hypot(CANVAS_WIDTH, CANVAS_HEIGHT)
    for index in range(1, points.shape[0]):
        if not (available[index - 1] and available[index]):
            continue
        distances[index - 1] = float(np.linalg.norm(points[index] - points[index - 1]) / normalizer)
    return distances


def _longest_frozen_transition_run(distances: np.ndarray) -> int:
    longest = 0
    current = 0
    for distance in distances:
        if not np.isnan(distance) and distance <= _FROZEN_MOTION_DISTANCE_RATIO:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
