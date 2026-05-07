"""Temporal coherence metrics for active signing motion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from text_to_sign_production.data.metrics.support import (
    UPPER_BODY_SUPPORT_LANDMARK_INDICES,
)
from text_to_sign_production.data.metrics.types import (
    ActiveSigningSpanMetrics,
    KinematicNaturalnessMetrics,
)
from text_to_sign_production.data.samples.types import ProcessedSamplePayload

_ABRUPT_MOTION_DISTANCE_RATIO = 0.12
_DISCONTINUITY_DISTANCE_RATIO = 0.22
_FROZEN_MOTION_DISTANCE_RATIO = 0.001
_SOURCE_STICKINESS_QUALITY_RATIO = 0.85

ArticulatorSource = Literal["left_hand", "right_hand", "body"]
TransitionClassification = Literal["unavailable", "source_switch", "comparable"]
_SOURCE_FALLBACK_ORDER: tuple[ArticulatorSource, ...] = ("left_hand", "right_hand", "body")


@dataclass(frozen=True, slots=True)
class _ArticulatorCandidateSeries:
    source: ArticulatorSource
    points: np.ndarray
    available: np.ndarray
    quality: np.ndarray


@dataclass(frozen=True, slots=True)
class _RepresentativeArticulatorSequence:
    points: np.ndarray
    available: np.ndarray
    sources: tuple[ArticulatorSource | None, ...]


def compute_temporal_coherence_metrics(
    payload: ProcessedSamplePayload,
    active_signing_span: ActiveSigningSpanMetrics,
) -> KinematicNaturalnessMetrics:
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

    candidates = _frame_articulator_candidates(
        body_coords,
        body_conf,
        left_coords,
        left_conf,
        right_coords,
        right_conf,
    )
    representatives = _resolve_stable_representative_sequence(candidates)
    transition_count = max(0, active_signing_span.frame_count - 1)
    if transition_count == 0:
        return KinematicNaturalnessMetrics(
            active_span_abrupt_motion_frame_ratio=0.0,
            active_span_discontinuity_frame_ratio=0.0,
            max_active_span_frozen_run_ratio=0.0,
        )

    classifications = _classify_adjacent_transitions(representatives)
    distances = _comparable_transition_distances(representatives, classifications)
    abrupt_count = int(np.count_nonzero(distances > _ABRUPT_MOTION_DISTANCE_RATIO))
    discontinuity_count = int(np.count_nonzero(distances > _DISCONTINUITY_DISTANCE_RATIO))
    max_frozen_run = _longest_frozen_transition_run(distances, classifications)

    return KinematicNaturalnessMetrics(
        active_span_abrupt_motion_frame_ratio=abrupt_count / transition_count,
        active_span_discontinuity_frame_ratio=discontinuity_count / transition_count,
        max_active_span_frozen_run_ratio=max_frozen_run / transition_count,
    )


def _frame_articulator_candidates(
    body_coords: np.ndarray,
    body_conf: np.ndarray,
    left_coords: np.ndarray,
    left_conf: np.ndarray,
    right_coords: np.ndarray,
    right_conf: np.ndarray,
) -> tuple[_ArticulatorCandidateSeries, ...]:
    body_points, body_available, body_quality = _centroids_with_quality(body_coords, body_conf)
    left_points, left_available, left_quality = _centroids_with_quality(left_coords, left_conf)
    right_points, right_available, right_quality = _centroids_with_quality(right_coords, right_conf)

    return (
        _ArticulatorCandidateSeries(
            source="left_hand",
            points=left_points,
            available=left_available,
            quality=left_quality,
        ),
        _ArticulatorCandidateSeries(
            source="right_hand",
            points=right_points,
            available=right_available,
            quality=right_quality,
        ),
        _ArticulatorCandidateSeries(
            source="body",
            points=body_points,
            available=body_available,
            quality=body_quality,
        ),
    )


def _resolve_stable_representative_sequence(
    candidates: tuple[_ArticulatorCandidateSeries, ...],
) -> _RepresentativeArticulatorSequence:
    """Resolve a stable articulator source sequence over the active span.

    The resolved sequence avoids eager left/right/body switching. A still
    available previous source is retained unless another source is materially
    stronger; when the previous source disappears, fallback is deterministic.
    """
    candidates_by_source = {candidate.source: candidate for candidate in candidates}
    num_frames = candidates[0].points.shape[0] if candidates else 0
    points = np.zeros((num_frames, 2), dtype=float)
    available = np.zeros((num_frames,), dtype=np.bool_)
    sources: list[ArticulatorSource | None] = []
    previous_source: ArticulatorSource | None = None

    for index in range(num_frames):
        selected_source = _select_representative_source(
            candidates_by_source,
            index,
            previous_source,
        )
        sources.append(selected_source)
        if selected_source is None:
            previous_source = None
            continue

        selected_candidate = candidates_by_source[selected_source]
        points[index] = selected_candidate.points[index]
        available[index] = True
        previous_source = selected_source

    return _RepresentativeArticulatorSequence(
        points=points,
        available=available,
        sources=tuple(sources),
    )


def _select_representative_source(
    candidates_by_source: dict[ArticulatorSource, _ArticulatorCandidateSeries],
    index: int,
    previous_source: ArticulatorSource | None,
) -> ArticulatorSource | None:
    available_sources = tuple(
        source for source in _SOURCE_FALLBACK_ORDER if candidates_by_source[source].available[index]
    )
    if not available_sources:
        return None

    best_source = min(
        available_sources,
        key=lambda source: (
            -float(candidates_by_source[source].quality[index]),
            _SOURCE_FALLBACK_ORDER.index(source),
        ),
    )
    if previous_source in available_sources:
        previous_quality = float(candidates_by_source[previous_source].quality[index])
        best_quality = float(candidates_by_source[best_source].quality[index])
        if previous_quality >= best_quality * _SOURCE_STICKINESS_QUALITY_RATIO:
            return previous_source
    return best_source


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


def _classify_adjacent_transitions(
    sequence: _RepresentativeArticulatorSequence,
) -> tuple[TransitionClassification, ...]:
    classifications: list[TransitionClassification] = []
    for index in range(1, sequence.points.shape[0]):
        previous_source = sequence.sources[index - 1]
        current_source = sequence.sources[index]
        if not (
            sequence.available[index - 1]
            and sequence.available[index]
            and previous_source is not None
            and current_source is not None
        ):
            classifications.append("unavailable")
        elif previous_source != current_source:
            classifications.append("source_switch")
        else:
            classifications.append("comparable")
    return tuple(classifications)


def _comparable_transition_distances(
    sequence: _RepresentativeArticulatorSequence,
    classifications: tuple[TransitionClassification, ...],
) -> np.ndarray:
    distances = np.full((len(classifications),), np.nan, dtype=float)
    for index, classification in enumerate(classifications, start=1):
        if classification != "comparable":
            continue
        # Representative points are normalized parser coordinates; distances
        # are directly comparable to normalized motion thresholds.
        distances[index - 1] = float(
            np.linalg.norm(sequence.points[index] - sequence.points[index - 1])
        )
    return distances


def _longest_frozen_transition_run(
    distances: np.ndarray,
    classifications: tuple[TransitionClassification, ...],
) -> int:
    longest = 0
    current = 0
    for distance, classification in zip(distances, classifications, strict=True):
        if classification == "comparable" and distance <= _FROZEN_MOTION_DISTANCE_RATIO:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
