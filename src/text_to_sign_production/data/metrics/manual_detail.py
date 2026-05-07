"""Fine manual-articulation detail metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from text_to_sign_production.data.metrics.types import (
    ActiveSigningSpanMetrics,
    ManualDetailMetrics,
)
from text_to_sign_production.data.pose.anatomy import (
    HAND_DISTAL_CHAIN_INDICES,
    HAND_FINGERTIP_INDICES,
)
from text_to_sign_production.data.samples.types import ProcessedSamplePayload

HandSource = Literal["left_hand", "right_hand"]
_HAND_SOURCE_ORDER: tuple[HandSource, ...] = ("left_hand", "right_hand")
_SOURCE_STICKINESS_QUALITY_RATIO = 0.90


@dataclass(frozen=True, slots=True)
class RepresentativeHandSequence:
    """Stable active-span representative hand sequence."""

    coordinates: np.ndarray
    confidence: np.ndarray
    available: np.ndarray
    sources: tuple[HandSource | None, ...]


def compute_manual_detail_metrics(
    payload: ProcessedSamplePayload,
    active_signing_span: ActiveSigningSpanMetrics,
) -> ManualDetailMetrics:
    """Compute active-span representative-hand fine-detail coverage."""
    sequence = select_representative_hand_sequence(payload, active_signing_span)
    confidence = sequence.confidence

    landmark_coverage_ratio = _coverage_ratio(confidence)
    fingertip_coverage_ratio = _coverage_ratio(confidence[:, HAND_FINGERTIP_INDICES])
    distal_chain_coverage_ratio = _coverage_ratio(confidence[:, HAND_DISTAL_CHAIN_INDICES])
    detail_available = np.any(confidence[:, HAND_FINGERTIP_INDICES] > 0.0, axis=1) & np.any(
        confidence[:, HAND_DISTAL_CHAIN_INDICES] > 0.0, axis=1
    )
    max_dropout_run = _longest_false_run(detail_available)

    return ManualDetailMetrics(
        active_span_representative_hand_landmark_coverage_ratio=landmark_coverage_ratio,
        active_span_representative_hand_fingertip_coverage_ratio=fingertip_coverage_ratio,
        active_span_representative_hand_distal_chain_coverage_ratio=distal_chain_coverage_ratio,
        max_active_span_representative_hand_detail_dropout_run_ratio=(
            max_dropout_run / active_signing_span.frame_count
        ),
    )


def select_representative_hand_sequence(
    payload: ProcessedSamplePayload,
    active_signing_span: ActiveSigningSpanMetrics,
) -> RepresentativeHandSequence:
    """Resolve a stable representative hand across the active signing span."""
    start = active_signing_span.start_frame_index
    end = active_signing_span.end_frame_index_exclusive
    left_coords = np.asarray(payload.pose.left_hand.coordinates)[start:end]
    left_conf = np.asarray(payload.pose.left_hand.confidence)[start:end]
    right_coords = np.asarray(payload.pose.right_hand.coordinates)[start:end]
    right_conf = np.asarray(payload.pose.right_hand.confidence)[start:end]
    if left_conf.shape[0] != active_signing_span.frame_count or right_conf.shape[0] != (
        active_signing_span.frame_count
    ):
        raise ValueError("Active signing span frame count does not match hand arrays.")

    coords_by_source: dict[HandSource, np.ndarray] = {
        "left_hand": left_coords,
        "right_hand": right_coords,
    }
    conf_by_source: dict[HandSource, np.ndarray] = {
        "left_hand": left_conf,
        "right_hand": right_conf,
    }
    quality_by_source = {
        source: _frame_detail_quality(confidence) for source, confidence in conf_by_source.items()
    }

    selected_coords = np.zeros_like(left_coords, dtype=float)
    selected_conf = np.zeros_like(left_conf, dtype=float)
    selected_available = np.zeros((active_signing_span.frame_count,), dtype=np.bool_)
    sources: list[HandSource | None] = []
    previous_source: HandSource | None = None

    for index in range(active_signing_span.frame_count):
        source = _select_hand_source(quality_by_source, index, previous_source)
        sources.append(source)
        if source is None:
            previous_source = None
            continue
        selected_coords[index] = coords_by_source[source][index]
        selected_conf[index] = conf_by_source[source][index]
        selected_available[index] = True
        previous_source = source

    return RepresentativeHandSequence(
        coordinates=selected_coords,
        confidence=selected_conf,
        available=selected_available,
        sources=tuple(sources),
    )


def _select_hand_source(
    quality_by_source: dict[HandSource, np.ndarray],
    index: int,
    previous_source: HandSource | None,
) -> HandSource | None:
    available_sources = tuple(
        source for source in _HAND_SOURCE_ORDER if quality_by_source[source][index] > 0.0
    )
    if not available_sources:
        return None
    if len(available_sources) == 1:
        return available_sources[0]

    best_source = min(
        available_sources,
        key=lambda source: (
            -float(quality_by_source[source][index]),
            _HAND_SOURCE_ORDER.index(source),
        ),
    )
    if previous_source in available_sources:
        previous_quality = float(quality_by_source[previous_source][index])
        best_quality = float(quality_by_source[best_source][index])
        if previous_quality >= best_quality * _SOURCE_STICKINESS_QUALITY_RATIO:
            return previous_source
    return best_source


def _frame_detail_quality(confidence: np.ndarray) -> np.ndarray:
    positive = confidence > 0.0
    landmark_coverage = np.count_nonzero(positive, axis=1) / confidence.shape[1]
    fingertip_coverage = np.count_nonzero(positive[:, HAND_FINGERTIP_INDICES], axis=1) / len(
        HAND_FINGERTIP_INDICES
    )
    distal_chain_coverage = np.count_nonzero(positive[:, HAND_DISTAL_CHAIN_INDICES], axis=1) / len(
        HAND_DISTAL_CHAIN_INDICES
    )
    return np.asarray(
        (landmark_coverage + fingertip_coverage + distal_chain_coverage) / 3.0,
        dtype=float,
    )


def _coverage_ratio(confidence: np.ndarray) -> float:
    if confidence.size == 0:
        raise ValueError("Cannot compute hand detail coverage for empty confidence.")
    return float(np.count_nonzero(confidence > 0.0) / confidence.size)


def _longest_false_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest
