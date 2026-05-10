"""Sample-internal geometric consistency metrics."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.legacy_data.metrics.manual_detail import (
    select_representative_hand_sequence,
)
from text_to_sign_production.legacy_data.metrics.types import (
    ActiveSigningSpanMetrics,
    GeometryMetrics,
)
from text_to_sign_production.legacy_data.pose.anatomy import (
    CROSS_CHANNEL_BODY_SCALE_SEGMENTS,
    CROSS_CHANNEL_HAND_SCALE_SEGMENTS,
    HAND_BONE_SEGMENTS,
    UPPER_BODY_BONE_SEGMENTS,
)
from text_to_sign_production.legacy_data.samples.types import ProcessedSamplePayload

_SEGMENT_LENGTH_OUTLIER_RELATIVE_DEVIATION = 0.55
_CROSS_CHANNEL_SCALE_OUTLIER_RELATIVE_DEVIATION = 0.75
_MIN_REFERENCE_LENGTH = 1e-6


def compute_geometry_metrics(
    payload: ProcessedSamplePayload,
    active_signing_span: ActiveSigningSpanMetrics,
) -> GeometryMetrics:
    """Compute active-span sample-internal 2D geometric consistency metrics."""
    start = active_signing_span.start_frame_index
    end = active_signing_span.end_frame_index_exclusive
    body_coords = np.asarray(payload.pose.body.coordinates)[start:end]
    body_conf = np.asarray(payload.pose.body.confidence)[start:end]
    if body_conf.shape[0] != active_signing_span.frame_count:
        raise ValueError("Active signing span frame count does not match body arrays.")

    representative_hand = select_representative_hand_sequence(payload, active_signing_span)

    return GeometryMetrics(
        active_span_upper_body_bone_length_outlier_frame_ratio=_segment_outlier_frame_ratio(
            body_coords,
            body_conf,
            UPPER_BODY_BONE_SEGMENTS,
            active_signing_span.frame_count,
        ),
        active_span_representative_hand_bone_length_outlier_frame_ratio=(
            _segment_outlier_frame_ratio(
                representative_hand.coordinates,
                representative_hand.confidence,
                HAND_BONE_SEGMENTS,
                active_signing_span.frame_count,
            )
        ),
        active_span_cross_channel_scale_outlier_frame_ratio=_cross_channel_scale_outlier_ratio(
            body_coords,
            body_conf,
            representative_hand.coordinates,
            representative_hand.confidence,
            active_signing_span.frame_count,
        ),
    )


def _segment_outlier_frame_ratio(
    coordinates: np.ndarray,
    confidence: np.ndarray,
    segments: tuple[tuple[int, int], ...],
    frame_count: int,
) -> float:
    lengths = _segment_lengths(coordinates, confidence, segments)
    reference = _robust_reference_by_segment(lengths)
    if not np.any(np.isfinite(reference)):
        return 0.0

    outlier_frames = np.zeros((frame_count,), dtype=np.bool_)
    for segment_index, reference_length in enumerate(reference):
        if not np.isfinite(reference_length) or reference_length <= _MIN_REFERENCE_LENGTH:
            continue
        segment_lengths = lengths[:, segment_index]
        comparable = np.isfinite(segment_lengths)
        relative_deviation = np.zeros((frame_count,), dtype=float)
        relative_deviation[comparable] = (
            np.abs(segment_lengths[comparable] - reference_length) / reference_length
        )
        outlier_frames |= comparable & (
            relative_deviation > _SEGMENT_LENGTH_OUTLIER_RELATIVE_DEVIATION
        )
    return float(np.count_nonzero(outlier_frames) / frame_count)


def _cross_channel_scale_outlier_ratio(
    body_coords: np.ndarray,
    body_conf: np.ndarray,
    hand_coords: np.ndarray,
    hand_conf: np.ndarray,
    frame_count: int,
) -> float:
    body_scale = _frame_median_segment_length(
        body_coords,
        body_conf,
        CROSS_CHANNEL_BODY_SCALE_SEGMENTS,
    )
    hand_scale = _frame_median_segment_length(
        hand_coords,
        hand_conf,
        CROSS_CHANNEL_HAND_SCALE_SEGMENTS,
    )
    comparable = (body_scale > _MIN_REFERENCE_LENGTH) & (hand_scale > _MIN_REFERENCE_LENGTH)
    if not np.any(comparable):
        return 0.0

    scale_ratio = np.full((frame_count,), np.nan, dtype=float)
    scale_ratio[comparable] = hand_scale[comparable] / body_scale[comparable]
    reference = float(np.nanmedian(scale_ratio))
    if not np.isfinite(reference) or reference <= _MIN_REFERENCE_LENGTH:
        return 0.0

    relative_deviation = np.zeros((frame_count,), dtype=float)
    relative_deviation[comparable] = np.abs(scale_ratio[comparable] - reference) / reference
    outlier_frames = comparable & (
        relative_deviation > _CROSS_CHANNEL_SCALE_OUTLIER_RELATIVE_DEVIATION
    )
    return float(np.count_nonzero(outlier_frames) / frame_count)


def _segment_lengths(
    coordinates: np.ndarray,
    confidence: np.ndarray,
    segments: tuple[tuple[int, int], ...],
) -> np.ndarray:
    lengths = np.full((confidence.shape[0], len(segments)), np.nan, dtype=float)
    for segment_index, (start, end) in enumerate(segments):
        available = (confidence[:, start] > 0.0) & (confidence[:, end] > 0.0)
        if not np.any(available):
            continue
        deltas = coordinates[available, start] - coordinates[available, end]
        lengths[available, segment_index] = np.linalg.norm(deltas, axis=1)
    return lengths


def _robust_reference_by_segment(lengths: np.ndarray) -> np.ndarray:
    references = np.full((lengths.shape[1],), np.nan, dtype=float)
    for segment_index in range(lengths.shape[1]):
        values = lengths[:, segment_index]
        values = values[np.isfinite(values) & (values > _MIN_REFERENCE_LENGTH)]
        if values.size > 0:
            references[segment_index] = float(np.median(values))
    return references


def _frame_median_segment_length(
    coordinates: np.ndarray,
    confidence: np.ndarray,
    segments: tuple[tuple[int, int], ...],
) -> np.ndarray:
    lengths = _segment_lengths(coordinates, confidence, segments)
    scales = np.zeros((confidence.shape[0],), dtype=float)
    for frame_index in range(confidence.shape[0]):
        values = lengths[frame_index]
        values = values[np.isfinite(values) & (values > _MIN_REFERENCE_LENGTH)]
        if values.size > 0:
            scales[frame_index] = float(np.median(values))
    return scales
