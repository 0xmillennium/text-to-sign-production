"""Geometry reference context builder."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.core.models import PreparedSample
from text_to_sign_production.data.tier.context.types import (
    GeometryFrameContext,
    GeometryReferenceContext,
    HandLabel,
    RepresentativeHandContext,
)

UPPER_BODY_BONE_SEGMENTS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (2, 5),
)
CROSS_CHANNEL_BODY_SCALE_SEGMENTS: tuple[tuple[int, int], ...] = ((2, 5), (1, 2), (1, 5))
CROSS_CHANNEL_HAND_SCALE_SEGMENTS: tuple[tuple[int, int], ...] = ((0, 5), (0, 17), (5, 17))
_MIN_REFERENCE_LENGTH = 1e-12


def build_geometry_reference_context(
    sample: PreparedSample,
    hand_context: RepresentativeHandContext,
) -> GeometryReferenceContext:
    """Build sample-internal geometry reference context."""
    body = sample.pose.body_xyc
    hand = _representative_hand_array(sample, hand_context)
    upper_lengths = _segment_lengths(body[..., :2], body[..., 2], UPPER_BODY_BONE_SEGMENTS)
    body_scale = _frame_median_segment_length(
        body[..., :2], body[..., 2], CROSS_CHANNEL_BODY_SCALE_SEGMENTS
    )
    hand_scale = _frame_median_segment_length(
        hand[..., :2], hand[..., 2], CROSS_CHANNEL_HAND_SCALE_SEGMENTS
    )
    frames = tuple(
        GeometryFrameContext(
            frame_index=frame_index,
            upper_body_segment_lengths=_frame_segment_mapping(
                upper_lengths, UPPER_BODY_BONE_SEGMENTS, frame_index
            ),
            body_scale_reference_length=_finite_float_or_none(body_scale[frame_index]),
            hand_scale_reference_length=_finite_float_or_none(hand_scale[frame_index]),
        )
        for frame_index in range(sample.pose.frame_count)
    )
    return GeometryReferenceContext(
        upper_body_segment_references=_reference_mapping(upper_lengths, UPPER_BODY_BONE_SEGMENTS),
        cross_channel_scale_reference=_cross_channel_reference(body_scale, hand_scale),
        frames=frames,
    )


def _representative_hand_array(
    sample: PreparedSample,
    hand_context: RepresentativeHandContext,
) -> np.ndarray:
    hand = np.zeros_like(sample.pose.left_hand_xyc, dtype=np.float32)
    for frame in hand_context.frames:
        if frame.selected_hand is HandLabel.LEFT:
            hand[frame.frame_index] = sample.pose.left_hand_xyc[frame.frame_index]
        elif frame.selected_hand is HandLabel.RIGHT:
            hand[frame.frame_index] = sample.pose.right_hand_xyc[frame.frame_index]
    return hand


def _segment_lengths(
    coordinates: np.ndarray,
    confidence: np.ndarray,
    segments: tuple[tuple[int, int], ...],
) -> np.ndarray:
    lengths = np.full((confidence.shape[0], len(segments)), np.nan, dtype=float)
    for segment_index, (start, end) in enumerate(segments):
        available = (confidence[:, start] > 0.0) & (confidence[:, end] > 0.0)
        if np.any(available):
            deltas = coordinates[available, start] - coordinates[available, end]
            lengths[available, segment_index] = np.linalg.norm(deltas, axis=1)
    return lengths


def _reference_mapping(
    lengths: np.ndarray,
    segments: tuple[tuple[int, int], ...],
) -> dict[tuple[int, int], float | None]:
    references: dict[tuple[int, int], float | None] = {}
    for segment_index, segment in enumerate(segments):
        values = lengths[:, segment_index]
        values = values[np.isfinite(values) & (values > _MIN_REFERENCE_LENGTH)]
        references[segment] = float(np.median(values)) if values.size else None
    return references


def _frame_segment_mapping(
    lengths: np.ndarray,
    segments: tuple[tuple[int, int], ...],
    frame_index: int,
) -> dict[tuple[int, int], float | None]:
    return {
        segment: _finite_float_or_none(lengths[frame_index, segment_index])
        for segment_index, segment in enumerate(segments)
    }


def _frame_median_segment_length(
    coordinates: np.ndarray,
    confidence: np.ndarray,
    segments: tuple[tuple[int, int], ...],
) -> np.ndarray:
    lengths = _segment_lengths(coordinates, confidence, segments)
    values = np.full((confidence.shape[0],), np.nan, dtype=float)
    for frame_index in range(confidence.shape[0]):
        frame_values = lengths[frame_index]
        frame_values = frame_values[
            np.isfinite(frame_values) & (frame_values > _MIN_REFERENCE_LENGTH)
        ]
        if frame_values.size:
            values[frame_index] = float(np.median(frame_values))
    return values


def _cross_channel_reference(body_scale: np.ndarray, hand_scale: np.ndarray) -> float | None:
    comparable = (
        np.isfinite(body_scale)
        & np.isfinite(hand_scale)
        & (body_scale > _MIN_REFERENCE_LENGTH)
        & (hand_scale > _MIN_REFERENCE_LENGTH)
    )
    if not np.any(comparable):
        return None
    ratios = hand_scale[comparable] / body_scale[comparable]
    return float(np.median(ratios)) if ratios.size else None


def _finite_float_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


__all__ = ["build_geometry_reference_context"]
