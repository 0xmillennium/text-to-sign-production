"""Deterministic active signing span metrics."""

from __future__ import annotations

import math
from typing import cast

import numpy as np

from text_to_sign_production.legacy_data.metrics.support import (
    UPPER_BODY_SUPPORT_LANDMARK_INDICES,
)
from text_to_sign_production.legacy_data.metrics.types import ActiveSigningSpanMetrics
from text_to_sign_production.legacy_data.samples.types import ProcessedSamplePayload

_SUSTAINED_EVIDENCE_WINDOW_RADIUS = 2
_MIN_SUSTAINED_EVIDENCE_FRAMES = 3
_MAX_BRIDGED_GAP_RATIO = 0.05
_CONTEXT_PADDING_RATIO = 0.03
_MIN_UPPER_BODY_SUPPORT_LANDMARKS = 4
_HAND_MOTION_EVIDENCE_THRESHOLD = 0.004
_UPPER_BODY_MOTION_EVIDENCE_THRESHOLD = 0.003


def compute_active_signing_span_metrics(
    payload: ProcessedSamplePayload,
) -> ActiveSigningSpanMetrics:
    """Compute the reusable signing-relevant span from sustained body/hand evidence."""
    num_frames = payload.num_frames
    if num_frames <= 0:
        raise ValueError(f"Invalid num_frames ({num_frames}) for active signing span metrics.")

    body_conf = np.asarray(payload.pose.body.confidence)
    body_coords = np.asarray(payload.pose.body.coordinates)
    left_hand_conf = np.asarray(payload.pose.left_hand.confidence)
    left_hand_coords = np.asarray(payload.pose.left_hand.coordinates)
    right_hand_conf = np.asarray(payload.pose.right_hand.confidence)
    right_hand_coords = np.asarray(payload.pose.right_hand.coordinates)

    any_hand_evidence = np.any(left_hand_conf > 0.0, axis=1) | np.any(right_hand_conf > 0.0, axis=1)
    upper_body_evidence = (
        np.count_nonzero(body_conf[:, UPPER_BODY_SUPPORT_LANDMARK_INDICES] > 0.0, axis=1)
        >= _MIN_UPPER_BODY_SUPPORT_LANDMARKS
    )
    motion_evidence = (
        _centroid_motion_evidence(left_hand_coords, left_hand_conf, _HAND_MOTION_EVIDENCE_THRESHOLD)
        | _centroid_motion_evidence(
            right_hand_coords,
            right_hand_conf,
            _HAND_MOTION_EVIDENCE_THRESHOLD,
        )
        | _landmark_motion_evidence(
            body_coords[:, UPPER_BODY_SUPPORT_LANDMARK_INDICES],
            body_conf[:, UPPER_BODY_SUPPORT_LANDMARK_INDICES],
            _UPPER_BODY_MOTION_EVIDENCE_THRESHOLD,
        )
    )
    signing_evidence = any_hand_evidence & (upper_body_evidence | motion_evidence)
    sustained_any_hand_evidence = _bridge_short_gaps(
        _sustained_evidence_mask(any_hand_evidence),
        num_frames,
    )
    sustained_upper_body_evidence = _bridge_short_gaps(
        _sustained_evidence_mask(upper_body_evidence),
        num_frames,
    )
    sustained_motion_evidence = _bridge_short_gaps(
        _sustained_evidence_mask(motion_evidence),
        num_frames,
    )
    sustained_signing_evidence = _bridge_short_gaps(
        _sustained_evidence_mask(signing_evidence),
        num_frames,
    )
    sustained_indices = np.flatnonzero(sustained_signing_evidence)

    if sustained_indices.size == 0:
        start = 0
        end = num_frames
        source = "full_clip_no_sustained_signing_evidence"
    else:
        padding = max(1, math.ceil(num_frames * _CONTEXT_PADDING_RATIO))
        start = max(0, int(sustained_indices[0]) - padding)
        end = min(num_frames, int(sustained_indices[-1]) + 1 + padding)
        source = _span_source(start, end, num_frames)

    frame_count = end - start
    sustained_any_hand_count = int(np.count_nonzero(sustained_any_hand_evidence[start:end]))
    sustained_upper_body_count = int(np.count_nonzero(sustained_upper_body_evidence[start:end]))
    sustained_motion_count = int(np.count_nonzero(sustained_motion_evidence[start:end]))
    return ActiveSigningSpanMetrics(
        start_frame_index=start,
        end_frame_index_exclusive=end,
        frame_count=frame_count,
        frame_ratio=frame_count / num_frames,
        trimmed_prefix_frame_count=start,
        trimmed_prefix_frame_ratio=start / num_frames,
        trimmed_suffix_frame_count=num_frames - end,
        trimmed_suffix_frame_ratio=(num_frames - end) / num_frames,
        sustained_any_hand_evidence_frame_count=sustained_any_hand_count,
        sustained_any_hand_evidence_frame_ratio=sustained_any_hand_count / frame_count,
        sustained_upper_body_evidence_frame_count=sustained_upper_body_count,
        sustained_upper_body_evidence_frame_ratio=sustained_upper_body_count / frame_count,
        sustained_motion_evidence_frame_count=sustained_motion_count,
        sustained_motion_evidence_frame_ratio=sustained_motion_count / frame_count,
        source=source,
    )


def _span_source(start: int, end: int, num_frames: int) -> str:
    if start > 0 and end < num_frames:
        return "active_signing_span_trimmed_prefix_and_suffix"
    if start > 0:
        return "active_signing_span_trimmed_prefix"
    if end < num_frames:
        return "active_signing_span_trimmed_suffix"
    return "full_clip_sustained_signing_evidence"


def _sustained_evidence_mask(frame_evidence: np.ndarray) -> np.ndarray:
    """Return frames supported by local evidence continuity."""
    num_frames = frame_evidence.shape[0]
    if num_frames == 0:
        return np.zeros((0,), dtype=np.bool_)

    min_required = min(_MIN_SUSTAINED_EVIDENCE_FRAMES, num_frames)
    sustained = np.zeros((num_frames,), dtype=np.bool_)
    for index in range(num_frames):
        start = max(0, index - _SUSTAINED_EVIDENCE_WINDOW_RADIUS)
        end = min(num_frames, index + _SUSTAINED_EVIDENCE_WINDOW_RADIUS + 1)
        if int(np.count_nonzero(frame_evidence[start:end])) >= min_required:
            sustained[index] = True
    return cast(np.ndarray, sustained & frame_evidence)


def _bridge_short_gaps(mask: np.ndarray, num_frames: int) -> np.ndarray:
    """Fill small internal evidence gaps without extending the evidence envelope."""
    if not np.any(mask):
        return mask

    max_gap = max(1, math.ceil(num_frames * _MAX_BRIDGED_GAP_RATIO))
    bridged = mask.copy()
    true_indices = np.flatnonzero(mask)
    previous = int(true_indices[0])
    for current_value in true_indices[1:]:
        current = int(current_value)
        gap = current - previous - 1
        if 0 < gap <= max_gap:
            bridged[previous + 1 : current] = True
        previous = current
    return bridged


def _centroid_motion_evidence(
    coordinates: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
) -> np.ndarray:
    centroids, available = _frame_centroids(coordinates, confidence)
    return _motion_evidence_from_centroids(centroids, available, threshold)


def _landmark_motion_evidence(
    coordinates: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
) -> np.ndarray:
    num_frames = confidence.shape[0]
    evidence = np.zeros((num_frames,), dtype=np.bool_)
    if num_frames < 2:
        return evidence

    for index in range(1, num_frames):
        shared = (confidence[index - 1] > 0.0) & (confidence[index] > 0.0)
        if not np.any(shared):
            continue
        deltas = coordinates[index, shared] - coordinates[index - 1, shared]
        # Parser outputs are already normalized coordinates, so distances are
        # normalized coordinate-space distances and must not be canvas-scaled.
        max_distance = float(np.max(np.linalg.norm(deltas, axis=1)))
        if max_distance >= threshold:
            evidence[index - 1] = True
            evidence[index] = True
    return evidence


def _motion_evidence_from_centroids(
    centroids: np.ndarray,
    available: np.ndarray,
    threshold: float,
) -> np.ndarray:
    num_frames = available.shape[0]
    evidence = np.zeros((num_frames,), dtype=np.bool_)
    if num_frames < 2:
        return evidence

    for index in range(1, num_frames):
        if not (available[index - 1] and available[index]):
            continue
        # Centroids are computed from normalized parser coordinates.
        distance = float(np.linalg.norm(centroids[index] - centroids[index - 1]))
        if distance >= threshold:
            evidence[index - 1] = True
            evidence[index] = True
    return evidence


def _frame_centroids(
    coordinates: np.ndarray,
    confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if coordinates.ndim != 3 or confidence.ndim != 2:
        raise ValueError(
            f"Expected coordinates [frame, landmark, xy] and confidence [frame, landmark], "
            f"got {coordinates.shape} and {confidence.shape}."
        )
    available = np.any(confidence > 0.0, axis=1)
    centroids = np.zeros((confidence.shape[0], 2), dtype=float)
    for index in np.flatnonzero(available):
        landmark_mask = confidence[index] > 0.0
        centroids[index] = np.mean(coordinates[index, landmark_mask], axis=0)
    return centroids, available
