"""Confidence metrics computation."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.data.metrics.types import ActiveSigningSpanMetrics, ConfidenceMetrics
from text_to_sign_production.data.metrics.support import (
    UPPER_BODY_SUPPORT_LANDMARK_INDICES,
)
from text_to_sign_production.data.samples.types import ProcessedSamplePayload


def compute_confidence_metrics(
    payload: ProcessedSamplePayload,
    active_signing_span: ActiveSigningSpanMetrics,
) -> ConfidenceMetrics:
    """Compute confidence quality metrics without treating missingness as confidence."""
    full_body_conf = np.asarray(payload.pose.body.confidence)
    left_hand_conf = np.asarray(payload.pose.left_hand.confidence)
    right_hand_conf = np.asarray(payload.pose.right_hand.confidence)
    face_conf = np.asarray(payload.pose.face.confidence)

    active_body_conf = full_body_conf[
        active_signing_span.start_frame_index : active_signing_span.end_frame_index_exclusive,
        UPPER_BODY_SUPPORT_LANDMARK_INDICES,
    ]
    full_clip_upper_body_conf = full_body_conf[:, UPPER_BODY_SUPPORT_LANDMARK_INDICES]
    active_left_hand_conf = left_hand_conf[
        active_signing_span.start_frame_index : active_signing_span.end_frame_index_exclusive
    ]
    active_right_hand_conf = right_hand_conf[
        active_signing_span.start_frame_index : active_signing_span.end_frame_index_exclusive
    ]
    if (
        active_body_conf.shape[0] != active_signing_span.frame_count
        or active_left_hand_conf.shape[0] != active_signing_span.frame_count
        or active_right_hand_conf.shape[0] != active_signing_span.frame_count
    ):
        raise ValueError("Active signing span frame count does not match confidence arrays.")

    active_any_hand_conf = _best_usable_hand_frame_confidence(
        active_left_hand_conf,
        active_right_hand_conf,
    )
    all_conf = np.concatenate(
        [
            full_body_conf.flatten(),
            active_any_hand_conf,
            face_conf.flatten(),
        ]
    )

    return ConfidenceMetrics(
        active_span_body_available_mean_confidence=_available_mean(active_body_conf),
        full_clip_body_available_mean_confidence=_available_mean(full_clip_upper_body_conf),
        active_span_left_hand_available_mean_confidence=_available_mean(active_left_hand_conf),
        active_span_right_hand_available_mean_confidence=_available_mean(active_right_hand_conf),
        active_span_any_hand_available_mean_confidence=_available_mean(active_any_hand_conf),
        face_available_mean_confidence=_available_mean(face_conf),
        overall_available_mean_confidence=_available_mean(all_conf),
        body_nonzero_confidence_ratio=_nonzero_ratio(full_body_conf),
        left_hand_nonzero_confidence_ratio=_nonzero_ratio(left_hand_conf),
        right_hand_nonzero_confidence_ratio=_nonzero_ratio(right_hand_conf),
        face_nonzero_confidence_ratio=_nonzero_ratio(face_conf),
        overall_nonzero_confidence_ratio=_nonzero_ratio(all_conf),
    )


def _best_usable_hand_frame_confidence(
    left_hand_conf: np.ndarray,
    right_hand_conf: np.ndarray,
) -> np.ndarray:
    """Return per-frame best usable hand mean without concatenating hand streams."""
    left_mean = _frame_available_mean(left_hand_conf)
    right_mean = _frame_available_mean(right_hand_conf)
    return np.maximum(left_mean, right_mean)


def _available_mean(arr: np.ndarray) -> float:
    if arr.size == 0:
        raise ValueError("Cannot compute mean confidence for empty array.")
    available = arr[arr > 0.0]
    if available.size == 0:
        return 0.0
    return float(np.mean(available))


def _nonzero_ratio(arr: np.ndarray) -> float:
    if arr.size == 0:
        raise ValueError("Cannot compute nonzero confidence ratio for empty array.")
    return float(np.count_nonzero(arr > 0.0) / arr.size)


def _frame_available_mean(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D hand confidence array, got shape {arr.shape}.")
    positive = arr > 0.0
    positive_counts = np.count_nonzero(positive, axis=1)
    summed = np.sum(np.where(positive, arr, 0.0), axis=1)
    means = np.zeros((arr.shape[0],), dtype=float)
    np.divide(summed, positive_counts, out=means, where=positive_counts > 0)
    return means
