"""Confidence metrics computation."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.data.metrics.types import AnalysisWindowMetrics, ConfidenceMetrics
from text_to_sign_production.data.samples.types import ProcessedSamplePayload


def compute_confidence_metrics(
    payload: ProcessedSamplePayload,
    analysis_window: AnalysisWindowMetrics,
) -> ConfidenceMetrics:
    """Compute confidence quality metrics without treating missingness as confidence."""
    body_conf = np.asarray(payload.pose.body.confidence)
    left_hand_conf = np.asarray(payload.pose.left_hand.confidence)
    right_hand_conf = np.asarray(payload.pose.right_hand.confidence)
    face_conf = np.asarray(payload.pose.face.confidence)

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

    active_left_hand_conf = left_hand_conf[
        analysis_window.start_frame_index : analysis_window.end_frame_index_exclusive
    ]
    active_right_hand_conf = right_hand_conf[
        analysis_window.start_frame_index : analysis_window.end_frame_index_exclusive
    ]
    if (
        active_left_hand_conf.shape[0] != analysis_window.frame_count
        or active_right_hand_conf.shape[0] != analysis_window.frame_count
    ):
        raise ValueError("Analysis-window frame count does not match hand confidence arrays.")

    active_any_hand_conf = np.maximum(
        _frame_available_mean(active_left_hand_conf),
        _frame_available_mean(active_right_hand_conf),
    )
    all_conf = np.concatenate(
        [
            body_conf.flatten(),
            active_any_hand_conf,
            face_conf.flatten(),
        ]
    )

    return ConfidenceMetrics(
        body_available_mean_confidence=_available_mean(body_conf),
        active_window_left_hand_available_mean_confidence=_available_mean(active_left_hand_conf),
        active_window_right_hand_available_mean_confidence=_available_mean(active_right_hand_conf),
        active_window_any_hand_available_mean_confidence=_available_mean(active_any_hand_conf),
        face_available_mean_confidence=_available_mean(face_conf),
        overall_available_mean_confidence=_available_mean(all_conf),
        body_nonzero_confidence_ratio=_nonzero_ratio(body_conf),
        left_hand_nonzero_confidence_ratio=_nonzero_ratio(left_hand_conf),
        right_hand_nonzero_confidence_ratio=_nonzero_ratio(right_hand_conf),
        face_nonzero_confidence_ratio=_nonzero_ratio(face_conf),
        overall_nonzero_confidence_ratio=_nonzero_ratio(all_conf),
    )
