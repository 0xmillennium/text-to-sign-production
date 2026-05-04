"""Confidence metrics computation."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.data.metrics.types import ConfidenceMetrics
from text_to_sign_production.data.samples.types import ProcessedSamplePayload


def compute_confidence_metrics(payload: ProcessedSamplePayload) -> ConfidenceMetrics:
    """Compute confidence metrics."""
    body_conf = np.asarray(payload.pose.body.confidence)
    left_hand_conf = np.asarray(payload.pose.left_hand.confidence)
    right_hand_conf = np.asarray(payload.pose.right_hand.confidence)
    face_conf = np.asarray(payload.pose.face.confidence)

    def _mean(arr: np.ndarray) -> float:
        if arr.size == 0:
            raise ValueError("Cannot compute mean confidence for empty array.")
        return float(np.mean(arr))

    def _nonzero_ratio(arr: np.ndarray) -> float:
        if arr.size == 0:
            raise ValueError("Cannot compute nonzero confidence ratio for empty array.")
        return float(np.count_nonzero(arr > 0.0) / arr.size)

    all_conf = np.concatenate(
        [
            body_conf.flatten(),
            left_hand_conf.flatten(),
            right_hand_conf.flatten(),
            face_conf.flatten(),
        ]
    )

    return ConfidenceMetrics(
        body_mean_confidence=_mean(body_conf),
        left_hand_mean_confidence=_mean(left_hand_conf),
        right_hand_mean_confidence=_mean(right_hand_conf),
        face_mean_confidence=_mean(face_conf),
        overall_mean_confidence=_mean(all_conf),
        body_nonzero_confidence_ratio=_nonzero_ratio(body_conf),
        left_hand_nonzero_confidence_ratio=_nonzero_ratio(left_hand_conf),
        right_hand_nonzero_confidence_ratio=_nonzero_ratio(right_hand_conf),
        face_nonzero_confidence_ratio=_nonzero_ratio(face_conf),
        overall_nonzero_confidence_ratio=_nonzero_ratio(all_conf),
    )
