"""Coverage metrics computation."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.data.metrics.types import CoverageMetrics
from text_to_sign_production.data.samples.types import PassedManifestEntry, ProcessedSamplePayload


def compute_coverage_metrics(
    payload: ProcessedSamplePayload, manifest: PassedManifestEntry
) -> CoverageMetrics:
    """Compute landmark completeness metrics independent of temporal availability."""
    num_frames = payload.num_frames
    if num_frames <= 0:
        raise ValueError(f"Invalid num_frames ({num_frames}) for coverage metrics.")

    body_conf = np.asarray(payload.pose.body.confidence)
    left_hand_conf = np.asarray(payload.pose.left_hand.confidence)
    right_hand_conf = np.asarray(payload.pose.right_hand.confidence)
    face_conf = np.asarray(payload.pose.face.confidence)

    left_hand_available = np.any(left_hand_conf > 0.0, axis=1)
    right_hand_available = np.any(right_hand_conf > 0.0, axis=1)
    face_available = np.any(face_conf > 0.0, axis=1)

    left_hand_ratio = _coverage_ratio(left_hand_conf, left_hand_available)
    right_hand_ratio = _coverage_ratio(right_hand_conf, right_hand_available)

    return CoverageMetrics(
        body_landmark_coverage_ratio=_coverage_ratio(body_conf),
        left_hand_landmark_coverage_ratio=left_hand_ratio,
        right_hand_landmark_coverage_ratio=right_hand_ratio,
        any_hand_landmark_coverage_ratio=max(left_hand_ratio, right_hand_ratio),
        face_landmark_coverage_ratio=_coverage_ratio(face_conf, face_available),
    )


def _coverage_ratio(confidence: np.ndarray, frame_mask: np.ndarray | None = None) -> float:
    """Return landmark density over all frames or over frames where a channel is available."""
    if confidence.size == 0:
        raise ValueError("Cannot compute coverage ratio for empty confidence array.")
    if frame_mask is not None:
        confidence = confidence[frame_mask]
        if confidence.size == 0:
            return 0.0
    return float(np.count_nonzero(confidence > 0.0) / confidence.size)
