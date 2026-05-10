"""Spatial out-of-bounds family metrics."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.data.tier.families.types import OobMetrics, QualityMetricBuildInput


def compute_oob_metrics(input: QualityMetricBuildInput) -> OobMetrics:
    """Measure how often pose channels contain spatially out-of-bounds landmarks."""
    body = _channel_oob_ratio(input.sample.pose.body_xyc)
    left = _channel_oob_ratio(input.sample.pose.left_hand_xyc)
    right = _channel_oob_ratio(input.sample.pose.right_hand_xyc)
    face = _channel_oob_ratio(input.sample.pose.face_xyc)
    return OobMetrics(
        body_out_of_bounds_frame_ratio=body,
        left_hand_out_of_bounds_frame_ratio=left,
        right_hand_out_of_bounds_frame_ratio=right,
        face_out_of_bounds_frame_ratio=face,
        max_channel_out_of_bounds_frame_ratio=max(body, left, right, face),
    )


def _channel_oob_ratio(xyc: np.ndarray) -> float:
    coordinates = xyc[..., :2]
    confidences = xyc[..., 2]
    frame_count = coordinates.shape[0]
    if frame_count == 0:
        return 0.0
    observed = confidences > 0.0
    oob_points = np.any((coordinates < 0.0) | (coordinates > 1.0), axis=2) & observed
    return float(int(np.count_nonzero(np.any(oob_points, axis=1))) / frame_count)


__all__ = ["compute_oob_metrics"]
