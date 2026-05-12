"""Confidence family metrics."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.data.tier.families.types import (
    ConfidenceMetrics,
    QualityMetricBuildInput,
)


def compute_confidence_metrics(input: QualityMetricBuildInput) -> ConfidenceMetrics:
    """Measure mean confidence over active signing without deriving availability."""
    active = np.asarray(input.quality_context.active_span.active_frame_mask, dtype=np.bool_)
    hand_conf = np.concatenate(
        (
            input.sample.pose.left_hand_xyc[..., 2],
            input.sample.pose.right_hand_xyc[..., 2],
        ),
        axis=1,
    )
    return ConfidenceMetrics(
        active_span_body_observed_mean_confidence=_mean_active_observed_confidence(
            input.sample.pose.body_xyc[..., 2], active
        ),
        active_span_hand_observed_mean_confidence=_mean_active_observed_confidence(
            hand_conf, active
        ),
        active_span_face_observed_mean_confidence=_mean_active_observed_confidence(
            input.sample.pose.face_xyc[..., 2], active
        ),
    )


def _mean_active_observed_confidence(confidences: np.ndarray, active: np.ndarray) -> float:
    if not np.any(active):
        return 0.0
    values = confidences[active]
    values = values[values > 0.0]
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


__all__ = ["compute_confidence_metrics"]
