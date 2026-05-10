"""Upper-body/signing-relevant support metrics."""

from __future__ import annotations

from typing import Final

import numpy as np

from text_to_sign_production.legacy_data.metrics.types import (
    ActiveSigningSpanMetrics,
    UpperBodySupportMetrics,
)
from text_to_sign_production.legacy_data.samples.types import ProcessedSamplePayload

UPPER_BODY_SUPPORT_LANDMARKS: Final[dict[int, str]] = {
    1: "neck",
    2: "right_shoulder",
    3: "right_elbow",
    4: "right_wrist",
    5: "left_shoulder",
    6: "left_elbow",
    7: "left_wrist",
}
UPPER_BODY_SUPPORT_LANDMARK_INDICES: Final[tuple[int, ...]] = tuple(UPPER_BODY_SUPPORT_LANDMARKS)


def compute_upper_body_support_metrics(
    payload: ProcessedSamplePayload,
    active_signing_span: ActiveSigningSpanMetrics,
) -> UpperBodySupportMetrics:
    """Compute active-span binding and full-clip diagnostic upper-body support."""
    body_conf = np.asarray(payload.pose.body.confidence)
    if body_conf.size == 0:
        raise ValueError("Cannot compute upper-body support for empty body confidence.")
    active_body_conf = body_conf[
        active_signing_span.start_frame_index : active_signing_span.end_frame_index_exclusive,
        UPPER_BODY_SUPPORT_LANDMARK_INDICES,
    ]
    if active_body_conf.shape[0] != active_signing_span.frame_count:
        raise ValueError("Active signing span frame count does not match body confidence array.")
    full_clip_body_conf = body_conf[:, UPPER_BODY_SUPPORT_LANDMARK_INDICES]
    return UpperBodySupportMetrics(
        active_span_upper_body_support_landmark_coverage_ratio=_coverage_ratio(active_body_conf),
        full_clip_upper_body_support_landmark_coverage_ratio=_coverage_ratio(full_clip_body_conf),
    )


def _coverage_ratio(confidence: np.ndarray) -> float:
    if confidence.size == 0:
        raise ValueError("Cannot compute upper-body support coverage for empty confidence.")
    return float(np.count_nonzero(confidence > 0.0) / confidence.size)
