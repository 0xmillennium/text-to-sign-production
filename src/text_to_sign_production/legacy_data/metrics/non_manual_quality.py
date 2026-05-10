"""Non-manual channel detail metrics."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.legacy_data.metrics.types import (
    ActiveSigningSpanMetrics,
    NonManualQualityMetrics,
)
from text_to_sign_production.legacy_data.pose.anatomy import (
    FACE_LOWER_REGION_INDICES,
    FACE_UPPER_REGION_INDICES,
)
from text_to_sign_production.legacy_data.samples.types import ProcessedSamplePayload


def compute_non_manual_quality_metrics(
    payload: ProcessedSamplePayload,
    active_signing_span: ActiveSigningSpanMetrics,
) -> NonManualQualityMetrics:
    """Compute active-span face detail and manual/face overlap metrics."""
    start = active_signing_span.start_frame_index
    end = active_signing_span.end_frame_index_exclusive
    face_conf = np.asarray(payload.pose.face.confidence)[start:end]
    left_hand_conf = np.asarray(payload.pose.left_hand.confidence)[start:end]
    right_hand_conf = np.asarray(payload.pose.right_hand.confidence)[start:end]
    if (
        face_conf.shape[0] != active_signing_span.frame_count
        or left_hand_conf.shape[0] != active_signing_span.frame_count
        or right_hand_conf.shape[0] != active_signing_span.frame_count
    ):
        raise ValueError("Active signing span frame count does not match quality arrays.")

    upper_face_conf = face_conf[:, FACE_UPPER_REGION_INDICES]
    lower_face_conf = face_conf[:, FACE_LOWER_REGION_INDICES]
    detail_available = np.any(upper_face_conf > 0.0, axis=1) & np.any(lower_face_conf > 0.0, axis=1)
    manual_available = np.any(left_hand_conf > 0.0, axis=1) | np.any(right_hand_conf > 0.0, axis=1)
    face_available = np.any(face_conf > 0.0, axis=1)
    manual_frame_count = int(np.count_nonzero(manual_available))
    manual_face_overlap_count = int(np.count_nonzero(manual_available & face_available))

    return NonManualQualityMetrics(
        active_span_face_landmark_coverage_ratio=_coverage_ratio(face_conf),
        active_span_upper_face_landmark_coverage_ratio=_coverage_ratio(upper_face_conf),
        active_span_lower_face_landmark_coverage_ratio=_coverage_ratio(lower_face_conf),
        max_active_span_face_detail_dropout_run_ratio=(
            _longest_false_run(detail_available) / active_signing_span.frame_count
        ),
        active_span_manual_face_overlap_frame_ratio=(
            manual_face_overlap_count / manual_frame_count if manual_frame_count > 0 else 0.0
        ),
    )


def _coverage_ratio(confidence: np.ndarray) -> float:
    if confidence.size == 0:
        raise ValueError("Cannot compute non-manual quality coverage for empty confidence.")
    return float(np.count_nonzero(confidence > 0.0) / confidence.size)


def _longest_false_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest
