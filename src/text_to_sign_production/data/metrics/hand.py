"""Hand metrics computation."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.data.metrics.types import (
    ActiveSigningSpanMetrics,
    ManualVisibilityMetrics,
)
from text_to_sign_production.data.samples.types import PassedManifestEntry, ProcessedSamplePayload


def compute_hand_metrics(
    payload: ProcessedSamplePayload,
    manifest: PassedManifestEntry,
    active_signing_span: ActiveSigningSpanMetrics,
) -> ManualVisibilityMetrics:
    """Compute active-signing-span manual visibility metrics."""
    left_hand_conf = np.asarray(payload.pose.left_hand.confidence)
    right_hand_conf = np.asarray(payload.pose.right_hand.confidence)

    left_frame_available = np.any(left_hand_conf > 0.0, axis=1)
    right_frame_available = np.any(right_hand_conf > 0.0, axis=1)
    any_frame_available = left_frame_available | right_frame_available

    left_hand_available_frame_count = int(np.count_nonzero(left_frame_available))
    right_hand_available_frame_count = int(np.count_nonzero(right_frame_available))
    any_hand_available_frame_count = int(np.count_nonzero(any_frame_available))

    num_frames = payload.num_frames
    if num_frames <= 0:
        raise ValueError(f"Invalid num_frames ({num_frames}) for hand metrics.")

    active_any_frame_available = any_frame_available[
        active_signing_span.start_frame_index : active_signing_span.end_frame_index_exclusive
    ]
    if active_any_frame_available.size != active_signing_span.frame_count:
        raise ValueError("Active signing span frame count does not match hand confidence arrays.")

    active_any_hand_available_frame_count = int(np.count_nonzero(active_any_frame_available))
    max_active_unavailable_run = _longest_false_run(active_any_frame_available)

    return ManualVisibilityMetrics(
        whole_clip_left_hand_available_frame_count=left_hand_available_frame_count,
        whole_clip_right_hand_available_frame_count=right_hand_available_frame_count,
        whole_clip_any_hand_available_frame_count=any_hand_available_frame_count,
        whole_clip_left_hand_available_frame_ratio=left_hand_available_frame_count / num_frames,
        whole_clip_right_hand_available_frame_ratio=right_hand_available_frame_count / num_frames,
        whole_clip_any_hand_available_frame_ratio=any_hand_available_frame_count / num_frames,
        active_span_any_hand_available_frame_count=active_any_hand_available_frame_count,
        active_span_any_hand_available_frame_ratio=(
            active_any_hand_available_frame_count / active_signing_span.frame_count
        ),
        max_active_span_any_hand_unavailable_run_count=max_active_unavailable_run,
        max_active_span_any_hand_unavailable_run_ratio=(
            max_active_unavailable_run / active_signing_span.frame_count
        ),
    )


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
