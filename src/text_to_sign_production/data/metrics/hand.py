"""Hand metrics computation."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.data.metrics.types import HandMetrics
from text_to_sign_production.data.samples.types import PassedManifestEntry, ProcessedSamplePayload


def compute_hand_metrics(
    payload: ProcessedSamplePayload, manifest: PassedManifestEntry
) -> HandMetrics:
    """Compute temporal hand availability metrics."""
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

    left_ratio = left_hand_available_frame_count / num_frames
    right_ratio = right_hand_available_frame_count / num_frames
    any_ratio = any_hand_available_frame_count / num_frames
    max_unavailable_run = _longest_false_run(any_frame_available)

    return HandMetrics(
        left_hand_available_frame_count=left_hand_available_frame_count,
        right_hand_available_frame_count=right_hand_available_frame_count,
        any_hand_available_frame_count=any_hand_available_frame_count,
        left_hand_available_frame_ratio=left_ratio,
        right_hand_available_frame_ratio=right_ratio,
        any_hand_available_frame_ratio=any_ratio,
        max_any_hand_unavailable_run_count=max_unavailable_run,
        max_any_hand_unavailable_run_ratio=max_unavailable_run / num_frames,
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
