"""Hand metrics computation."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.data.metrics.types import HandMetrics
from text_to_sign_production.data.samples.types import PassedManifestEntry, ProcessedSamplePayload


def compute_hand_metrics(
    payload: ProcessedSamplePayload, manifest: PassedManifestEntry
) -> HandMetrics:
    """Compute hand presence metrics."""
    left_hand_conf = np.asarray(payload.pose.left_hand.confidence)
    right_hand_conf = np.asarray(payload.pose.right_hand.confidence)

    left_frame_nonzero = np.any(left_hand_conf > 0.0, axis=1)
    right_frame_nonzero = np.any(right_hand_conf > 0.0, axis=1)
    any_frame_nonzero = left_frame_nonzero | right_frame_nonzero

    left_hand_nonzero_frame_count = int(np.count_nonzero(left_frame_nonzero))
    right_hand_nonzero_frame_count = int(np.count_nonzero(right_frame_nonzero))
    any_hand_nonzero_frame_count = int(np.count_nonzero(any_frame_nonzero))

    num_frames = payload.num_frames
    if num_frames <= 0:
        raise ValueError(f"Invalid num_frames ({num_frames}) for hand metrics.")

    left_ratio = left_hand_nonzero_frame_count / num_frames
    right_ratio = right_hand_nonzero_frame_count / num_frames
    any_ratio = any_hand_nonzero_frame_count / num_frames

    return HandMetrics(
        left_hand_nonzero_frame_count=left_hand_nonzero_frame_count,
        right_hand_nonzero_frame_count=right_hand_nonzero_frame_count,
        any_hand_nonzero_frame_count=any_hand_nonzero_frame_count,
        left_hand_nonzero_frame_ratio=left_ratio,
        right_hand_nonzero_frame_ratio=right_ratio,
        any_hand_nonzero_frame_ratio=any_ratio,
    )
