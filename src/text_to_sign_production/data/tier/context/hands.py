"""Representative-hand context builder."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.core.models import PreparedSample
from text_to_sign_production.data.tier.context.types import (
    ActiveSpanContext,
    HandLabel,
    RepresentativeHandContext,
    RepresentativeHandFrame,
)

HAND_FINGERTIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)
HAND_DISTAL_CHAIN_INDICES: tuple[int, ...] = (3, 4, 7, 8, 11, 12, 15, 16, 19, 20)


def build_representative_hand_context(
    sample: PreparedSample,
    active_span: ActiveSpanContext,
) -> RepresentativeHandContext:
    """Build stable hand-only representative context from PreparedSample."""
    left_quality = _detail_quality(sample.pose.left_hand_xyc)
    right_quality = _detail_quality(sample.pose.right_hand_xyc)
    frames: list[RepresentativeHandFrame] = []
    previous: HandLabel | None = None
    for frame_index in range(active_span.frame_count):
        selected = _select_hand(left_quality, right_quality, frame_index)
        if selected is None or not active_span.active_frame_mask[frame_index]:
            frames.append(
                RepresentativeHandFrame(
                    frame_index,
                    None,
                    False,
                    0,
                    0,
                    0,
                    False,
                    previous is not None,
                )
            )
            previous = None
            continue
        confidence = (
            sample.pose.left_hand_xyc[frame_index, :, 2]
            if selected is HandLabel.LEFT
            else sample.pose.right_hand_xyc[frame_index, :, 2]
        )
        landmark_count = int(np.count_nonzero(confidence > 0.0))
        fingertip_count = int(np.count_nonzero(confidence[list(HAND_FINGERTIP_INDICES)] > 0.0))
        distal_count = int(np.count_nonzero(confidence[list(HAND_DISTAL_CHAIN_INDICES)] > 0.0))
        frames.append(
            RepresentativeHandFrame(
                frame_index,
                selected,
                True,
                landmark_count,
                fingertip_count,
                distal_count,
                fingertip_count > 0 and distal_count > 0,
                False,
            )
        )
        previous = selected
    return RepresentativeHandContext(frames=tuple(frames))


def _select_hand(
    left_quality: np.ndarray,
    right_quality: np.ndarray,
    frame_index: int,
) -> HandLabel | None:
    left = float(left_quality[frame_index])
    right = float(right_quality[frame_index])
    if left <= 0.0 and right <= 0.0:
        return None
    return HandLabel.LEFT if left >= right else HandLabel.RIGHT


def _detail_quality(xyc: np.ndarray) -> np.ndarray:
    positive = xyc[..., 2] > 0.0
    landmark = np.count_nonzero(positive, axis=1)
    fingertip = np.count_nonzero(positive[:, list(HAND_FINGERTIP_INDICES)], axis=1)
    distal = np.count_nonzero(positive[:, list(HAND_DISTAL_CHAIN_INDICES)], axis=1)
    return np.asarray(landmark + fingertip + distal, dtype=float)


__all__ = [
    "HAND_DISTAL_CHAIN_INDICES",
    "HAND_FINGERTIP_INDICES",
    "build_representative_hand_context",
]
