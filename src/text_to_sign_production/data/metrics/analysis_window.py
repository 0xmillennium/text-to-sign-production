"""Active-signing analysis window metrics."""

from __future__ import annotations

import math

import numpy as np

from text_to_sign_production.data.metrics.types import AnalysisWindowMetrics
from text_to_sign_production.data.samples.types import ProcessedSamplePayload

_SUSTAINED_EVIDENCE_WINDOW_RADIUS = 2
_MIN_SUSTAINED_EVIDENCE_FRAMES = 2
_MAX_BRIDGED_GAP_RATIO = 0.05
_CONTEXT_PADDING_RATIO = 0.05


def compute_analysis_window_metrics(payload: ProcessedSamplePayload) -> AnalysisWindowMetrics:
    """Compute the deterministic active-signing span used by hand quality metrics."""
    num_frames = payload.num_frames
    if num_frames <= 0:
        raise ValueError(f"Invalid num_frames ({num_frames}) for analysis-window metrics.")

    left_hand_conf = np.asarray(payload.pose.left_hand.confidence)
    right_hand_conf = np.asarray(payload.pose.right_hand.confidence)
    any_hand_available = (
        np.any(left_hand_conf > 0.0, axis=1) | np.any(right_hand_conf > 0.0, axis=1)
    )

    sustained_hand_evidence = _sustained_evidence_mask(any_hand_available)
    sustained_hand_evidence = _bridge_short_gaps(sustained_hand_evidence, num_frames)
    sustained_indices = np.flatnonzero(sustained_hand_evidence)

    if sustained_indices.size == 0:
        start = 0
        end = num_frames
        source = "full_clip_no_sustained_hand_evidence"
    else:
        padding = max(1, math.ceil(num_frames * _CONTEXT_PADDING_RATIO))
        start = max(0, int(sustained_indices[0]) - padding)
        end = min(num_frames, int(sustained_indices[-1]) + 1 + padding)
        if start > 0 and end < num_frames:
            source = "active_span_trimmed_prefix_and_terminal_rest"
        elif start > 0:
            source = "active_span_trimmed_prefix_rest"
        elif end < num_frames:
            source = "active_span_trimmed_terminal_rest"
        else:
            source = "full_clip_sustained_hand_evidence"

    frame_count = end - start
    pre_sign_excluded = start
    post_sign_excluded = num_frames - end
    sustained_count = int(np.count_nonzero(sustained_hand_evidence[start:end]))
    return AnalysisWindowMetrics(
        start_frame_index=start,
        end_frame_index_exclusive=end,
        frame_count=frame_count,
        frame_ratio=frame_count / num_frames,
        pre_sign_excluded_frame_count=pre_sign_excluded,
        pre_sign_excluded_frame_ratio=pre_sign_excluded / num_frames,
        post_sign_excluded_frame_count=post_sign_excluded,
        post_sign_excluded_frame_ratio=post_sign_excluded / num_frames,
        sustained_hand_evidence_frame_count=sustained_count,
        sustained_hand_evidence_frame_ratio=sustained_count / frame_count,
        source=source,
    )


def _sustained_evidence_mask(any_hand_available: np.ndarray) -> np.ndarray:
    """Return frames supported by local hand-evidence continuity."""
    num_frames = any_hand_available.shape[0]
    if num_frames == 0:
        return np.zeros((0,), dtype=np.bool_)

    min_required = min(_MIN_SUSTAINED_EVIDENCE_FRAMES, num_frames)
    sustained = np.zeros((num_frames,), dtype=np.bool_)
    for index in range(num_frames):
        start = max(0, index - _SUSTAINED_EVIDENCE_WINDOW_RADIUS)
        end = min(num_frames, index + _SUSTAINED_EVIDENCE_WINDOW_RADIUS + 1)
        if int(np.count_nonzero(any_hand_available[start:end])) >= min_required:
            sustained[index] = True
    return sustained & any_hand_available


def _bridge_short_gaps(mask: np.ndarray, num_frames: int) -> np.ndarray:
    """Fill small gaps inside sustained evidence without extending the span."""
    if not np.any(mask):
        return mask

    max_gap = max(1, math.ceil(num_frames * _MAX_BRIDGED_GAP_RATIO))
    bridged = mask.copy()
    true_indices = np.flatnonzero(mask)
    previous = int(true_indices[0])
    for current_value in true_indices[1:]:
        current = int(current_value)
        gap = current - previous - 1
        if 0 < gap <= max_gap:
            bridged[previous + 1 : current] = True
        previous = current
    return bridged
