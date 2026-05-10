"""Coarse manual visibility family metrics."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.data.tier.families.types import (
    ManualVisibilityMetrics,
    QualityMetricBuildInput,
)


def compute_manual_visibility_metrics(input: QualityMetricBuildInput) -> ManualVisibilityMetrics:
    """Measure coarse hand availability over the shared active signing span."""
    active = np.asarray(input.quality_context.active_span.active_frame_mask, dtype=np.bool_)
    left = np.any(input.sample.pose.left_hand_xyc[..., 2] > 0.0, axis=1)
    right = np.any(input.sample.pose.right_hand_xyc[..., 2] > 0.0, axis=1)
    any_hand = left | right
    both_unavailable = ~left & ~right
    active_count = int(np.count_nonzero(active))
    return ManualVisibilityMetrics(
        active_span_any_hand_available_frame_ratio=_ratio(
            int(np.count_nonzero(active & any_hand)),
            active_count,
        ),
        active_span_both_hands_unavailable_frame_ratio=_ratio(
            int(np.count_nonzero(active & both_unavailable)),
            active_count,
        ),
        max_active_span_any_hand_dropout_run_ratio=_ratio(
            _max_run(tuple(bool(value) for value in (active & ~any_hand).tolist())),
            active_count,
        ),
    )


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _max_run(mask: tuple[bool, ...]) -> int:
    longest = 0
    current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


__all__ = ["compute_manual_visibility_metrics"]
