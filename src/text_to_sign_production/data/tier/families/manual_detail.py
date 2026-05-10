"""Manual detail family metrics."""

from __future__ import annotations

from text_to_sign_production.data.tier.context.hands import (
    HAND_DISTAL_CHAIN_INDICES,
    HAND_FINGERTIP_INDICES,
)
from text_to_sign_production.data.tier.families.types import (
    ManualDetailMetrics,
    QualityMetricBuildInput,
)

_HAND_LANDMARK_COUNT = 21


def compute_manual_detail_metrics(input: QualityMetricBuildInput) -> ManualDetailMetrics:
    """Measure representative-hand fine detail over the shared active span."""
    active = input.quality_context.active_span.active_frame_mask
    frames = input.quality_context.representative_hand.frames
    active_frames = [frame for frame in frames if active[frame.frame_index]]
    active_count = len(active_frames)
    detail_dropout = tuple(
        active[frame.frame_index] and not frame.detail_supported for frame in frames
    )
    return ManualDetailMetrics(
        active_span_representative_hand_landmark_coverage_ratio=_ratio(
            sum(frame.landmark_support_count for frame in active_frames),
            active_count * _HAND_LANDMARK_COUNT,
        ),
        active_span_representative_hand_fingertip_coverage_ratio=_ratio(
            sum(frame.fingertip_support_count for frame in active_frames),
            active_count * len(HAND_FINGERTIP_INDICES),
        ),
        active_span_representative_hand_distal_chain_coverage_ratio=_ratio(
            sum(frame.distal_chain_support_count for frame in active_frames),
            active_count * len(HAND_DISTAL_CHAIN_INDICES),
        ),
        max_active_span_representative_hand_detail_dropout_run_ratio=_ratio(
            _max_run(detail_dropout),
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


__all__ = ["compute_manual_detail_metrics"]
