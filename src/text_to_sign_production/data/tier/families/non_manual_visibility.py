"""Coarse non-manual visibility family metrics."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.data.tier.families.types import (
    NonManualVisibilityMetrics,
    QualityMetricBuildInput,
)


def compute_non_manual_visibility_metrics(
    input: QualityMetricBuildInput,
) -> NonManualVisibilityMetrics:
    """Measure coarse face availability over the shared active signing span."""
    active = np.asarray(input.quality_context.active_span.active_frame_mask, dtype=np.bool_)
    face_available = np.asarray(
        input.quality_context.face.active_face_available_mask,
        dtype=np.bool_,
    )
    active_count = int(np.count_nonzero(active))
    return NonManualVisibilityMetrics(
        active_span_face_available_frame_ratio=_ratio(
            int(np.count_nonzero(face_available)),
            active_count,
        ),
        max_active_span_face_unavailable_run_ratio=_ratio(
            _max_run(tuple(bool(value) for value in (active & ~face_available).tolist())),
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


__all__ = ["compute_non_manual_visibility_metrics"]
