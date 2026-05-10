"""Upper-body support family metrics."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.data.tier.context.span import UPPER_BODY_TRACKING_LANDMARK_INDICES
from text_to_sign_production.data.tier.families.types import (
    QualityMetricBuildInput,
    UpperBodySupportMetrics,
)


def compute_upper_body_support_metrics(
    input: QualityMetricBuildInput,
) -> UpperBodySupportMetrics:
    """Measure upper-body support over the shared active signing span."""
    active = np.asarray(input.quality_context.active_span.active_frame_mask, dtype=np.bool_)
    conf = input.sample.pose.body_xyc[:, list(UPPER_BODY_TRACKING_LANDMARK_INDICES), 2]
    available = np.any(conf > 0.0, axis=1)
    active_count = int(np.count_nonzero(active))
    total_points = active_count * len(UPPER_BODY_TRACKING_LANDMARK_INDICES)
    return UpperBodySupportMetrics(
        active_span_upper_body_available_frame_ratio=_ratio(
            int(np.count_nonzero(active & available)),
            active_count,
        ),
        active_span_upper_body_landmark_coverage_ratio=_ratio(
            int(np.count_nonzero(conf[active] > 0.0)),
            total_points,
        ),
        max_active_span_upper_body_dropout_run_ratio=_ratio(
            _max_run(tuple(bool(value) for value in (active & ~available).tolist())),
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


__all__ = ["compute_upper_body_support_metrics"]
