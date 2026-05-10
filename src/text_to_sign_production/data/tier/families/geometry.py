"""Geometry consistency family metrics."""

from __future__ import annotations

from text_to_sign_production.data.tier.families.types import (
    GeometryMetrics,
    QualityMetricBuildInput,
)

_SEGMENT_RELATIVE_DEVIATION_LIMIT = 0.55
_CROSS_CHANNEL_RELATIVE_DEVIATION_LIMIT = 0.75
_MIN_REFERENCE_LENGTH = 1e-12


def compute_geometry_metrics(input: QualityMetricBuildInput) -> GeometryMetrics:
    """Measure sample-relative geometry consistency from reference context."""
    active = input.quality_context.active_span.active_frame_mask
    geometry = input.quality_context.geometry
    active_count = sum(1 for value in active if value)
    upper_outlier_count = 0
    hand_outlier_count = 0
    scale_outlier_count = 0

    for frame in geometry.frames:
        if not active[frame.frame_index]:
            continue
        if _frame_has_segment_outlier(
            frame.upper_body_segment_lengths,
            geometry.upper_body_segment_references,
            _SEGMENT_RELATIVE_DEVIATION_LIMIT,
        ):
            upper_outlier_count += 1
        if _frame_has_segment_outlier(
            frame.representative_hand_segment_lengths,
            geometry.representative_hand_segment_references,
            _SEGMENT_RELATIVE_DEVIATION_LIMIT,
        ):
            hand_outlier_count += 1
        if _cross_channel_scale_outlier(
            frame.body_scale_reference_length,
            frame.hand_scale_reference_length,
            geometry.cross_channel_scale_reference,
        ):
            scale_outlier_count += 1

    return GeometryMetrics(
        active_span_upper_body_bone_length_outlier_frame_ratio=_ratio(
            upper_outlier_count,
            active_count,
        ),
        active_span_representative_hand_bone_length_outlier_frame_ratio=_ratio(
            hand_outlier_count,
            active_count,
        ),
        active_span_cross_channel_scale_outlier_frame_ratio=_ratio(
            scale_outlier_count,
            active_count,
        ),
    )


def _frame_has_segment_outlier(
    observed: dict[tuple[int, int], float | None],
    references: dict[tuple[int, int], float | None],
    relative_deviation_limit: float,
) -> bool:
    for segment, value in observed.items():
        reference = references.get(segment)
        if value is None or reference is None or reference <= _MIN_REFERENCE_LENGTH:
            continue
        if abs(value - reference) / reference > relative_deviation_limit:
            return True
    return False


def _cross_channel_scale_outlier(
    body_scale: float | None,
    hand_scale: float | None,
    reference: float | None,
) -> bool:
    if (
        body_scale is None
        or hand_scale is None
        or reference is None
        or body_scale <= _MIN_REFERENCE_LENGTH
        or reference <= _MIN_REFERENCE_LENGTH
    ):
        return False
    observed = hand_scale / body_scale
    return abs(observed - reference) / reference > _CROSS_CHANNEL_RELATIVE_DEVIATION_LIMIT


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


__all__ = ["compute_geometry_metrics"]
