"""Non-manual detail family metrics."""

from __future__ import annotations

from collections.abc import Iterable

from text_to_sign_production.data.tier.families.types import (
    NonManualQualityMetrics,
    QualityMetricBuildInput,
)


def compute_non_manual_quality_metrics(
    input: QualityMetricBuildInput,
) -> NonManualQualityMetrics:
    """Measure face/non-manual detail using shared face-region context."""
    active = input.quality_context.active_span.active_frame_mask
    frames = input.quality_context.face.frames
    active_frames = [frame for frame in frames if active[frame.frame_index]]
    active_count = len(active_frames)
    face_detail_dropout = tuple(
        active[frame.frame_index]
        and not (frame.upper_face_supported and frame.lower_face_supported)
        for frame in frames
    )
    manual_active_count = sum(1 for frame in active_frames if frame.manual_available)
    return NonManualQualityMetrics(
        active_span_face_landmark_coverage_ratio=_mean(
            frame.whole_face_coverage_fraction for frame in active_frames
        ),
        active_span_upper_face_landmark_coverage_ratio=_mean(
            frame.upper_face_coverage_fraction for frame in active_frames
        ),
        active_span_lower_face_landmark_coverage_ratio=_mean(
            frame.lower_face_coverage_fraction for frame in active_frames
        ),
        max_active_span_face_detail_dropout_run_ratio=_ratio(
            _max_run(face_detail_dropout),
            active_count,
        ),
        active_span_face_available_given_manual_frame_ratio=_ratio(
            sum(
                1
                for frame in active_frames
                if frame.face_available_given_manual_frame
            ),
            manual_active_count,
        ),
    )


def _mean(values: Iterable[float]) -> float:
    sequence = tuple(float(value) for value in values)
    if not sequence:
        return 0.0
    return sum(sequence) / len(sequence)


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


__all__ = ["compute_non_manual_quality_metrics"]
