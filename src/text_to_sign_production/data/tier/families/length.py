"""Length diagnostic family metrics."""

from __future__ import annotations

from text_to_sign_production.data.tier.families.types import (
    LengthDiagnosticMetrics,
    QualityMetricBuildInput,
)


def compute_length_diagnostic_metrics(input: QualityMetricBuildInput) -> LengthDiagnosticMetrics:
    """Compute length-side diagnostics from persisted facts."""
    frame_count = input.quality_facts.frame.frame_count
    duration = input.quality_facts.text_length.duration_seconds
    return LengthDiagnosticMetrics(
        frame_count=frame_count,
        duration_seconds=duration,
        frames_per_second_observed=_rate(frame_count, duration),
    )


def _rate(frame_count: int, duration_seconds: float) -> float | None:
    if duration_seconds <= 0.0:
        return None
    return frame_count / duration_seconds


__all__ = ["compute_length_diagnostic_metrics"]
