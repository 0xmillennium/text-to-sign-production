"""Tracking quality family metrics."""

from __future__ import annotations

from text_to_sign_production.data.tier.families.types import (
    QualityMetricBuildInput,
    TrackingQualityMetrics,
)


def compute_tracking_quality_metrics(input: QualityMetricBuildInput) -> TrackingQualityMetrics:
    """Measure tracking stability from persisted tracking facts."""
    tracking = input.quality_facts.tracking
    return TrackingQualityMetrics(
        tracked_target_missing_frame_ratio=tracking.missing_frame_ratio,
        person_tracking_continuity_break_ratio=tracking.continuity_break_ratio,
        person_tracking_reanchor_ratio=tracking.reanchor_ratio,
    )


__all__ = ["compute_tracking_quality_metrics"]
