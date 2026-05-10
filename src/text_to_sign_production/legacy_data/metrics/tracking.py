"""Tracking-quality metrics from persisted frame-quality facts."""

from __future__ import annotations

from text_to_sign_production.legacy_data.metrics.types import TrackingQualityMetrics
from text_to_sign_production.legacy_data.samples.types import PassedManifestEntry


def compute_tracking_quality_metrics(manifest: PassedManifestEntry) -> TrackingQualityMetrics:
    """Compute tracking-quality metrics from persisted sample facts."""
    frame_quality = manifest.frame_quality
    return TrackingQualityMetrics(
        tracked_target_missing_frame_ratio=frame_quality.tracked_target_missing_frame_ratio,
        person_tracking_continuity_break_ratio=(
            frame_quality.person_tracking_continuity_break_ratio
        ),
        person_tracking_reanchor_ratio=frame_quality.person_tracking_reanchor_ratio,
    )
