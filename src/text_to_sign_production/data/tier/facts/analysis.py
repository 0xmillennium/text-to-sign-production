"""Read-only analysis helpers for quality facts."""

from __future__ import annotations

from text_to_sign_production.data.tier.facts.types import QualityFacts, QualityFactsSummary


def summarize_quality_facts(facts: QualityFacts) -> QualityFactsSummary:
    """Summarize composed quality facts."""
    integrity_ok = (
        facts.integrity.source_pose_frame_count_match
        and facts.integrity.tensor_frame_shape_valid
        and facts.integrity.all_required_channels_present
        and facts.integrity.coordinate_space_normalized
    )
    return QualityFactsSummary(
        sample_id=facts.source.sample_id,
        split=facts.source.split,
        frame_count=facts.frame.frame_count,
        valid_frame_ratio=facts.frame.valid_frame_ratio,
        missing_tracking_ratio=facts.tracking.missing_frame_ratio,
        channel_count=len(facts.channel),
        integrity_ok=integrity_ok,
    )


__all__ = ["summarize_quality_facts"]
