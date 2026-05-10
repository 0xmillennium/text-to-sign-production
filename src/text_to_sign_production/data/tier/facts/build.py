"""Assembly of quality facts from PreparedSample truth."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.core.models import PreparedSample
from text_to_sign_production.data.tier.facts.types import (
    CANONICAL_QUALITY_CHANNELS,
    ChannelQualityFacts,
    FrameQualityFacts,
    IntegrityFacts,
    QualityFacts,
    SourceQualityFacts,
    TextLengthFacts,
    TrackingQualityFacts,
)


def build_quality_facts(sample: PreparedSample) -> QualityFacts:
    """Build composed policy-agnostic quality facts from PreparedSample."""
    frame_count = sample.pose.frame_count
    valid_frame_count = int(np.count_nonzero(sample.pose.valid_frame_mask))
    invalid_frame_count = frame_count - valid_frame_count
    source_duration_seconds = frame_count / sample.source.fps
    source_facts = SourceQualityFacts(
        sample_id=sample.source.sample_id,
        split=sample.source.split,
        source_video_id=sample.source.source_video_id,
        source_sentence_id=sample.source.source_sentence_id,
        source_sentence_name=sample.source.source_sentence_name,
        source_duration_seconds=source_duration_seconds,
        source_frame_count=frame_count,
        source_fps=sample.source.fps,
        source_issue_count=len(sample.source.source_issue_codes),
    )
    frame_facts = FrameQualityFacts(
        frame_count=frame_count,
        valid_frame_count=valid_frame_count,
        invalid_frame_count=invalid_frame_count,
        valid_frame_ratio=_ratio(valid_frame_count, frame_count),
        invalid_frame_ratio=_ratio(invalid_frame_count, frame_count),
    )
    tracking_facts = TrackingQualityFacts(
        selected_frame_count=frame_count - sample.pose.tracked_target_missing_frame_count,
        missing_frame_count=sample.pose.tracked_target_missing_frame_count,
        missing_frame_ratio=_ratio(sample.pose.tracked_target_missing_frame_count, frame_count),
        continuity_break_count=sample.pose.continuity_break_count,
        continuity_break_ratio=_ratio(sample.pose.continuity_break_count, frame_count),
        reanchor_count=sample.pose.reanchor_count,
        reanchor_ratio=_ratio(sample.pose.reanchor_count, frame_count),
    )
    channel_facts = _build_channel_quality_facts(sample)
    text = sample.source.canonical_normalized_text
    text_facts = TextLengthFacts(
        text_character_count=len(text),
        text_non_whitespace_character_count=sum(1 for char in text if not char.isspace()),
        text_token_count=len(text.split()),
        duration_seconds=source_duration_seconds,
    )
    integrity_facts = IntegrityFacts(
        source_pose_frame_count_match=True,
        tensor_frame_shape_valid=all(
            tensor.shape[0] == frame_count and tensor.ndim == 3 and tensor.shape[-1] == 3
            for tensor in (
                sample.pose.body_xyc,
                sample.pose.left_hand_xyc,
                sample.pose.right_hand_xyc,
                sample.pose.face_xyc,
            )
        ),
        all_required_channels_present=tuple(fact.channel for fact in channel_facts)
        == CANONICAL_QUALITY_CHANNELS,
        coordinate_space_normalized=sample.pose.coordinate_space.value == "normalized_image",
        canonical_normalized_text_present=bool(sample.source.canonical_normalized_text),
    )
    return QualityFacts(
        source=source_facts,
        frame=frame_facts,
        tracking=tracking_facts,
        channel=channel_facts,
        text_length=text_facts,
        integrity=integrity_facts,
    )


def _build_channel_quality_facts(sample: PreparedSample) -> tuple[ChannelQualityFacts, ...]:
    frame_count = sample.pose.frame_count
    counts = {
        "body": sample.pose.body_nonzero_frame_count,
        "left_hand": sample.pose.left_hand_nonzero_frame_count,
        "right_hand": sample.pose.right_hand_nonzero_frame_count,
        "face": sample.pose.face_nonzero_frame_count,
    }
    return tuple(
        ChannelQualityFacts(
            channel=channel,
            coordinate_space=sample.pose.coordinate_space.value,
            nonzero_frame_count=counts[channel],
            nonzero_frame_ratio=_ratio(counts[channel], frame_count),
            zero_frame_count=frame_count - counts[channel],
            zero_frame_ratio=_ratio(frame_count - counts[channel], frame_count),
        )
        for channel in CANONICAL_QUALITY_CHANNELS
    )


def _ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


__all__ = ["build_quality_facts"]
