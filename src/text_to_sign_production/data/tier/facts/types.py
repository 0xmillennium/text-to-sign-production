"""Factual type system for PreparedSample-derived quality facts."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from text_to_sign_production.core.ids import SampleSplit

CANONICAL_QUALITY_CHANNELS: tuple[str, ...] = (
    "body",
    "left_hand",
    "right_hand",
    "face",
)


class FactsValidationIssueCode(enum.StrEnum):
    """Controlled quality-facts validation issue codes."""

    EMPTY_SAMPLE_ID = "empty_sample_id"
    EMPTY_SOURCE_IDENTITY = "empty_source_identity"
    NEGATIVE_SOURCE_FRAME_COUNT = "negative_source_frame_count"
    INVALID_SOURCE_FPS = "invalid_source_fps"
    NEGATIVE_SOURCE_ISSUE_COUNT = "negative_source_issue_count"
    NEGATIVE_FRAME_COUNT = "negative_frame_count"
    NEGATIVE_VALID_FRAME_COUNT = "negative_valid_frame_count"
    INVALID_VALID_FRAME_RATIO = "invalid_valid_frame_ratio"
    NEGATIVE_TRACKING_COUNT = "negative_tracking_count"
    INVALID_TRACKING_RATIO = "invalid_tracking_ratio"
    INVALID_CHANNEL = "invalid_channel"
    INVALID_COORDINATE_SPACE = "invalid_coordinate_space"
    NEGATIVE_CHANNEL_COUNT = "negative_channel_count"
    INVALID_CHANNEL_RATIO = "invalid_channel_ratio"
    NEGATIVE_TEXT_COUNT = "negative_text_count"
    INVALID_DURATION = "invalid_duration"
    INVALID_INTEGRITY_FLAG = "invalid_integrity_flag"
    CHANNEL_FACT_COUNT_MISMATCH = "channel_fact_count_mismatch"
    FRAME_COUNT_PARTITION_MISMATCH = "frame_count_partition_mismatch"


@dataclass(frozen=True, slots=True)
class FactsValidationIssue:
    """A stable quality-facts validation issue."""

    code: FactsValidationIssueCode
    message: str
    field_path: str | None = None


@dataclass(frozen=True, slots=True)
class SourceQualityFacts:
    """Source-derived factual truth for one prepared sample."""

    sample_id: str
    split: SampleSplit
    source_video_id: str
    source_sentence_id: str
    source_sentence_name: str
    source_duration_seconds: float
    source_frame_count: int
    source_fps: float
    source_issue_count: int


@dataclass(frozen=True, slots=True)
class FrameQualityFacts:
    """Frame-derived factual truth from PreparedSample pose truth."""

    frame_count: int
    valid_frame_count: int
    invalid_frame_count: int
    valid_frame_ratio: float
    invalid_frame_ratio: float


@dataclass(frozen=True, slots=True)
class TrackingQualityFacts:
    """Tracking-derived factual truth from PreparedSample pose truth."""

    selected_frame_count: int
    missing_frame_count: int
    missing_frame_ratio: float
    continuity_break_count: int
    continuity_break_ratio: float
    reanchor_count: int
    reanchor_ratio: float


@dataclass(frozen=True, slots=True)
class ChannelQualityFacts:
    """Channel-level coverage factual truth."""

    channel: str
    coordinate_space: str
    nonzero_frame_count: int
    nonzero_frame_ratio: float
    zero_frame_count: int
    zero_frame_ratio: float


@dataclass(frozen=True, slots=True)
class TextLengthFacts:
    """Text and length factual truth."""

    text_character_count: int
    text_non_whitespace_character_count: int
    text_token_count: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class IntegrityFacts:
    """Cross-layer factual integrity truth."""

    source_pose_frame_count_match: bool
    tensor_frame_shape_valid: bool
    all_required_channels_present: bool
    coordinate_space_normalized: bool


@dataclass(frozen=True, slots=True)
class QualityFacts:
    """Composed quality facts for one prepared sample."""

    source: SourceQualityFacts
    frame: FrameQualityFacts
    tracking: TrackingQualityFacts
    channel: tuple[ChannelQualityFacts, ...]
    text_length: TextLengthFacts
    integrity: IntegrityFacts


@dataclass(frozen=True, slots=True)
class QualityFactsSummary:
    """Compact read-only summary of composed quality facts."""

    sample_id: str
    split: SampleSplit
    frame_count: int
    valid_frame_ratio: float
    missing_tracking_ratio: float
    channel_count: int
    integrity_ok: bool


__all__ = [
    "CANONICAL_QUALITY_CHANNELS",
    "ChannelQualityFacts",
    "FactsValidationIssue",
    "FactsValidationIssueCode",
    "FrameQualityFacts",
    "IntegrityFacts",
    "QualityFacts",
    "QualityFactsSummary",
    "SourceQualityFacts",
    "TextLengthFacts",
    "TrackingQualityFacts",
]
