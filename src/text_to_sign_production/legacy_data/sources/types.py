"""Typed models for source-level semantics.

This module defines the domain models that represent:
- translation-row records from the How2Sign dataset,
- video metadata facts,
- keypoint source facts,
- source match results,
- raw sample candidates suitable for downstream pose processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.legacy_data._shared.types import ValidationIssue


@dataclass(frozen=True, slots=True)
class TranslationRow:
    """A single validated row from a How2Sign translation file.

    Represents the canonical fields from a tab-delimited translation CSV
    after structural validation and type conversion.
    """

    video_id: str
    video_name: str
    sentence_id: str
    sentence_name: str
    start_time: float
    end_time: float
    text: str


@dataclass(frozen=True, slots=True)
class VideoMetadataFacts:
    """Lightweight metadata extracted from an MP4 file header.

    All fields may be ``None`` when the video is unreadable or missing.
    The ``error`` field carries a machine-readable error tag when extraction
    fails, allowing typed diagnostics without exceptions.
    """

    width: int | None
    height: int | None
    fps: float | None
    error: str | None = None

    @property
    def is_readable(self) -> bool:
        """Whether metadata was successfully extracted."""
        return self.error is None


@dataclass(frozen=True, slots=True)
class KeypointSourceFacts:
    """Facts about a keypoint directory for one sample.

    Represents source-level availability — whether the directory exists and
    how many frame JSON files it contains — without reading any frame content.
    """

    directory: Path
    exists: bool
    frame_count: int


@dataclass(frozen=True, slots=True)
class SourceMatchResult:
    """The outcome of matching a translation row against keypoint and video sources.

    Represents a single translation row matched (or unmatched) against its
    expected keypoint directory and video file. ``matched`` only means the
    source identity resolved to an available keypoint directory. Structural
    viability is represented separately by ``source_issues``.
    """

    translation: TranslationRow
    split: SampleSplit
    keypoints: KeypointSourceFacts | None
    video_metadata: VideoMetadataFacts | None
    matched: bool
    unmatched_reason: str | None = None
    source_issues: tuple[str, ...] = field(default_factory=tuple)

    @property
    def structurally_viable(self) -> bool:
        """Whether matched sources carry no source-side structural issues."""
        return self.matched and not self.source_issues


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    """A fully assembled source-side sample candidate.

    Carries enough information to locate and interpret a sample for
    downstream pose processing. Only matched sources become candidates; source
    issues retain facts that may still make the candidate structurally
    unusable.

    This is the natural input surface for ``pose`` package operations.
    """

    sample_id: str
    split: SampleSplit
    text: str
    start_time: float
    end_time: float
    video_id: str
    sentence_id: str
    sentence_name: str
    keypoints_dir: Path
    frame_count: int
    video_path: Path
    video_metadata: VideoMetadataFacts
    source_issues: tuple[str, ...] = field(default_factory=tuple)

    @property
    def structurally_viable(self) -> bool:
        """Whether the source candidate is structurally viable before pose parsing."""
        return not self.source_issues


SourceValidationIssue = ValidationIssue


@dataclass(frozen=True, slots=True)
class SourceAvailabilitySummaryRecord:
    """Availability summary for source match results."""

    match_count: int
    matched_count: int
    structurally_viable_count: int
    keypoint_available_count: int
    readable_video_count: int
    matched_ratio: float
    structurally_viable_ratio: float
    readable_video_ratio: float


@dataclass(frozen=True, slots=True)
class SourceIssueFrequencyRecord:
    """Frequency of source-side issue codes."""

    issue_code: str
    count: int


@dataclass(frozen=True, slots=True)
class SourceSplitIssueFrequencyRecord:
    """Split-aware frequency of source-side issue codes."""

    split: SampleSplit
    issue_code: str
    count: int


@dataclass(frozen=True, slots=True)
class SourceUnmatchedReasonCountRecord:
    """Frequency of unmatched source reasons."""

    reason: str
    count: int


@dataclass(frozen=True, slots=True)
class SourceSplitUnmatchedReasonCountRecord:
    """Split-aware frequency of unmatched source reasons."""

    split: SampleSplit
    reason: str
    count: int


@dataclass(frozen=True, slots=True)
class SourceFrameCountDistributionRecord:
    """Distribution of source keypoint frame counts."""

    sample_count: int
    missing_count: int
    minimum: float | None
    p50: float | None
    p95: float | None
    maximum: float | None


@dataclass(frozen=True, slots=True)
class SourceVideoReadabilityRecord:
    """Readable-video coverage summary."""

    source_count: int
    readable_count: int
    unreadable_count: int
    readable_ratio: float
