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

from text_to_sign_production.data._shared.identities import SampleSplit


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


@dataclass(frozen=True, slots=True)
class SourceValidationIssue:
    """A specific issue found during source candidate validation."""

    code: str
    message: str
