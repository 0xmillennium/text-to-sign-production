"""Core source-domain contracts for source matching and source-level truth."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

from text_to_sign_production.core.ids import SampleSplit, SourceIssueCode


class SourceAmbiguityCode(enum.StrEnum):
    """Controlled source ambiguity codes."""

    MULTIPLE_VIDEO_MATCHES = "multiple_video_matches"
    MULTIPLE_KEYPOINT_MATCHES = "multiple_keypoint_matches"


class SourceValidationIssueCode(enum.StrEnum):
    """Controlled source validation issue codes."""

    EMPTY_VIDEO_ID = "empty_video_id"
    EMPTY_VIDEO_NAME = "empty_video_name"
    EMPTY_SENTENCE_ID = "empty_sentence_id"
    EMPTY_SENTENCE_NAME = "empty_sentence_name"
    EMPTY_TEXT = "empty_text"
    INVALID_TIME_RANGE = "invalid_time_range"
    INVALID_VIDEO_WIDTH = "invalid_video_width"
    INVALID_VIDEO_HEIGHT = "invalid_video_height"
    INVALID_VIDEO_FPS = "invalid_video_fps"
    INVALID_VIDEO_FRAME_COUNT = "invalid_video_frame_count"
    INVALID_VIDEO_DURATION = "invalid_video_duration"
    EMPTY_KEYPOINT_SAMPLE_ID = "empty_keypoint_sample_id"
    INVALID_KEYPOINT_DIRECTORY = "invalid_keypoint_directory"
    NEGATIVE_KEYPOINT_FRAME_COUNT = "negative_keypoint_frame_count"
    INVALID_VIDEO_PATH = "invalid_video_path"
    INVALID_SPLIT = "invalid_split"
    INVALID_MATCH_STATUS = "invalid_match_status"
    MATCHED_HAS_UNMATCHED_REASON = "matched_has_unmatched_reason"
    MATCHED_HAS_AMBIGUITY = "matched_has_ambiguity"
    MATCHED_CARDINALITY_INVALID = "matched_cardinality_invalid"
    NO_MATCH_MISSING_REASON = "no_match_missing_reason"
    NO_MATCH_HAS_AMBIGUITY = "no_match_has_ambiguity"
    AMBIGUITY_MISSING_REASON = "ambiguity_missing_reason"
    AMBIGUITY_HAS_UNMATCHED_REASON = "ambiguity_has_unmatched_reason"
    CANDIDATE_EMPTY_SAMPLE_ID = "candidate_empty_sample_id"
    CANDIDATE_EMPTY_TEXT = "candidate_empty_text"
    CANDIDATE_INVALID_TIME_RANGE = "candidate_invalid_time_range"
    CANDIDATE_INVALID_FRAME_COUNT = "candidate_invalid_frame_count"
    CANDIDATE_ISSUES_BUT_VIABLE = "candidate_issues_but_viable"
    MISSING_VIDEO_IDENTITY = "missing_video_identity"
    MISSING_KEYPOINT_IDENTITY = "missing_keypoint_identity"
    MISSING_CANDIDATE_IDENTITY = "missing_candidate_identity"
    TRANSLATION_IDENTITY_INCOHERENT = "translation_identity_incoherent"
    VIDEO_IDENTITY_INCOHERENT = "video_identity_incoherent"
    KEYPOINT_IDENTITY_INCOHERENT = "keypoint_identity_incoherent"
    CANDIDATE_IDENTITY_INCOHERENT = "candidate_identity_incoherent"
    DUPLICATE_CANDIDATE_SAMPLE_ID = "duplicate_candidate_sample_id"


class CandidateViabilityStatus(enum.StrEnum):
    """Pose/source structural viability for a matched source candidate."""

    VIABLE = "viable"
    NON_VIABLE = "non_viable"


@dataclass(frozen=True, slots=True)
class SourceIssue:
    """Structured source-side issue."""

    code: SourceIssueCode
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class SourceAmbiguity:
    """Structured source ambiguity."""

    code: SourceAmbiguityCode
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class SourceValidationIssue:
    """A stable source-domain validation issue."""

    code: SourceValidationIssueCode
    message: str


@dataclass(frozen=True, slots=True)
class CandidateViabilityIssue:
    """A structural reason a matched candidate cannot be posed/materialized."""

    code: SourceIssueCode
    message: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateViabilityReport:
    """Viability truth for one matched candidate, separate from identity truth."""

    split: SampleSplit
    sample_id: str
    sentence_id: str
    status: CandidateViabilityStatus
    issues: tuple[CandidateViabilityIssue, ...] = ()

    @property
    def viable(self) -> bool:
        """Whether the candidate can proceed to pose materialization."""
        return self.status is CandidateViabilityStatus.VIABLE


@dataclass(frozen=True, slots=True)
class SourceIdentityKey:
    """Comparable source identity value from an explicit source authority."""

    namespace: str
    value: str


@dataclass(frozen=True, slots=True)
class VideoIdentity:
    """Explicit identity for one physical video source."""

    video_key: SourceIdentityKey


@dataclass(frozen=True, slots=True)
class KeypointIdentity:
    """Explicit identity for one physical keypoint source."""

    sample_key: SourceIdentityKey


@dataclass(frozen=True, slots=True)
class TranslationIdentity:
    """Explicit identity truth carried by a translation source row."""

    video: VideoIdentity
    keypoint: KeypointIdentity
    sentence_key: SourceIdentityKey


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    """Coherent identity binding for a matched source candidate."""

    split: SampleSplit
    translation: TranslationIdentity
    video: VideoIdentity
    keypoint: KeypointIdentity


def source_identity_key(namespace: str, value: str) -> SourceIdentityKey:
    """Build a normalized comparable source identity key."""
    return SourceIdentityKey(namespace=namespace.strip(), value=value.strip())


def video_identity(video_id: str) -> VideoIdentity:
    """Build explicit video identity from source video id truth."""
    return VideoIdentity(video_key=source_identity_key("video_id", video_id))


def keypoint_identity(sample_id: str) -> KeypointIdentity:
    """Build explicit keypoint identity from source sample id truth."""
    return KeypointIdentity(sample_key=source_identity_key("sample_id", sample_id))


def translation_identity(
    video_id: str,
    sentence_id: str,
    sentence_name: str,
) -> TranslationIdentity:
    """Build explicit translation identity from source identity authorities.

    ``sentence_id`` is the semantic sentence grouping key and may repeat across
    physical views. ``sentence_name`` is the row/view-aware physical sample key
    used for payload, manifest, and archive identity.
    """
    sample_key = source_identity_key("sample_id", sentence_name)
    return TranslationIdentity(
        video=video_identity(video_id),
        keypoint=KeypointIdentity(sample_key=sample_key),
        sentence_key=source_identity_key("sentence_id", sentence_id),
    )


@dataclass(frozen=True, slots=True)
class TranslationSourceRecord:
    """Text-bearing translation source truth for one source sentence."""

    video_id: str
    video_name: str
    sentence_id: str
    sentence_name: str
    start_time: float
    end_time: float
    text: str
    identity: TranslationIdentity | None = None

    def __post_init__(self) -> None:
        """Default translation identity to explicit source ids, not source names."""
        if self.identity is None:
            object.__setattr__(
                self,
                "identity",
                translation_identity(self.video_id, self.sentence_id, self.sentence_name),
            )


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Lightweight video-source metadata truth."""

    width: int | None
    height: int | None
    fps: float | None
    frame_count: int | None = None
    duration_seconds: float | None = None
    error: str | None = None

    @property
    def is_readable(self) -> bool:
        """Whether video metadata was successfully read."""
        return self.error is None


@dataclass(frozen=True, slots=True)
class VideoSourceRecord:
    """Video source truth for one physical video asset."""

    path: Path
    video_name: str
    video_id: str | None = None
    identity: VideoIdentity | None = None
    metadata: VideoMetadata = field(
        default_factory=lambda: VideoMetadata(
            width=None,
            height=None,
            fps=None,
            error="video_metadata_not_provided",
        )
    )

    def __post_init__(self) -> None:
        """Default video identity only from explicit source video id truth."""
        if self.identity is None and self.video_id is not None:
            object.__setattr__(self, "identity", video_identity(self.video_id))


@dataclass(frozen=True, slots=True)
class KeypointSourceRecord:
    """Keypoint source truth for one physical keypoint directory."""

    sample_id: str
    directory: Path
    exists: bool
    frame_count: int
    identity: KeypointIdentity | None = None


class SourceMatchStatus(enum.StrEnum):
    """Deterministic source matching outcome."""

    MATCHED = "matched"
    NO_MATCH = "no_match"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class SourceMatchResult:
    """Outcome of matching translation truth to video and keypoint truth."""

    translation: TranslationSourceRecord
    split: SampleSplit
    status: SourceMatchStatus
    video_matches: tuple[VideoSourceRecord, ...] = ()
    keypoint_matches: tuple[KeypointSourceRecord, ...] = ()
    unmatched_reason: SourceIssue | None = None
    ambiguity_reasons: tuple[SourceAmbiguity, ...] = ()
    source_issues: tuple[SourceIssue, ...] = ()
    candidate_identity: CandidateIdentity | None = None

    @property
    def matched(self) -> bool:
        """Whether exactly one video and keypoint source matched."""
        return self.status is SourceMatchStatus.MATCHED

    @property
    def structurally_viable(self) -> bool:
        """Whether matched sources have no source-side structural issues."""
        return self.matched and not self.source_issues


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    """Final downstream-facing source candidate assembled from a match result."""

    sample_id: str
    split: SampleSplit
    text: str
    start_time: float
    end_time: float
    video_id: str
    video_name: str
    sentence_id: str
    sentence_name: str
    video_path: Path
    keypoints_dir: Path
    frame_count: int
    video_metadata: VideoMetadata
    source_issues: tuple[SourceIssue, ...] = ()
    identity: CandidateIdentity | None = None

    @property
    def structurally_viable(self) -> bool:
        """Whether the source candidate has no source-side structural issues."""
        return not self.source_issues


@dataclass(frozen=True, slots=True)
class SourceAvailabilitySummary:
    """Compact availability summary over source matches."""

    match_count: int
    matched_count: int
    ambiguous_count: int
    no_match_count: int
    structurally_viable_count: int
    matched_ratio: float
    structurally_viable_ratio: float


@dataclass(frozen=True, slots=True)
class SourceIssueFrequency:
    """Frequency for a source-side issue code."""

    issue_code: SourceIssueCode
    detail: str | None
    count: int


@dataclass(frozen=True, slots=True)
class SourceUnmatchedReasonFrequency:
    """Frequency for a source no-match reason."""

    reason: SourceIssueCode
    detail: str | None
    count: int


@dataclass(frozen=True, slots=True)
class SourceAmbiguityFrequency:
    """Frequency for a source ambiguity reason."""

    reason: SourceAmbiguityCode
    detail: str | None
    count: int


__all__ = [
    "CandidateIdentity",
    "KeypointIdentity",
    "KeypointSourceRecord",
    "SourceAmbiguity",
    "SourceAmbiguityCode",
    "SourceAmbiguityFrequency",
    "SourceAvailabilitySummary",
    "SourceCandidate",
    "SourceIdentityKey",
    "SourceIssue",
    "SourceIssueCode",
    "SourceIssueFrequency",
    "SourceMatchResult",
    "SourceMatchStatus",
    "SourceUnmatchedReasonFrequency",
    "SourceValidationIssue",
    "SourceValidationIssueCode",
    "TranslationIdentity",
    "TranslationSourceRecord",
    "VideoIdentity",
    "VideoMetadata",
    "VideoSourceRecord",
    "keypoint_identity",
    "source_identity_key",
    "translation_identity",
    "video_identity",
]
