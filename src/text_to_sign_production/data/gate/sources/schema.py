"""Structural validation for source-domain truths."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.data.gate.sources.types import (
    KeypointIdentity,
    KeypointSourceRecord,
    SourceCandidate,
    SourceIdentityKey,
    SourceMatchResult,
    SourceMatchStatus,
    SourceValidationIssue,
    SourceValidationIssueCode,
    TranslationSourceRecord,
    VideoIdentity,
    VideoMetadata,
    VideoSourceRecord,
)


def _has_text(value: str) -> bool:
    return bool(value.strip())


def _path_has_name(value: Path) -> bool:
    return bool(str(value).strip()) and bool(value.name)


def _issue(code: SourceValidationIssueCode, message: str) -> SourceValidationIssue:
    return SourceValidationIssue(code=code, message=message)


def _validate_identity_key(
    key: SourceIdentityKey,
    *,
    field_name: str,
) -> tuple[SourceValidationIssue, ...]:
    issues: list[SourceValidationIssue] = []
    if not _has_text(key.namespace):
        issues.append(
            _issue(
                SourceValidationIssueCode.CANDIDATE_IDENTITY_INCOHERENT,
                f"{field_name} identity namespace must be non-empty.",
            )
        )
    if not _has_text(key.value):
        issues.append(
            _issue(
                SourceValidationIssueCode.CANDIDATE_IDENTITY_INCOHERENT,
                f"{field_name} identity value must be non-empty.",
            )
        )
    return tuple(issues)


def _validate_video_identity(
    identity: VideoIdentity,
    *,
    field_name: str,
) -> tuple[SourceValidationIssue, ...]:
    return _validate_identity_key(identity.video_key, field_name=field_name)


def _validate_keypoint_identity(
    identity: KeypointIdentity,
    *,
    field_name: str,
) -> tuple[SourceValidationIssue, ...]:
    return _validate_identity_key(identity.sample_key, field_name=field_name)


def _validate_translation_identity(
    record: TranslationSourceRecord,
) -> tuple[SourceValidationIssue, ...]:
    issues: list[SourceValidationIssue] = []
    if record.identity is None:
        issues.append(
            _issue(
                SourceValidationIssueCode.TRANSLATION_IDENTITY_INCOHERENT,
                "Translation record must carry explicit identity.",
            )
        )
        return tuple(issues)
    issues.extend(_validate_video_identity(record.identity.video, field_name="Translation video"))
    issues.extend(
        _validate_keypoint_identity(
            record.identity.keypoint,
            field_name="Translation keypoint",
        )
    )
    issues.extend(
        _validate_identity_key(record.identity.sentence_key, field_name="Translation sentence")
    )
    if record.identity.video.video_key.value != record.video_id:
        issues.append(
            _issue(
                SourceValidationIssueCode.TRANSLATION_IDENTITY_INCOHERENT,
                "Translation identity video key must match source video_id.",
            )
        )
    if record.identity.sentence_key.value != record.sentence_id:
        issues.append(
            _issue(
                SourceValidationIssueCode.TRANSLATION_IDENTITY_INCOHERENT,
                "Translation identity sentence key must match source sentence_id.",
            )
        )
    if record.identity.keypoint.sample_key.value != record.sentence_name:
        issues.append(
            _issue(
                SourceValidationIssueCode.TRANSLATION_IDENTITY_INCOHERENT,
                "Translation identity sample key must match source sentence_name.",
            )
        )
    return tuple(issues)


def _validate_candidate_identity(
    candidate: SourceCandidate,
) -> tuple[SourceValidationIssue, ...]:
    issues: list[SourceValidationIssue] = []
    if candidate.identity is None:
        issues.append(
            _issue(
                SourceValidationIssueCode.MISSING_CANDIDATE_IDENTITY,
                "Candidate must carry explicit matched source identity.",
            )
        )
        return tuple(issues)
    issues.extend(_validate_video_identity(candidate.identity.video, field_name="Candidate video"))
    issues.extend(
        _validate_keypoint_identity(
            candidate.identity.keypoint,
            field_name="Candidate keypoint",
        )
    )
    if candidate.identity.split is not candidate.split:
        issues.append(
            _issue(
                SourceValidationIssueCode.CANDIDATE_IDENTITY_INCOHERENT,
                "Candidate identity split must match candidate split.",
            )
        )
    if candidate.identity.translation.video != candidate.identity.video:
        issues.append(
            _issue(
                SourceValidationIssueCode.CANDIDATE_IDENTITY_INCOHERENT,
                "Candidate translation and video identities must agree.",
            )
        )
    if candidate.identity.translation.keypoint != candidate.identity.keypoint:
        issues.append(
            _issue(
                SourceValidationIssueCode.CANDIDATE_IDENTITY_INCOHERENT,
                "Candidate translation and keypoint identities must agree.",
            )
        )
    if candidate.identity.video.video_key.value != candidate.video_id:
        issues.append(
            _issue(
                SourceValidationIssueCode.CANDIDATE_IDENTITY_INCOHERENT,
                "Candidate identity video key must match candidate video_id.",
            )
        )
    if candidate.identity.translation.sentence_key.value != candidate.sentence_id:
        issues.append(
            _issue(
                SourceValidationIssueCode.CANDIDATE_IDENTITY_INCOHERENT,
                "Candidate identity sentence key must match candidate sentence_id.",
            )
        )
    if candidate.identity.keypoint.sample_key.value != candidate.sample_id:
        issues.append(
            _issue(
                SourceValidationIssueCode.CANDIDATE_IDENTITY_INCOHERENT,
                "Candidate identity keypoint sample key must match candidate sample_id.",
            )
        )
    return tuple(issues)


_CANDIDATE_EMPTY_CODES: dict[str, SourceValidationIssueCode] = {
    "sample_id": SourceValidationIssueCode.CANDIDATE_EMPTY_SAMPLE_ID,
    "text": SourceValidationIssueCode.CANDIDATE_EMPTY_TEXT,
    "video_id": SourceValidationIssueCode.EMPTY_VIDEO_ID,
    "video_name": SourceValidationIssueCode.EMPTY_VIDEO_NAME,
    "sentence_id": SourceValidationIssueCode.EMPTY_SENTENCE_ID,
    "sentence_name": SourceValidationIssueCode.EMPTY_SENTENCE_NAME,
}


def validate_translation_record(
    record: TranslationSourceRecord,
) -> tuple[SourceValidationIssue, ...]:
    """Validate the structural shape of translation source truth."""
    issues: list[SourceValidationIssue] = []
    for field_name in ("video_id", "video_name", "sentence_id", "sentence_name", "text"):
        if not _has_text(getattr(record, field_name)):
            issues.append(
                _issue(
                    SourceValidationIssueCode(f"empty_{field_name}"),
                    f"Translation {field_name} is empty.",
                )
            )
    if record.start_time < 0 or record.end_time < 0 or record.start_time >= record.end_time:
        issues.append(
            _issue(
                SourceValidationIssueCode.INVALID_TIME_RANGE,
                "Translation timestamps must be ordered.",
            )
        )
    issues.extend(_validate_translation_identity(record))
    return tuple(issues)


def validate_video_metadata(metadata: VideoMetadata) -> tuple[SourceValidationIssue, ...]:
    """Validate the structural shape of video metadata truth."""
    issues: list[SourceValidationIssue] = []
    if metadata.width is not None and metadata.width <= 0:
        issues.append(
            _issue(
                SourceValidationIssueCode.INVALID_VIDEO_WIDTH,
                "Video width must be positive when present.",
            )
        )
    if metadata.height is not None and metadata.height <= 0:
        issues.append(
            _issue(
                SourceValidationIssueCode.INVALID_VIDEO_HEIGHT,
                "Video height must be positive when present.",
            )
        )
    if metadata.fps is not None and metadata.fps <= 0:
        issues.append(
            _issue(
                SourceValidationIssueCode.INVALID_VIDEO_FPS,
                "Video fps must be positive when present.",
            )
        )
    if metadata.frame_count is not None and metadata.frame_count <= 0:
        issues.append(
            _issue(
                SourceValidationIssueCode.INVALID_VIDEO_FRAME_COUNT,
                "Video frame_count must be positive when present.",
            )
        )
    if metadata.duration_seconds is not None and metadata.duration_seconds <= 0:
        issues.append(
            _issue(
                SourceValidationIssueCode.INVALID_VIDEO_DURATION,
                "Video duration_seconds must be positive when present.",
            )
        )
    return tuple(issues)


def validate_video_record(record: VideoSourceRecord) -> tuple[SourceValidationIssue, ...]:
    """Validate the structural shape of video source truth."""
    issues: list[SourceValidationIssue] = []
    if record.video_id is not None and not _has_text(record.video_id):
        issues.append(
            _issue(
                SourceValidationIssueCode.EMPTY_VIDEO_ID,
                "Video video_id must be non-empty when present.",
            )
        )
    if record.identity is None:
        issues.append(
            _issue(
                SourceValidationIssueCode.MISSING_VIDEO_IDENTITY,
                "Video record must carry explicit video identity to participate in matching.",
            )
        )
    else:
        issues.extend(_validate_video_identity(record.identity, field_name="Video source"))
        if record.video_id is not None and record.identity.video_key.value != record.video_id:
            issues.append(
                _issue(
                    SourceValidationIssueCode.VIDEO_IDENTITY_INCOHERENT,
                    "Video identity key must match source video_id.",
                )
            )
    if not _has_text(record.video_name):
        issues.append(
            _issue(
                SourceValidationIssueCode.EMPTY_VIDEO_NAME, "Video video_name must be non-empty."
            )
        )
    if not _path_has_name(record.path):
        issues.append(
            _issue(
                SourceValidationIssueCode.INVALID_VIDEO_PATH, "Video path must have a file name."
            )
        )
    issues.extend(validate_video_metadata(record.metadata))
    return tuple(issues)


def validate_keypoint_record(record: KeypointSourceRecord) -> tuple[SourceValidationIssue, ...]:
    """Validate the structural shape of keypoint source truth."""
    issues: list[SourceValidationIssue] = []
    if not _has_text(record.sample_id):
        issues.append(
            _issue(
                SourceValidationIssueCode.EMPTY_KEYPOINT_SAMPLE_ID,
                "Keypoint sample_id must be non-empty.",
            )
        )
    if not _path_has_name(record.directory):
        issues.append(
            _issue(
                SourceValidationIssueCode.INVALID_KEYPOINT_DIRECTORY,
                "Keypoint directory must have a final name.",
            )
        )
    if record.frame_count < 0:
        issues.append(
            _issue(
                SourceValidationIssueCode.NEGATIVE_KEYPOINT_FRAME_COUNT,
                "Keypoint frame_count cannot be negative.",
            )
        )
    if record.identity is None:
        issues.append(
            _issue(
                SourceValidationIssueCode.MISSING_KEYPOINT_IDENTITY,
                "Keypoint record must carry explicit identity to participate in matching.",
            )
        )
    else:
        issues.extend(
            _validate_keypoint_identity(
                record.identity,
                field_name="Keypoint source",
            )
        )
        if record.identity.sample_key.value != record.sample_id:
            issues.append(
                _issue(
                    SourceValidationIssueCode.KEYPOINT_IDENTITY_INCOHERENT,
                    "Keypoint identity sample key must match source sample_id.",
                )
            )
    return tuple(issues)


def validate_match_result(match: SourceMatchResult) -> tuple[SourceValidationIssue, ...]:
    """Validate the structural shape of a source match result."""
    issues: list[SourceValidationIssue] = list(validate_translation_record(match.translation))
    if not isinstance(match.split, SampleSplit):
        issues.append(
            _issue(
                SourceValidationIssueCode.INVALID_SPLIT, f"Invalid source split: {match.split!r}."
            )
        )
    if not isinstance(match.status, SourceMatchStatus):
        issues.append(
            _issue(
                SourceValidationIssueCode.INVALID_MATCH_STATUS,
                f"Invalid match status: {match.status!r}.",
            )
        )
    for video in match.video_matches:
        issues.extend(validate_video_record(video))
    for keypoint in match.keypoint_matches:
        issues.extend(validate_keypoint_record(keypoint))
    return tuple(issues)


def validate_candidate(candidate: SourceCandidate) -> tuple[SourceValidationIssue, ...]:
    """Validate the structural shape of a final source candidate."""
    issues: list[SourceValidationIssue] = []
    for field_name in (
        "sample_id",
        "text",
        "video_id",
        "video_name",
        "sentence_id",
        "sentence_name",
    ):
        if not _has_text(getattr(candidate, field_name)):
            issues.append(
                _issue(_CANDIDATE_EMPTY_CODES[field_name], f"Candidate {field_name} is empty.")
            )
    if not isinstance(candidate.split, SampleSplit):
        issues.append(
            _issue(
                SourceValidationIssueCode.INVALID_SPLIT,
                f"Invalid source split: {candidate.split!r}.",
            )
        )
    if (
        candidate.start_time < 0
        or candidate.end_time < 0
        or candidate.start_time >= candidate.end_time
    ):
        issues.append(
            _issue(
                SourceValidationIssueCode.CANDIDATE_INVALID_TIME_RANGE,
                "Candidate timestamps must be ordered.",
            )
        )
    if not _path_has_name(candidate.video_path):
        issues.append(
            _issue(
                SourceValidationIssueCode.INVALID_VIDEO_PATH,
                "Candidate video_path must have a file name.",
            )
        )
    if not _path_has_name(candidate.keypoints_dir):
        issues.append(
            _issue(
                SourceValidationIssueCode.INVALID_KEYPOINT_DIRECTORY,
                "Candidate keypoints_dir must have a final name.",
            )
        )
    if candidate.frame_count <= 0:
        issues.append(
            _issue(
                SourceValidationIssueCode.CANDIDATE_INVALID_FRAME_COUNT,
                "Candidate frame_count must be positive.",
            )
        )
    issues.extend(validate_video_metadata(candidate.video_metadata))
    issues.extend(_validate_candidate_identity(candidate))
    return tuple(issues)


__all__ = [
    "validate_candidate",
    "validate_keypoint_record",
    "validate_match_result",
    "validate_translation_record",
    "validate_video_metadata",
    "validate_video_record",
]
