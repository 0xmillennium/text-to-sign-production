"""Deterministic matching between source records."""

from __future__ import annotations

from collections.abc import Sequence

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.data.gate.sources.types import (
    CandidateIdentity,
    KeypointSourceRecord,
    SourceAmbiguity,
    SourceAmbiguityCode,
    SourceIssue,
    SourceIssueCode,
    SourceMatchResult,
    SourceMatchStatus,
    TranslationSourceRecord,
    VideoSourceRecord,
)


def _sort_videos(videos: Sequence[VideoSourceRecord]) -> tuple[VideoSourceRecord, ...]:
    return tuple(
        sorted(
            videos,
            key=lambda video: (
                video.identity.video_key.value if video.identity is not None else "",
                video.video_name,
                video.path.as_posix(),
            ),
        )
    )


def _sort_keypoints(
    keypoints: Sequence[KeypointSourceRecord],
) -> tuple[KeypointSourceRecord, ...]:
    return tuple(
        sorted(
            keypoints,
            key=lambda source: (
                source.identity.sample_key.value if source.identity is not None else "",
                source.sample_id,
                source.directory.as_posix(),
            ),
        )
    )


def find_video_matches(
    translation: TranslationSourceRecord,
    videos: Sequence[VideoSourceRecord],
) -> tuple[VideoSourceRecord, ...]:
    """Find video records matching a translation by exact source identity."""
    if translation.identity is None:
        return ()
    matches = [
        video
        for video in videos
        if video.identity is not None and video.identity == translation.identity.video
    ]
    return _sort_videos(matches)


def find_keypoint_matches(
    translation: TranslationSourceRecord,
    keypoints: Sequence[KeypointSourceRecord],
) -> tuple[KeypointSourceRecord, ...]:
    """Find keypoint records matching a translation by exact sample identity."""
    if translation.identity is None:
        return ()
    matches = [
        source
        for source in keypoints
        if source.identity is not None and source.identity == translation.identity.keypoint
    ]
    return _sort_keypoints(matches)


def match_sources(
    *,
    translation: TranslationSourceRecord,
    split: SampleSplit | str,
    videos: Sequence[VideoSourceRecord],
    keypoints: Sequence[KeypointSourceRecord],
) -> SourceMatchResult:
    """Match translation truth to video and keypoint truth deterministically."""
    video_matches = find_video_matches(translation, videos)
    keypoint_matches = find_keypoint_matches(translation, keypoints)

    ambiguity_reasons: list[SourceAmbiguity] = []
    source_issues: list[SourceIssue] = []
    unmatched_reasons: list[SourceIssue] = []

    if len(video_matches) > 1:
        ambiguity_reasons.append(SourceAmbiguity(SourceAmbiguityCode.MULTIPLE_VIDEO_MATCHES))
    elif not video_matches:
        unmatched_reasons.append(SourceIssue(SourceIssueCode.MISSING_VIDEO_SOURCE))

    if len(keypoint_matches) > 1:
        ambiguity_reasons.append(SourceAmbiguity(SourceAmbiguityCode.MULTIPLE_KEYPOINT_MATCHES))
    elif not keypoint_matches:
        unmatched_reasons.append(SourceIssue(SourceIssueCode.MISSING_KEYPOINT_SOURCE))

    status: SourceMatchStatus
    unmatched_reason: SourceIssue | None
    if ambiguity_reasons:
        status = SourceMatchStatus.AMBIGUOUS
        unmatched_reason = None
    elif unmatched_reasons:
        status = SourceMatchStatus.NO_MATCH
        unmatched_reason = unmatched_reasons[0]
    else:
        status = SourceMatchStatus.MATCHED
        unmatched_reason = None

    candidate_identity: CandidateIdentity | None = None
    if (
        status is SourceMatchStatus.MATCHED
        and translation.identity is not None
        and video_matches[0].identity is not None
        and keypoint_matches[0].identity is not None
    ):
        candidate_identity = CandidateIdentity(
            split=SampleSplit(split),
            translation=translation.identity,
            video=video_matches[0].identity,
            keypoint=keypoint_matches[0].identity,
        )

    if len(video_matches) == 1 and not video_matches[0].metadata.is_readable:
        error = video_matches[0].metadata.error
        issue_code = (
            SourceIssueCode.VIDEO_METADATA_NOT_PROVIDED
            if error == SourceIssueCode.VIDEO_METADATA_NOT_PROVIDED.value
            else SourceIssueCode.VIDEO_METADATA_UNREADABLE
        )
        source_issues.append(SourceIssue(issue_code, detail=error))
    if len(keypoint_matches) == 1:
        keypoint = keypoint_matches[0]
        if not keypoint.exists:
            source_issues.append(SourceIssue(SourceIssueCode.MISSING_KEYPOINT_DIRECTORY))
        elif keypoint.frame_count <= 0:
            source_issues.append(SourceIssue(SourceIssueCode.MISSING_FRAME_JSON_FILES))

    return SourceMatchResult(
        translation=translation,
        split=SampleSplit(split),
        status=status,
        video_matches=video_matches,
        keypoint_matches=keypoint_matches,
        unmatched_reason=unmatched_reason,
        ambiguity_reasons=tuple(ambiguity_reasons),
        source_issues=tuple(source_issues),
        candidate_identity=candidate_identity,
    )


__all__ = [
    "find_keypoint_matches",
    "find_video_matches",
    "match_sources",
]
