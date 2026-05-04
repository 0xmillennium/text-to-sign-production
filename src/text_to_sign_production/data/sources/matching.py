"""Source matching between translation rows, keypoint sources, and video metadata."""

from __future__ import annotations

from text_to_sign_production.data._shared.identities import SampleSplit
from text_to_sign_production.data.sources.types import (
    KeypointSourceFacts,
    SourceMatchResult,
    TranslationRow,
    VideoMetadataFacts,
)


def match_sources(
    *,
    translation: TranslationRow,
    split: SampleSplit | str,
    keypoints: KeypointSourceFacts | None,
    video_metadata: VideoMetadataFacts | None,
) -> SourceMatchResult:
    """Match a translation row against its resolved physical sources.

    A match requires the keypoint directory to exist. Structural viability is
    reported separately from identity/source matching.
    """
    matched = True
    unmatched_reason = None
    source_issues: list[str] = []

    if keypoints is None or not keypoints.exists:
        matched = False
        unmatched_reason = "missing_keypoint_directory"
        source_issues.append("missing_keypoint_directory")
    elif keypoints.frame_count == 0:
        source_issues.append("missing_frame_json_files")

    if video_metadata is None:
        source_issues.append("video_metadata_not_provided")
    elif not video_metadata.is_readable:
        source_issues.append(f"video_metadata:{video_metadata.error}")

    return SourceMatchResult(
        translation=translation,
        split=SampleSplit(split),
        keypoints=keypoints,
        video_metadata=video_metadata,
        matched=matched,
        unmatched_reason=unmatched_reason,
        source_issues=tuple(source_issues),
    )
