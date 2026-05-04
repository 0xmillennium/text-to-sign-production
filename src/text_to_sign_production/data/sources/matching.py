"""Source matching between translation rows, keypoint sources, and video metadata."""

from __future__ import annotations

from text_to_sign_production.data.sources.types import (
    KeypointSourceFacts,
    SourceMatchResult,
    TranslationRow,
    VideoMetadataFacts,
)


def match_sources(
    *,
    translation: TranslationRow,
    split: str,
    keypoints: KeypointSourceFacts | None,
    video_metadata: VideoMetadataFacts | None,
) -> SourceMatchResult:
    """Match a translation row against its resolved physical sources.

    A match requires the keypoint directory to exist.
    """
    matched = True
    unmatched_reason = None

    if keypoints is None or not keypoints.exists:
        matched = False
        unmatched_reason = "missing_keypoint_directory"

    return SourceMatchResult(
        translation=translation,
        split=split,
        keypoints=keypoints,
        video_metadata=video_metadata,
        matched=matched,
        unmatched_reason=unmatched_reason,
    )
