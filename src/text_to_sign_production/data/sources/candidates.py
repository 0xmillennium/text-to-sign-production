"""Assembly of typed raw sample candidates."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.data.sources.types import (
    SourceCandidate,
    SourceMatchResult,
    TranslationRow,
    VideoMetadataFacts,
)


def sample_id_from_translation(translation: TranslationRow) -> str:
    """Return the canonical sample identity for a translation row."""
    return translation.sentence_name


def assemble_candidate(match: SourceMatchResult, video_path: Path) -> SourceCandidate:
    """Convert a matched source result into a canonical candidate.

    A candidate requires a successful match. If the match failed, this
    function will raise a ValueError.
    """
    if not match.matched or match.keypoints is None:
        raise ValueError(
            f"Cannot assemble candidate from unmatched sources: {match.unmatched_reason}"
        )

    translation = match.translation

    source_issues: list[str] = list(match.source_issues)

    video_metadata = match.video_metadata
    if video_metadata is None:
        # Create an empty video metadata fact with a generic error
        video_metadata = VideoMetadataFacts(
            width=None, height=None, fps=None, error="video_metadata_not_provided"
        )

    return SourceCandidate(
        sample_id=sample_id_from_translation(translation),
        split=match.split,
        text=translation.text,
        start_time=translation.start_time,
        end_time=translation.end_time,
        video_id=translation.video_id,
        sentence_id=translation.sentence_id,
        sentence_name=translation.sentence_name,
        keypoints_dir=match.keypoints.directory,
        frame_count=match.keypoints.frame_count,
        video_path=video_path,
        video_metadata=video_metadata,
        source_issues=tuple(source_issues),
    )
