"""Final source candidate assembly."""

from __future__ import annotations

from text_to_sign_production.data.gate.sources.types import (
    SourceCandidate,
    SourceMatchResult,
    SourceMatchStatus,
    TranslationSourceRecord,
)


def sample_id_from_translation(translation: TranslationSourceRecord) -> str:
    """Return the physical sample identity for a translation source record."""
    if translation.identity is None:
        return translation.sentence_name
    return translation.identity.keypoint.sample_key.value


def assemble_candidate(match: SourceMatchResult) -> SourceCandidate:
    """Assemble a downstream-facing source candidate from a matched result."""
    if match.status is not SourceMatchStatus.MATCHED:
        detail = (
            match.unmatched_reason.code
            if match.unmatched_reason is not None
            else ",".join(reason.code for reason in match.ambiguity_reasons)
        )
        raise ValueError(f"Cannot assemble candidate from non-matched sources: {detail}.")
    if len(match.video_matches) != 1 or len(match.keypoint_matches) != 1:
        raise ValueError("Matched source result must carry exactly one video and keypoint match.")
    if match.candidate_identity is None:
        raise ValueError("Matched source result must carry explicit candidate identity.")

    translation = match.translation
    video = match.video_matches[0]
    keypoint = match.keypoint_matches[0]

    return SourceCandidate(
        sample_id=match.candidate_identity.keypoint.sample_key.value,
        split=match.split,
        text=translation.text,
        start_time=translation.start_time,
        end_time=translation.end_time,
        video_id=match.candidate_identity.video.video_key.value,
        video_name=video.video_name,
        sentence_id=translation.sentence_id,
        sentence_name=translation.sentence_name,
        video_path=video.path,
        keypoints_dir=keypoint.directory,
        frame_count=keypoint.frame_count,
        video_metadata=video.metadata,
        source_issues=match.source_issues,
        identity=match.candidate_identity,
    )


def assemble_candidates(matches: tuple[SourceMatchResult, ...]) -> tuple[SourceCandidate, ...]:
    """Assemble candidates from matched source results only."""
    return tuple(assemble_candidate(match) for match in matches if match.matched)


__all__ = ["assemble_candidate", "assemble_candidates", "sample_id_from_translation"]
