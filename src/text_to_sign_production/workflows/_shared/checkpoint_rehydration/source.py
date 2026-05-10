"""Source truth rehydration from checkpoint authority."""

from __future__ import annotations

from text_to_sign_production.data.tier._shared.types import QualitySourceTruth
from text_to_sign_production.workflows._shared.checkpoint_rehydration.types import (
    CheckpointRehydrationInput,
)


def build_quality_source_truth(
    input: CheckpointRehydrationInput,
) -> QualitySourceTruth:
    """Build authoritative downstream source truth from checkpoint truth.

    Consumes checkpoint truth only. Carries canonical normalized text
    from checkpoint authority. Does NOT invent canonical text.
    Does NOT rerun source matching.
    """
    manifest = input.passed_manifest
    source = input.payload.source

    return QualitySourceTruth(
        sample_id=manifest.sample_id,
        split=manifest.split,
        text=manifest.text,
        canonical_normalized_text=manifest.canonical_normalized_text,
        video_id=manifest.source_video_id,
        video_name=manifest.source_sentence_name,
        sentence_id=manifest.source_sentence_id,
        sentence_name=manifest.source_sentence_name,
        start_time=source.start_time,
        end_time=source.end_time,
        source_frame_count=source.source_frame_count,
        fps=manifest.fps,
        video_readable=source.video_readable,
        source_issue_count=len(source.source_issue_codes),
        source_issue_codes=tuple(source.source_issue_codes),
        identity_present=source.identity is not None,
        identity_label=_build_identity_label(source.identity) if source.identity else None,
        video_width=source.video_width,
        video_height=source.video_height,
        video_duration_seconds=source.video_duration_seconds,
    )


__all__ = ["build_quality_source_truth"]


def _build_identity_label(identity) -> str:
    return f"{identity.split}:{identity.video_key}:{identity.sample_key}:{identity.sentence_key}"
