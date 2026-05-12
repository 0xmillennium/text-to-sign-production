"""Gate-stage source bundling and matching boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.data.gate.sources.candidates import assemble_candidate
from text_to_sign_production.data.gate.sources.keypoints import build_keypoint_record
from text_to_sign_production.data.gate.sources.matching import match_sources
from text_to_sign_production.data.gate.sources.types import (
    SourceMatchResult,
    TranslationSourceRecord,
)
from text_to_sign_production.data.gate.sources.videos import build_video_record


@dataclass(frozen=True, slots=True)
class GateSourceBundle:
    """Data-owned source matching result for one translation row."""

    translation: TranslationSourceRecord
    match: SourceMatchResult


def build_gate_source_bundle(
    *,
    split: str,
    translation: TranslationSourceRecord,
    keypoint_json_root: Path,
    keypoint_video_root: Path,
) -> GateSourceBundle:
    """Build source matching truth for one translation row."""
    keypoint_dir = keypoint_json_root / translation.sentence_name
    video_path = keypoint_video_root / f"{translation.sentence_name}.mp4"
    keypoint_sample_id = (
        translation.identity.keypoint.sample_key.value
        if translation.identity is not None
        else translation.sentence_name
    )
    match = match_sources(
        translation=translation,
        split=split,
        videos=(
            build_video_record(
                video_path,
                video_id=translation.video_id,
                video_name=translation.video_name,
            ),
        ),
        keypoints=(
            build_keypoint_record(
                keypoint_dir,
                sample_id=keypoint_sample_id,
            ),
        ),
    )
    if match.matched:
        assemble_candidate(match)
    return GateSourceBundle(translation=translation, match=match)


__all__ = [
    "GateSourceBundle",
    "build_gate_source_bundle",
]
