from __future__ import annotations

from text_to_sign_production.data.sources import (
    TranslationRow,
    assemble_candidate,
    match_sources,
    read_video_metadata,
    resolve_keypoint_source,
)
from text_to_sign_production.workflows.samples.contracts import SamplesSplitRuntimeInputs
from text_to_sign_production.workflows.samples.processing.models import SamplesSourceBundle


def build_samples_source_bundle(
    *,
    split_inputs: SamplesSplitRuntimeInputs,
    translation: TranslationRow,
) -> SamplesSourceBundle:
    keypoint_dir = split_inputs.keypoint_json_root / translation.sentence_name
    video_path = split_inputs.keypoint_video_root / f"{translation.sentence_name}.mp4"
    keypoints = resolve_keypoint_source(keypoint_dir)
    video_metadata = read_video_metadata(video_path)
    match = match_sources(
        translation=translation,
        split=split_inputs.split,
        keypoints=keypoints,
        video_metadata=video_metadata,
    )
    candidate = assemble_candidate(match, video_path) if match.matched else None
    return SamplesSourceBundle(
        translation=translation,
        video_path=video_path,
        match=match,
        candidate=candidate,
    )
