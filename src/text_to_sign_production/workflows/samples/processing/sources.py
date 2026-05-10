from __future__ import annotations

from text_to_sign_production.data.gate.sources import (
    TranslationSourceRecord,
    assemble_candidate,
    build_keypoint_record,
    build_video_record,
    match_sources,
)
from text_to_sign_production.workflows.samples.contracts import SamplesSplitRuntimeInputs
from text_to_sign_production.workflows.samples.processing.models import SamplesSourceBundle


def build_samples_source_bundle(
    *,
    split_inputs: SamplesSplitRuntimeInputs,
    translation: TranslationSourceRecord,
) -> SamplesSourceBundle:
    keypoint_dir = split_inputs.keypoint_json_root / translation.sentence_name
    video_path = split_inputs.keypoint_video_root / f"{translation.sentence_name}.mp4"
    keypoint_sample_id = (
        translation.identity.keypoint.sample_key.value
        if translation.identity is not None
        else translation.sentence_id
    )
    match = match_sources(
        translation=translation,
        split=split_inputs.split,
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
    return SamplesSourceBundle(translation=translation, match=match)


__all__ = ["build_samples_source_bundle"]
