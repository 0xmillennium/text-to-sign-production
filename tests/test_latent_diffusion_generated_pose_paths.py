from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.artifacts import (
    GENERATED_POSE_CHANNEL_POLICY,
    GENERATED_POSE_PAYLOAD_SCHEMA_VERSION,
    GeneratedPoseConfidencePolicy,
    GeneratedPoseGenerationMode,
    GeneratedPoseLengthPolicy,
    GeneratedPoseProducerType,
    GeneratedPoseSample,
    diagnostic_generated_pose_paths,
    read_generated_pose_manifest_jsonl,
    validate_diagnostic_generated_pose_paths,
    write_generated_pose_split_to_explicit_root,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.spec import (
    LATENT_DIFFUSION_CANONICAL_ID,
    LATENT_DIFFUSION_MODEL_KEY,
    LATENT_DIFFUSION_PHASE_NUMBER,
    LATENT_DIFFUSION_RESEARCH_ROLE,
)
from text_to_sign_production.modeling.data.bfh_schema import (
    BFH_CHANNEL_SPECS,
    BfhPoseArrays,
)


@pytest.mark.unit
def test_latent_diffusion_decoded_intermediate_uses_diagnostic_layout(
    tmp_path: Path,
) -> None:
    root = tmp_path / "intermediates" / "generation" / "decoded_pose_intermediates" / "val"
    paths = diagnostic_generated_pose_paths(root)
    validate_diagnostic_generated_pose_paths(paths)

    written = write_generated_pose_split_to_explicit_root(
        manifest_path=paths.manifest_path,
        samples_root=paths.samples_root,
        payload_ref_root=paths.payload_ref_root,
        split=SampleSplit.VAL,
        samples=(_generated_sample("sample1"),),
    )

    assert written.manifest_path == root / "generated_pose" / "manifest.jsonl"
    assert written.payload_paths == (root / "generated_pose" / "samples" / "sample1__g0.npz",)
    entries = read_generated_pose_manifest_jsonl(written.manifest_path, expected_split=SampleSplit.VAL)
    assert entries[0].producer_key == LATENT_DIFFUSION_MODEL_KEY
    assert entries[0].generated_payload_ref == "generated_pose/samples/sample1__g0.npz"


def _generated_sample(sample_id: str) -> GeneratedPoseSample:
    return GeneratedPoseSample(
        schema_version=GENERATED_POSE_PAYLOAD_SCHEMA_VERSION,
        producer_type=GeneratedPoseProducerType.MODEL,
        producer_key=LATENT_DIFFUSION_MODEL_KEY,
        canonical_id=LATENT_DIFFUSION_CANONICAL_ID,
        phase_number=LATENT_DIFFUSION_PHASE_NUMBER,
        research_role=LATENT_DIFFUSION_RESEARCH_ROLE,
        run_name="run001",
        split=SampleSplit.VAL,
        sample_id=sample_id,
        text=f"text {sample_id}",
        source_video_id=f"video-{sample_id}",
        source_sentence_id=f"sent-{sample_id}",
        source_sentence_name=f"sentence_{sample_id}",
        reference_payload_ref=f"passed/val/{sample_id}.npz",
        generation_index=0,
        num_candidates_for_sample=1,
        generation_mode=GeneratedPoseGenerationMode.STOCHASTIC,
        length_policy=GeneratedPoseLengthPolicy.PREDICTED_LENGTH,
        channel_policy=GENERATED_POSE_CHANNEL_POLICY,
        confidence_policy=GeneratedPoseConfidencePolicy.SYNTHETIC_VALIDITY,
        seed=123,
        failure_reason=None,
        pose=_pose(),
    )


def _pose(frame_count: int = 2) -> BfhPoseArrays:
    return BfhPoseArrays(
        body_xyc=_channel(frame_count, PoseChannel.BODY),
        left_hand_xyc=_channel(frame_count, PoseChannel.LEFT_HAND),
        right_hand_xyc=_channel(frame_count, PoseChannel.RIGHT_HAND),
        face_xyc=_channel(frame_count, PoseChannel.FACE),
        valid_frame_mask=np.ones((frame_count,), dtype=np.bool_),
    )


def _channel(frame_count: int, channel: PoseChannel) -> np.ndarray:
    array = np.zeros(
        (frame_count, BFH_CHANNEL_SPECS[channel].joint_count, 3),
        dtype=np.float32,
    )
    array[..., 2] = 1.0
    return array
