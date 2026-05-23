from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from text_to_sign_production.artifacts.store import build_artifact_topology
from text_to_sign_production.core import build_repo_roots
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.artifacts import (
    load_generated_pose_payload,
    read_generated_pose_manifest_jsonl,
    validate_generated_pose_manifest_entries,
    write_generated_pose_split,
)
from text_to_sign_production.modeling.backbones.bfh_vectorization import vectorize_bfh_pose_arrays
from text_to_sign_production.modeling.candidates.learned_pose_token.dataset import (
    PoseTokenSourceSample,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.exporter import (
    DecodedPoseSample,
    export_predicted_pose_samples,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.spec import (
    LEARNED_POSE_TOKEN_MODEL_KEY,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.tokens import (
    POSE_TOKEN_SCHEMA_VERSION,
    PoseTokenSequence,
)
from text_to_sign_production.modeling.data.bfh_schema import (
    BFH_CHANNEL_SPECS,
    BfhPoseArrays,
)


@pytest.mark.unit
def test_learned_pose_token_decoded_intermediate_uses_diagnostic_layout(
    tmp_path: Path,
) -> None:
    root = tmp_path / "intermediates" / "decode_to_pose" / "decoded_pose_intermediates" / "val"

    written = export_predicted_pose_samples(
        decoded_samples=(_decoded_sample("sample1"),),
        diagnostic_root=root,
        producer_key=LEARNED_POSE_TOKEN_MODEL_KEY,
        producer_stage="decode_to_pose",
        run_name="run001",
        split=SampleSplit.VAL,
        generation_mode="deterministic",
        seed=123,
    )

    assert written.manifest_path == root / "generated_pose" / "manifest.jsonl"
    assert written.payload_paths == (root / "generated_pose" / "samples" / "sample1__g0.npz",)
    entries = read_generated_pose_manifest_jsonl(written.manifest_path, expected_split=SampleSplit.VAL)
    assert entries[0].generated_payload_ref == "generated_pose/samples/sample1__g0.npz"
    assert not validate_generated_pose_manifest_entries(entries, expected_split=SampleSplit.VAL)


@pytest.mark.unit
def test_learned_pose_token_intermediate_and_final_generated_pose_paths_are_separate(
    tmp_path: Path,
) -> None:
    topology = build_artifact_topology(build_repo_roots(tmp_path / "runtime"))
    decoded_root = (
        topology.models.model_intermediate_root(
            LEARNED_POSE_TOKEN_MODEL_KEY,
            "run001",
            "decode_to_pose",
        ).path
        / "decoded_pose_intermediates"
        / "val"
    )
    intermediate = export_predicted_pose_samples(
        decoded_samples=(_decoded_sample("sample1"),),
        diagnostic_root=decoded_root,
        producer_key=LEARNED_POSE_TOKEN_MODEL_KEY,
        producer_stage="decode_to_pose",
        run_name="run001",
        split=SampleSplit.VAL,
        generation_mode="deterministic",
        seed=123,
    )
    final = write_generated_pose_split(
        topology,
        producer_key=LEARNED_POSE_TOKEN_MODEL_KEY,
        run_name="run001",
        split=SampleSplit.VAL,
        samples=(load_generated_pose_payload(intermediate.payload_paths[0]),),
    )

    assert intermediate.manifest_path != final.manifest_path
    assert intermediate.manifest_path.as_posix().endswith(
        "decoded_pose_intermediates/val/generated_pose/manifest.jsonl"
    )
    assert (
        "evaluations/generated_pose/learned_pose_token/run001/val/manifest.jsonl"
        in final.manifest_path.as_posix()
    )


def _decoded_sample(sample_id: str) -> DecodedPoseSample:
    pose = _pose()
    source = PoseTokenSourceSample(
        sample_id=sample_id,
        source_sentence_name=f"sentence_{sample_id}",
        text=f"text {sample_id}",
        source_video_id=f"video-{sample_id}",
        source_sentence_id=f"sent-{sample_id}",
        reference_payload_ref=f"passed/val/{sample_id}.npz",
        split=SampleSplit.VAL,
        frame_count=pose.frame_count,
        vectorized_pose=vectorize_bfh_pose_arrays(pose, sample_id=sample_id),
    )
    return DecodedPoseSample(
        source=source,
        token_sequence=PoseTokenSequence(
            schema_version=POSE_TOKEN_SCHEMA_VERSION,
            sample_id=sample_id,
            source_sentence_name=f"sentence_{sample_id}",
            split=SampleSplit.VAL,
            token_ids=np.asarray([0, 1], dtype=np.int64),
            frame_count=2,
            token_count=2,
            codebook_size=4,
            temporal_granularity="frame",
            window_size=1,
            stride=1,
        ),
        pose=pose,
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
