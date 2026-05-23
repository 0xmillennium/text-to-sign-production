from __future__ import annotations

from pathlib import Path

import numpy as np

from text_to_sign_production.artifacts.store import ArtifactTopology, build_artifact_topology
from text_to_sign_production.core import build_repo_roots
from text_to_sign_production.core.ids import CoordinateSpace, SampleSplit, TierMembership
from text_to_sign_production.core.models import PassedManifestEntry, PoseTruth, PreparedSample, SourceTruth
from text_to_sign_production.data.dataset import (
    PREPARED_SAMPLE_SCHEMA_VERSION,
    write_prepared_sample_payload,
    write_tier_manifest_json,
)
from text_to_sign_production.data.dataset.manifests import GATE_MANIFEST_SCHEMA_VERSION


def write_real_modeling_topology(root: Path, *, sample_count: int = 3) -> ArtifactTopology:
    topology = build_artifact_topology(build_repo_roots(root))
    for split in (SampleSplit.TRAIN, SampleSplit.VAL, SampleSplit.TEST):
        entries: list[PassedManifestEntry] = []
        for index in range(sample_count):
            sample_id = f"{split.value}-{index}"
            sample = _prepared_sample(sample_id, split=split, shift=float(index) * 0.01)
            payload_ref = f"passed/{split.value}/{sample_id}.npz"
            write_prepared_sample_payload(topology.samples_root / payload_ref, sample)
            entries.append(_entry(sample, payload_ref=payload_ref))
        write_tier_manifest_json(
            topology.manifests.tiered_manifest("clean", TierMembership.INCLUDED, split).path,
            entries,
            tier="clean",
            membership=TierMembership.INCLUDED,
            split=split,
        )
    return topology


def _entry(sample: PreparedSample, *, payload_ref: str) -> PassedManifestEntry:
    pose = sample.pose
    return PassedManifestEntry(
        schema_version=GATE_MANIFEST_SCHEMA_VERSION,
        sample_id=sample.source.sample_id,
        split=sample.source.split,
        payload_ref=payload_ref,
        text=sample.source.text,
        fps=25.0,
        frame_count=pose.frame_count,
        source_video_id=sample.source.source_video_id,
        source_sentence_id=sample.source.source_sentence_id,
        source_sentence_name=sample.source.source_sentence_name,
        valid_frame_count=int(np.count_nonzero(pose.valid_frame_mask)),
        body_nonzero_frame_count=pose.body_nonzero_frame_count,
        face_nonzero_frame_count=pose.face_nonzero_frame_count,
        left_hand_nonzero_frame_count=pose.left_hand_nonzero_frame_count,
        right_hand_nonzero_frame_count=pose.right_hand_nonzero_frame_count,
    )


def _prepared_sample(sample_id: str, *, split: SampleSplit, shift: float) -> PreparedSample:
    frames = 4
    body = _channel(frames, 25, shift)
    face = _channel(frames, 70, shift + 0.1)
    hand_l = _channel(frames, 21, shift + 0.2)
    hand_r = _channel(frames, 21, shift + 0.3)
    valid = np.ones((frames,), dtype=np.bool_)
    return PreparedSample(
        schema_version=PREPARED_SAMPLE_SCHEMA_VERSION,
        source=SourceTruth(
            sample_id=sample_id,
            split=split,
            text=f"text {sample_id}",
            fps=25.0,
            source_video_id=f"video-{sample_id}",
            source_sentence_id=f"sent-{sample_id}",
            source_sentence_name=f"sentence_{sample_id}",
            source_issue_codes=(),
        ),
        pose=PoseTruth(
            coordinate_space=CoordinateSpace.NORMALIZED_IMAGE,
            frame_count=frames,
            valid_frame_mask=valid,
            selected_person_indices=tuple(0 for _ in range(frames)),
            tracked_target_missing_frame_count=0,
            continuity_break_count=0,
            reanchor_count=0,
            body_xyc=body,
            face_xyc=face,
            left_hand_xyc=hand_l,
            right_hand_xyc=hand_r,
            body_nonzero_frame_count=frames,
            face_nonzero_frame_count=frames,
            left_hand_nonzero_frame_count=frames,
            right_hand_nonzero_frame_count=frames,
        ),
    )


def _channel(frames: int, joints: int, shift: float) -> np.ndarray:
    values = np.zeros((frames, joints, 3), dtype=np.float32)
    for frame in range(frames):
        values[frame, :, 0] = np.linspace(0.1, 0.9, joints, dtype=np.float32) + shift
        values[frame, :, 1] = np.linspace(0.2, 0.8, joints, dtype=np.float32) + frame * 0.01
        values[frame, :, 2] = 1.0
    return values
