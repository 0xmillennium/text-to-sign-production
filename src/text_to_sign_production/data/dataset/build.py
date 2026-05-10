"""Prepared sample construction from source and pose authorities."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.core.models import PoseTruth, PreparedSample, SourceTruth
from text_to_sign_production.data.gate.pose import (
    CoordinateSpace,
    PoseBuildOutput,
    PoseChannel,
    PoseChannelTensor,
)
from text_to_sign_production.data.gate.sources import SourceCandidate


def build_prepared_sample(
    source_candidate: SourceCandidate,
    pose_output: PoseBuildOutput,
    *,
    schema_version: str,
) -> PreparedSample:
    """Build the canonical samples-stage object from matched source and pose truth."""
    source = SourceTruth(
        sample_id=source_candidate.sample_id,
        split=source_candidate.split,
        text=source_candidate.text,
        canonical_normalized_text=_require_canonical_normalized_text(source_candidate),
        fps=_require_fps(source_candidate),
        source_video_id=source_candidate.video_id,
        source_sentence_id=source_candidate.sentence_id,
        source_sentence_name=source_candidate.sentence_name,
        source_issue_codes=tuple(issue.code for issue in source_candidate.source_issues),
    )
    tensors = pose_output.tensors
    counts = tensors.channel_nonzero_frame_counts
    pose = PoseTruth(
        coordinate_space=CoordinateSpace.NORMALIZED_IMAGE,
        frame_count=len(tensors.frame_valid_mask),
        valid_frame_mask=np.asarray(tensors.frame_valid_mask, dtype=np.bool_),
        selected_person_indices=tuple(tensors.selected_person_indices),
        tracked_target_missing_frame_count=pose_output.tracking.target_missing_frame_count,
        continuity_break_count=pose_output.tracking.continuity_break_count,
        reanchor_count=pose_output.tracking.reanchor_count,
        body_xyc=_xyc(tensors.channels[PoseChannel.BODY]),
        face_xyc=_xyc(tensors.channels[PoseChannel.FACE]),
        left_hand_xyc=_xyc(tensors.channels[PoseChannel.LEFT_HAND]),
        right_hand_xyc=_xyc(tensors.channels[PoseChannel.RIGHT_HAND]),
        body_nonzero_frame_count=int(counts[PoseChannel.BODY]),
        face_nonzero_frame_count=int(counts[PoseChannel.FACE]),
        left_hand_nonzero_frame_count=int(counts[PoseChannel.LEFT_HAND]),
        right_hand_nonzero_frame_count=int(counts[PoseChannel.RIGHT_HAND]),
    )
    return PreparedSample(schema_version=schema_version, source=source, pose=pose)


def _require_fps(candidate: SourceCandidate) -> float:
    fps = candidate.video_metadata.fps
    if fps is None:
        raise ValueError("Source candidate video metadata must carry fps.")
    return float(fps)


def _require_canonical_normalized_text(candidate: SourceCandidate) -> str:
    value = candidate.canonical_normalized_text
    if value is None or value == "":
        raise ValueError("Source candidate must carry authoritative canonical_normalized_text.")
    return value


def _xyc(channel: PoseChannelTensor) -> np.ndarray:
    coordinates = np.asarray(channel.coordinates, dtype=np.float32)
    confidences = np.asarray(channel.confidences, dtype=np.float32)
    return np.concatenate((coordinates, confidences[..., np.newaxis]), axis=-1)


__all__ = ["build_prepared_sample"]
