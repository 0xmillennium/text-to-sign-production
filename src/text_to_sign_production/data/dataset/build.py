"""Dataset production boundary for prepared sample payload construction.

This module owns dataset-stage sample construction and payload write planning.
Gate-specific manifest row and drop planning lives under ``data.gate``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from text_to_sign_production.core.ids import (
    CoordinateSpace,
    SampleStatus,
)
from text_to_sign_production.core.models import (
    PoseTruth,
    PreparedSample,
    SourceTruth,
)
from text_to_sign_production.data.dataset.payloads import (
    PREPARED_SAMPLE_PAYLOAD_SCHEMA_VERSION,
    write_prepared_sample_payload,
)
from text_to_sign_production.data.dataset.types import (
    DatasetPayloadProduction,
)
from text_to_sign_production.data.gate.pose import (
    PoseBuildOutput,
    PoseChannel,
    PoseChannelTensor,
)
from text_to_sign_production.data.gate.sources import (
    SourceCandidate,
)

PREPARED_SAMPLE_SCHEMA_VERSION = PREPARED_SAMPLE_PAYLOAD_SCHEMA_VERSION


# Sample construction


def build_prepared_sample(
    source_candidate: SourceCandidate,
    pose_output: PoseBuildOutput,
    *,
    schema_version: str,
) -> PreparedSample:
    """Build the canonical gate-stage object from matched source and pose truth."""
    source = SourceTruth(
        sample_id=source_candidate.sample_id,
        split=source_candidate.split,
        text=source_candidate.text,
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


# Payload production helpers


def plan_prepared_sample_payload(
    *,
    sample: PreparedSample,
    status: SampleStatus,
    payload_path: str | Path,
    payload_ref: str,
) -> DatasetPayloadProduction:
    """Build a prepared sample payload write plan without mutating the filesystem."""
    path = Path(payload_path)
    return DatasetPayloadProduction(
        sample=sample,
        path=path,
        payload_ref=payload_ref,
        status=SampleStatus(status),
    )


def write_prepared_sample_payload_plan(
    production: DatasetPayloadProduction,
) -> DatasetPayloadProduction:
    """Persist one prepared sample payload plan."""
    write_prepared_sample_payload(production.path, production.sample)
    return production


def produce_prepared_sample_payload(
    *,
    sample: PreparedSample,
    status: SampleStatus,
    payload_path: str | Path,
    payload_ref: str,
) -> DatasetPayloadProduction:
    """Write a prepared sample payload and return dataset-owned production facts."""
    return write_prepared_sample_payload_plan(
        plan_prepared_sample_payload(
            sample=sample,
            status=status,
            payload_path=payload_path,
            payload_ref=payload_ref,
        )
    )


def _require_fps(candidate: SourceCandidate) -> float:
    fps = candidate.video_metadata.fps
    if fps is None:
        raise ValueError("Source candidate video metadata must carry fps.")
    return float(fps)


def _xyc(channel: PoseChannelTensor) -> np.ndarray:
    coordinates = np.asarray(channel.coordinates, dtype=np.float32)
    confidences = np.asarray(channel.confidences, dtype=np.float32)
    return np.concatenate((coordinates, confidences[..., np.newaxis]), axis=-1)


__all__ = [
    "DatasetPayloadProduction",
    "PREPARED_SAMPLE_SCHEMA_VERSION",
    "build_prepared_sample",
    "plan_prepared_sample_payload",
    "produce_prepared_sample_payload",
    "write_prepared_sample_payload_plan",
]
