"""Pose truth rehydration from checkpoint authority."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.data.gate.pose import CoordinateSpace, PoseChannel
from text_to_sign_production.data.tier._shared.types import (
    QualityPoseChannelTruth,
    QualityPoseTruth,
)
from text_to_sign_production.workflows._shared.checkpoint_rehydration.types import (
    CheckpointRehydrationInput,
)


def build_quality_pose_truth(
    input: CheckpointRehydrationInput,
) -> QualityPoseTruth:
    """Build downstream pose truth from payload checkpoint truth.

    Consumes payload truth only. Does NOT rerun raw pose
    parsing, tracking, or tensorization. Does NOT go back to
    raw keypoint JSON.
    """
    payload = input.payload
    frame = payload.frame
    tracking = payload.tracking

    channels: dict[PoseChannel, QualityPoseChannelTruth] = {}
    for channel, channel_payload in payload.pose_channels.items():
        channels[channel] = QualityPoseChannelTruth(
            channel=channel,
            coordinates=channel_payload.coordinates,
            confidences=channel_payload.confidences,
            coordinate_space=channel_payload.coordinate_space,
        )

    channel_nonzero: dict[PoseChannel, int] = {
        channel: count for channel, count in frame.channel_nonzero_frame_counts.items()
    }

    return QualityPoseTruth(
        frame_count=frame.frame_count,
        valid_frame_count=int(np.count_nonzero(frame.frame_valid_mask)),
        people_per_frame=frame.people_per_frame,
        frame_valid_mask=frame.frame_valid_mask,
        channels=channels,
        channel_nonzero_frame_counts=channel_nonzero,
        coordinate_space=CoordinateSpace.NORMALIZED_IMAGE,
        anchor_person_index=tracking.anchor_person_index,
        selected_person_indices=tracking.selected_person_indices,
        target_missing_frame_count=tracking.target_missing_frame_count,
        continuity_break_count=tracking.continuity_break_count,
        reanchor_count=tracking.reanchor_count,
        diagnostic_codes=(),
    )


__all__ = ["build_quality_pose_truth"]
