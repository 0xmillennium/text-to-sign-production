from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from text_to_sign_production.core.ids import CoordinateSpace, SampleSplit
from text_to_sign_production.core.models import PoseTruth, PreparedSample, SourceTruth
from text_to_sign_production.data.dataset.build import (
    PREPARED_SAMPLE_SCHEMA_VERSION,
    build_prepared_sample,
)
from text_to_sign_production.data.dataset.types import DatasetValidationIssueCode
from text_to_sign_production.data.dataset.validate import validate_prepared_sample
from text_to_sign_production.data.gate.pose.schema import POSE_CHANNEL_JOINT_COUNTS
from text_to_sign_production.data.gate.pose.types import (
    AnchorSelection,
    FrameTrackingSelection,
    PersonSelectionPolicy,
    PersonTrackingResult,
    PoseBuildDiagnostics,
    PoseBuildOutput,
    PoseChannel,
    PoseChannelTensor,
    PoseTensorOutput,
    TrackingSelectionReason,
)
from text_to_sign_production.data.gate.sources.types import SourceCandidate, VideoMetadata


@pytest.mark.unit
def test_build_prepared_sample_clamps_confidence() -> None:
    sample = build_prepared_sample(
        _source_candidate(),
        _pose_output(
            PoseChannel.BODY,
            np.asarray([-0.25, 0.0, 0.5, 1.0, 1.25], dtype=np.float32),
        ),
        schema_version=PREPARED_SAMPLE_SCHEMA_VERSION,
    )

    assert sample.pose.body_xyc[0, :5, 2].tolist() == [0.0, 0.0, 0.5, 1.0, 1.0]


@pytest.mark.unit
def test_validator_catches_corrupted_prepared_sample_confidence_range() -> None:
    sample = _prepared_sample_with_body_confidence(np.asarray([1.0], dtype=np.float32))
    body_xyc = sample.pose.body_xyc.copy()
    body_xyc[0, 0, 2] = 1.2
    corrupted = _replace_pose(sample, body_xyc=body_xyc)

    issues = validate_prepared_sample(corrupted)

    assert DatasetValidationIssueCode.INVALID_POSE_CONFIDENCE_RANGE in {
        issue.code for issue in issues
    }


@pytest.mark.unit
def test_built_prepared_sample_with_raw_confidence_overflow_validates_cleanly() -> None:
    sample = _prepared_sample_with_body_confidence(np.asarray([1.25], dtype=np.float32))

    issues = validate_prepared_sample(sample)

    assert DatasetValidationIssueCode.INVALID_POSE_CONFIDENCE_RANGE not in {
        issue.code for issue in issues
    }
    assert float(np.max(sample.pose.body_xyc[..., 2])) == 1.0


def _prepared_sample_with_body_confidence(confidences: np.ndarray) -> PreparedSample:
    return build_prepared_sample(
        _source_candidate(),
        _pose_output(PoseChannel.BODY, confidences),
        schema_version=PREPARED_SAMPLE_SCHEMA_VERSION,
    )


def _source_candidate() -> SourceCandidate:
    return SourceCandidate(
        sample_id="sample_1",
        split=SampleSplit.VAL,
        text="hello",
        start_time=0.0,
        end_time=1.0,
        video_id="video_1",
        video_name="video_1.mp4",
        sentence_id="sentence_1",
        sentence_name="sentence_1",
        video_path=Path("video_1.mp4"),
        keypoints_dir=Path("keypoints"),
        frame_count=1,
        video_metadata=VideoMetadata(width=640, height=480, fps=25.0, frame_count=1),
    )


def _pose_output(channel: PoseChannel, confidences: np.ndarray) -> PoseBuildOutput:
    channels = {
        item: _channel_tensor(
            item,
            confidences if item is channel else np.asarray([1.0], dtype=np.float32),
        )
        for item in PoseChannel
    }
    return PoseBuildOutput(
        tensors=PoseTensorOutput(
            candidate=_source_candidate(),
            channels=channels,
            people_per_frame=np.ones((1,), dtype=np.int16),
            frame_valid_mask=np.ones((1,), dtype=np.bool_),
            selected_person_indices=(0,),
            channel_nonzero_frame_counts={item: 1 for item in PoseChannel},
        ),
        tracking=PersonTrackingResult(
            anchor=AnchorSelection(
                anchor_person_index=0,
                policy=PersonSelectionPolicy.OPENPOSE_PRIMARY,
                fallback_used=False,
                fallback_reason=None,
                candidate_scores=(),
            ),
            frame_selections=(
                FrameTrackingSelection(
                    frame_index=0,
                    selected_person_index=0,
                    reason=TrackingSelectionReason.ANCHOR,
                ),
            ),
            continuity_break_count=0,
            reanchor_count=0,
            target_missing_frame_count=0,
        ),
        diagnostics=PoseBuildDiagnostics(),
    )


def _channel_tensor(channel: PoseChannel, confidences: np.ndarray) -> PoseChannelTensor:
    joint_count = POSE_CHANNEL_JOINT_COUNTS[channel]
    coordinates = np.zeros((1, joint_count, 2), dtype=np.float32)
    confidence_array = np.ones((1, joint_count), dtype=np.float32)
    confidence_array[0, : confidences.size] = confidences
    return PoseChannelTensor(
        channel=channel,
        coordinates=coordinates,
        confidences=confidence_array,
    )


def _replace_pose(sample: PreparedSample, *, body_xyc: np.ndarray) -> PreparedSample:
    pose = sample.pose
    return PreparedSample(
        schema_version=sample.schema_version,
        source=sample.source,
        pose=PoseTruth(
            coordinate_space=CoordinateSpace.NORMALIZED_IMAGE,
            frame_count=pose.frame_count,
            valid_frame_mask=pose.valid_frame_mask,
            selected_person_indices=pose.selected_person_indices,
            tracked_target_missing_frame_count=pose.tracked_target_missing_frame_count,
            continuity_break_count=pose.continuity_break_count,
            reanchor_count=pose.reanchor_count,
            body_xyc=body_xyc,
            face_xyc=pose.face_xyc,
            left_hand_xyc=pose.left_hand_xyc,
            right_hand_xyc=pose.right_hand_xyc,
            body_nonzero_frame_count=pose.body_nonzero_frame_count,
            face_nonzero_frame_count=pose.face_nonzero_frame_count,
            left_hand_nonzero_frame_count=pose.left_hand_nonzero_frame_count,
            right_hand_nonzero_frame_count=pose.right_hand_nonzero_frame_count,
        ),
    )
