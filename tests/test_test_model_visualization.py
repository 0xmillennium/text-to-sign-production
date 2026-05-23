from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.data import BFH_CHANNEL_SPECS
from text_to_sign_production.visualization import SkeletonRenderConfig, render_pose_pair_video
from text_to_sign_production.visualization.pose import PoseSample


@pytest.mark.unit
def test_render_pose_pair_video_writes_side_by_side_mp4(tmp_path: Path) -> None:
    output_path = tmp_path / "reference_vs_generated_pose.mp4"

    metadata = render_pose_pair_video(
        left_pose_sample=_pose_sample("left", frame_count=2),
        right_pose_sample=_pose_sample("right", frame_count=3),
        output_path=output_path,
        fps=12.0,
        left_label="reference",
        right_label="generated",
        config=SkeletonRenderConfig(canvas_width=96, canvas_height=72),
    )

    assert output_path.is_file()
    assert metadata["layout"] == "pose_pair_side_by_side"
    assert metadata["output_frame_count"] == 3
    assert metadata["left_frames"] == 2
    assert metadata["right_frames"] == 3
    assert metadata["fps"] == pytest.approx(12.0)


def _pose_sample(name: str, *, frame_count: int) -> PoseSample:
    return PoseSample(
        path=Path(f"{name}.npz"),
        schema_version="test-schema",
        body=_coords(frame_count, PoseChannel.BODY),
        body_confidence=_confidence(frame_count, PoseChannel.BODY),
        left_hand=_coords(frame_count, PoseChannel.LEFT_HAND),
        left_hand_confidence=_confidence(frame_count, PoseChannel.LEFT_HAND),
        right_hand=_coords(frame_count, PoseChannel.RIGHT_HAND),
        right_hand_confidence=_confidence(frame_count, PoseChannel.RIGHT_HAND),
        face=_coords(frame_count, PoseChannel.FACE),
        face_confidence=_confidence(frame_count, PoseChannel.FACE),
        people_per_frame=np.ones((frame_count,), dtype=np.int16),
        selected_person_index=0,
        frame_valid_mask=np.ones((frame_count,), dtype=np.bool_),
    )


def _coords(frame_count: int, channel: PoseChannel) -> np.ndarray:
    joints = BFH_CHANNEL_SPECS[channel].joint_count
    coords = np.full((frame_count, joints, 2), 0.5, dtype=np.float32)
    return coords


def _confidence(frame_count: int, channel: PoseChannel) -> np.ndarray:
    joints = BFH_CHANNEL_SPECS[channel].joint_count
    return np.ones((frame_count, joints), dtype=np.float32)
