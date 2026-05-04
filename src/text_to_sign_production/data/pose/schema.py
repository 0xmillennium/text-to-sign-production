"""Schema and constants for OpenPose parsing and tensors."""

from __future__ import annotations

from typing import Final

EXPECTED_OPENPOSE_TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset({"people", "version"})
EXPECTED_OPENPOSE_PERSON_KEYS: Final[frozenset[str]] = frozenset(
    {
        "person_id",
        "pose_keypoints_2d",
        "face_keypoints_2d",
        "hand_left_keypoints_2d",
        "hand_right_keypoints_2d",
        "pose_keypoints_3d",
        "face_keypoints_3d",
        "hand_left_keypoints_3d",
        "hand_right_keypoints_3d",
    }
)
EXPECTED_OPENPOSE_3D_KEYS: Final[tuple[str, ...]] = (
    "pose_keypoints_3d",
    "face_keypoints_3d",
    "hand_left_keypoints_3d",
    "hand_right_keypoints_3d",
)

OPENPOSE_CHANNEL_SPECS: Final[dict[str, tuple[str, int]]] = {
    "body": ("pose_keypoints_2d", 25),
    "face": ("face_keypoints_2d", 70),
    "left_hand": ("hand_left_keypoints_2d", 21),
    "right_hand": ("hand_right_keypoints_2d", 21),
}

CANONICAL_POSE_CHANNELS: Final[tuple[str, ...]] = ("body", "left_hand", "right_hand", "face")
POSE_CHANNEL_JOINT_COUNTS: Final[dict[str, int]] = {
    channel: joint_count for channel, (_, joint_count) in OPENPOSE_CHANNEL_SPECS.items()
}
POSE_COORDINATE_DIMENSIONS: Final[int] = 2

CANVAS_WIDTH: Final[int] = 1280
CANVAS_HEIGHT: Final[int] = 720
