"""Domain constants shared across the Dataset Build data pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Final

DEFAULT_FILTER_CONFIG_RELATIVE_PATH: Final[Path] = Path("configs") / "data" / "filter-v2.yaml"

SPLITS: Final[tuple[str, str, str]] = ("train", "val", "test")
SPLIT_TO_TRANSLATION_FILE: Final[dict[str, str]] = {
    "train": "how2sign_realigned_train.csv",
    "val": "how2sign_realigned_val.csv",
    "test": "how2sign_realigned_test.csv",
}
SPLIT_TO_KEYPOINT_DIR: Final[dict[str, str]] = {
    "train": "train_2D_keypoints",
    "val": "val_2D_keypoints",
    "test": "test_2D_keypoints",
}
EXPECTED_TRANSLATION_COLUMNS: Final[tuple[str, ...]] = (
    "VIDEO_ID",
    "VIDEO_NAME",
    "SENTENCE_ID",
    "SENTENCE_NAME",
    "START_REALIGNED",
    "END_REALIGNED",
    "SENTENCE",
)

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

OPENPOSE_CHANNEL_SPECS: Final[dict[str, tuple[str, int, bool]]] = {
    "body": ("pose_keypoints_2d", 25, True),
    "face": ("face_keypoints_2d", 70, False),
    "left_hand": ("hand_left_keypoints_2d", 21, True),
    "right_hand": ("hand_right_keypoints_2d", 21, True),
}
CORE_CHANNELS: Final[tuple[str, ...]] = ("body", "left_hand", "right_hand")
LEGACY_REQUIRED_CORE_CHANNELS: Final[tuple[str, ...]] = CORE_CHANNELS
OPTIONAL_CHANNELS: Final[tuple[str, ...]] = ("face",)

CANVAS_WIDTH: Final[int] = 1280
CANVAS_HEIGHT: Final[int] = 720
PROCESSED_SCHEMA_VERSION: Final[str] = "t2sp-processed-v1"
