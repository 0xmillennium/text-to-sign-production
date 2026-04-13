"""Constants shared across the Sprint 2 data pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final


def _resolve_repo_root() -> Path:
    override = os.environ.get("T2SP_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


REPO_ROOT: Final[Path] = _resolve_repo_root()

RAW_ROOT: Final[Path] = REPO_ROOT / "data" / "raw" / "how2sign"
TRANSLATIONS_DIR: Final[Path] = RAW_ROOT / "translations"
BFH_KEYPOINTS_ROOT: Final[Path] = RAW_ROOT / "bfh_keypoints"
ARCHIVES_RELATIVE_ROOT: Final[Path] = Path("data") / "archives"
ARCHIVES_ROOT: Final[Path] = REPO_ROOT / ARCHIVES_RELATIVE_ROOT

INTERIM_ROOT: Final[Path] = REPO_ROOT / "data" / "interim"
RAW_MANIFESTS_ROOT: Final[Path] = INTERIM_ROOT / "raw_manifests"
NORMALIZED_MANIFESTS_ROOT: Final[Path] = INTERIM_ROOT / "normalized_manifests"
FILTERED_MANIFESTS_ROOT: Final[Path] = INTERIM_ROOT / "filtered_manifests"
INTERIM_REPORTS_ROOT: Final[Path] = INTERIM_ROOT / "reports"

PROCESSED_ROOT: Final[Path] = REPO_ROOT / "data" / "processed" / "v1"
PROCESSED_SAMPLES_ROOT: Final[Path] = PROCESSED_ROOT / "samples"
PROCESSED_MANIFESTS_ROOT: Final[Path] = PROCESSED_ROOT / "manifests"
PROCESSED_REPORTS_ROOT: Final[Path] = PROCESSED_ROOT / "reports"

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
SPLIT_TO_KEYPOINT_ARCHIVE_FILE: Final[dict[str, str]] = {
    "train": "train_2D_keypoints.tar.zst",
    "val": "val_2D_keypoints.tar.zst",
    "test": "test_2D_keypoints.tar.zst",
}

MANIFESTS_AND_REPORTS_ARCHIVE_NAME: Final[str] = "sprint2_manifests_reports.tar.zst"
SAMPLE_ARCHIVE_NAME_TEMPLATE: Final[str] = "sprint2_samples_{split}.tar.zst"

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
REQUIRED_CORE_CHANNELS: Final[tuple[str, ...]] = ("body", "left_hand", "right_hand")
OPTIONAL_CHANNELS: Final[tuple[str, ...]] = ("face",)

CANVAS_WIDTH: Final[int] = 1280
CANVAS_HEIGHT: Final[int] = 720
PROCESSED_SCHEMA_VERSION: Final[str] = "t2sp-processed-v1"

COLAB_DRIVE_MOUNT_ROOT: Final[Path] = Path("/content/drive")
COLAB_DRIVE_PROJECT_ROOT: Final[Path] = (
    COLAB_DRIVE_MOUNT_ROOT / "MyDrive" / "text-to-sign-production"
)
COLAB_DRIVE_RAW_ROOT: Final[Path] = COLAB_DRIVE_PROJECT_ROOT / "raw" / "how2sign"
COLAB_DRIVE_TRANSLATIONS_DIR: Final[Path] = COLAB_DRIVE_RAW_ROOT / "translations"
COLAB_DRIVE_ARCHIVES_DIR: Final[Path] = COLAB_DRIVE_RAW_ROOT / "archives"
COLAB_DRIVE_ARTIFACTS_ROOT: Final[Path] = (
    COLAB_DRIVE_PROJECT_ROOT / "artifacts" / "sprint2" / "processed-v1"
)
COLAB_DOWNLOAD_ROOT: Final[Path] = Path("/content/how2sign_downloads")

SPLIT_TO_COLAB_DRIVE_TRANSLATION_PATH: Final[dict[str, Path]] = {
    split: COLAB_DRIVE_TRANSLATIONS_DIR / filename
    for split, filename in SPLIT_TO_TRANSLATION_FILE.items()
}
SPLIT_TO_COLAB_DRIVE_KEYPOINT_ARCHIVE_PATH: Final[dict[str, Path]] = {
    split: COLAB_DRIVE_ARCHIVES_DIR / filename
    for split, filename in SPLIT_TO_KEYPOINT_ARCHIVE_FILE.items()
}
