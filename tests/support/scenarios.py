"""Higher-level tiny workspace builders for stage and e2e tests."""

from __future__ import annotations

from pathlib import Path

from tests.support.builders.media import write_minimal_mp4
from tests.support.builders.openpose import person_payload, write_openpose_frame
from tests.support.builders.translations import translation_row, write_translation_file
from tests.support.paths import write_text
from text_to_sign_production.data.constants import (
    DEFAULT_FILTER_CONFIG_RELATIVE_PATH,
    SPLIT_TO_KEYPOINT_DIR,
    SPLIT_TO_TRANSLATION_FILE,
)


def write_filter_config(path: Path) -> None:
    write_text(
        path,
        "\n".join(
            [
                "schema_version: 2",
                "require_nonempty_text: true",
                "require_positive_duration: true",
                "require_keypoints_dir: true",
                "require_frames: true",
                "drop_on_sample_parse_error: true",
                "require_at_least_one_valid_frame: true",
                "minimum_nonzero_frames_per_core_channel: 1",
                "required_all_core_channels:",
                "  - body",
                "required_any_core_channel_groups:",
                "  - - left_hand",
                "    - right_hand",
            ]
        )
        + "\n",
    )


def write_legacy_filter_config(path: Path) -> None:
    write_text(
        path,
        "\n".join(
            [
                "schema_version: 1",
                "require_nonempty_text: true",
                "require_positive_duration: true",
                "require_keypoints_dir: true",
                "require_frames: true",
                "drop_on_sample_parse_error: true",
                "require_at_least_one_valid_frame: true",
                "minimum_nonzero_frames_per_core_channel: 1",
            ]
        )
        + "\n",
    )


def create_tiny_dataset_workspace(
    root: Path,
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> Path:
    translations_dir = root / "data/raw/how2sign/translations"
    bfh_root = root / "data/raw/how2sign/bfh_keypoints"
    write_filter_config(root / DEFAULT_FILTER_CONFIG_RELATIVE_PATH)
    write_legacy_filter_config(root / "configs/data/filter-v1.yaml")

    for split in splits:
        sentence_name = f"{split}_sample_0-1-rgb_front"
        rows = [translation_row(split=split, sentence_name=sentence_name)]
        if split == "train":
            rows.append(
                translation_row(
                    split=split,
                    sentence_name="train_unmatched_1-1-rgb_front",
                    sentence_index=1,
                    text="train unmatched text",
                    start_time="0.5",
                    end_time="1.0",
                )
            )
        write_translation_file(translations_dir / SPLIT_TO_TRANSLATION_FILE[split], rows)

        split_dir = SPLIT_TO_KEYPOINT_DIR[split]
        json_dir = bfh_root / split_dir / "openpose_output/json" / sentence_name
        video_dir = bfh_root / split_dir / "openpose_output/video"
        write_openpose_frame(
            json_dir / f"{sentence_name}_000000000000_keypoints.json",
            people=[person_payload()],
        )
        write_openpose_frame(
            json_dir / f"{sentence_name}_000000000001_keypoints.json",
            people=[person_payload()],
        )
        write_minimal_mp4(video_dir / f"{sentence_name}.mp4")
    return root
