"""Manifest and report builders for pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from text_to_sign_production.data.constants import PROCESSED_SCHEMA_VERSION
from text_to_sign_production.data.schemas import NormalizedManifestEntry, RawManifestEntry


def raw_manifest_entry(
    *,
    split: str = "train",
    sample_id: str = "sample",
    keypoints_dir: str | None = "data/raw/how2sign/bfh_keypoints/train/sample",
    num_frames: int = 2,
) -> RawManifestEntry:
    return RawManifestEntry(
        sample_id=sample_id,
        source_split=split,
        video_id="video",
        video_name="video-rgb_front",
        sentence_id="sentence",
        sentence_name=sample_id,
        text="text",
        start_time=0.0,
        end_time=1.0,
        keypoints_dir=keypoints_dir,
        source_metadata_path="data/raw/how2sign/translations/train.tsv",
        has_face=True,
        num_frames=num_frames,
        source_video_path=f"data/raw/how2sign/bfh_keypoints/{split}/sample.mp4",
        video_width=1280,
        video_height=720,
        video_fps=24.0,
        video_metadata_error=None,
    )


def normalized_manifest_entry(
    *,
    split: str = "train",
    sample_id: str = "sample",
    sample_path: str | None = "data/processed/v1/samples/train/sample.npz",
    source_keypoints_dir: str | None = "data/raw/how2sign/bfh_keypoints/train/sample",
) -> NormalizedManifestEntry:
    return NormalizedManifestEntry(
        sample_id=sample_id,
        processed_schema_version=PROCESSED_SCHEMA_VERSION,
        text="text",
        split=split,
        start_time=0.0,
        end_time=1.0,
        num_frames=2,
        sample_path=sample_path,
        source_video_id="video",
        source_sentence_id="sentence",
        source_sentence_name=sample_id,
        source_metadata_path="data/raw/how2sign/translations/train.tsv",
        source_keypoints_dir=source_keypoints_dir,
        source_video_path=f"data/raw/how2sign/bfh_keypoints/{split}/sample.mp4",
        fps=24.0,
        video_width=1280,
        video_height=720,
        video_metadata_error=None,
        selected_person_index=0,
        multi_person_frame_count=0,
        max_people_per_frame=1,
        frame_valid_count=2,
        frame_invalid_count=0,
        face_missing_frame_count=0,
        out_of_bounds_coordinate_count=0,
        frames_with_any_zeroed_required_joint=0,
        frame_issue_counts={},
        core_channel_nonzero_frames={"body": 2, "left_hand": 2, "right_hand": 2},
        sample_parse_error=None,
    )


def processed_record(
    sample_path: str,
    *,
    split: str = "train",
    sample_id: str = "sample",
    text: str = "text",
    fps: float | None = 24.0,
    num_frames: int = 2,
    selected_person_index: int = 0,
    frame_valid_count: int | None = None,
    frame_invalid_count: int = 0,
) -> dict[str, Any]:
    resolved_frame_valid_count = num_frames if frame_valid_count is None else frame_valid_count
    return {
        "sample_id": sample_id,
        "processed_schema_version": PROCESSED_SCHEMA_VERSION,
        "text": text,
        "split": split,
        "fps": fps,
        "num_frames": num_frames,
        "sample_path": sample_path,
        "source_video_id": "video",
        "source_sentence_id": "sentence",
        "source_sentence_name": sample_id,
        "selected_person_index": selected_person_index,
        "multi_person_frame_count": 0,
        "max_people_per_frame": 1,
        "source_metadata_path": "data/raw/how2sign/translations/train.tsv",
        "source_keypoints_dir": "data/raw/how2sign/bfh_keypoints/train/sample",
        "source_video_path": "data/raw/how2sign/bfh_keypoints/train/sample.mp4",
        "video_width": 1280,
        "video_height": 720,
        "video_metadata_error": None,
        "frame_valid_count": resolved_frame_valid_count,
        "frame_invalid_count": frame_invalid_count,
        "face_missing_frame_count": 0,
        "out_of_bounds_coordinate_count": 0,
        "frames_with_any_zeroed_required_joint": 0,
        "frame_issue_counts": {},
        "core_channel_nonzero_frames": {
            "body": num_frames,
            "left_hand": num_frames,
            "right_hand": num_frames,
        },
    }


def assumption_report_for_splits(splits: tuple[str, ...]) -> dict[str, Any]:
    return {
        "generated_at": "2026-04-07T00:00:00+00:00",
        "splits": {
            split: {
                "generated_at": "2026-04-07T00:00:00+00:00",
                "raw_root": "data/raw/how2sign",
                "translation_columns": [],
                "translation_row_count": 1,
                "matched_sample_count": 1,
                "unmatched_sample_count": 0,
                "unmatched_examples": [],
                "video_metadata": {
                    "readable_count": 1,
                    "unreadable_count": 0,
                    "dimension_counts": {"1280x720": 1},
                    "fps_counts": {"24.0": 1},
                },
                "first_frame_people_counter": {"1": 1},
                "openpose_schema": {
                    "channel_lengths": {
                        "face_keypoints_2d": {"210": 1},
                        "hand_left_keypoints_2d": {"63": 1},
                        "hand_right_keypoints_2d": {"63": 1},
                        "pose_keypoints_2d": {"75": 1},
                    },
                    "deviation_counts": {},
                },
            }
            for split in splits
        },
    }


def filter_report_for_splits(splits: tuple[str, ...]) -> dict[str, Any]:
    return {
        "generated_at": "2026-04-07T00:00:00+00:00",
        "splits": {
            split: {
                "dropped_samples": 0,
                "drop_reason_counts": {},
            }
            for split in splits
        },
    }


def write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
