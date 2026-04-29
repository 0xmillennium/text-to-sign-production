"""Reusable tiny visual-debug workspaces and pose payloads for tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tests.support.builders.manifests import (
    normalized_manifest_entry,
    processed_record,
    write_jsonl_records,
)
from tests.support.builders.media import write_tiny_decodable_mp4
from tests.support.builders.samples import write_processed_sample_npz


def renderable_pose_overrides(
    *,
    num_frames: int = 3,
    frame_valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return pose arrays with visible BODY_25, HAND_21, and face features."""

    body = np.zeros((num_frames, 25, 2), dtype=np.float32)
    body_confidence = np.zeros((num_frames, 25), dtype=np.float32)
    body[:, 1, :] = [0.5, 0.2]
    body[:, 8, :] = [0.5, 0.5]
    body_confidence[:, [1, 8]] = 1.0

    left_hand = np.zeros((num_frames, 21, 2), dtype=np.float32)
    left_hand_confidence = np.zeros((num_frames, 21), dtype=np.float32)
    left_hand[:, 0, :] = [0.25, 0.5]
    left_hand[:, 1, :] = [0.3, 0.5]
    left_hand_confidence[:, [0, 1]] = 1.0

    right_hand = np.zeros((num_frames, 21, 2), dtype=np.float32)
    right_hand_confidence = np.zeros((num_frames, 21), dtype=np.float32)
    right_hand[:, 0, :] = [0.75, 0.5]
    right_hand[:, 1, :] = [0.7, 0.5]
    right_hand_confidence[:, [0, 1]] = 1.0

    face = np.zeros((num_frames, 70, 2), dtype=np.float32)
    face_confidence = np.zeros((num_frames, 70), dtype=np.float32)
    face[:, 0, :] = [0.1, 0.1]
    face_confidence[:, 0] = 1.0

    resolved_frame_valid_mask = (
        np.ones((num_frames,), dtype=np.bool_)
        if frame_valid_mask is None
        else frame_valid_mask.astype(np.bool_)
    )
    return {
        "body": body,
        "body_confidence": body_confidence,
        "left_hand": left_hand,
        "left_hand_confidence": left_hand_confidence,
        "right_hand": right_hand,
        "right_hand_confidence": right_hand_confidence,
        "face": face,
        "face_confidence": face_confidence,
        "people_per_frame": np.ones((num_frames,), dtype=np.int16),
        "selected_person_index": np.asarray(0, dtype=np.int16),
        "frame_valid_mask": resolved_frame_valid_mask,
    }


def write_visual_debug_workspace(
    root: Path,
    *,
    splits: tuple[str, ...] = ("val",),
    samples_per_split: int = 2,
    duplicate_sample_id: str | None = None,
    with_source_videos: bool = False,
) -> Path:
    """Write a tiny processed/interim visual-debug workspace under `root/data`."""

    data_root = root / "data"
    for split in splits:
        records: list[dict[str, Any]] = []
        for index in range(samples_per_split):
            sample_id = duplicate_sample_id or f"{split}-sample-{index}"
            sample_path = f"data/processed/v1/samples/{split}/{sample_id}.npz"
            write_processed_sample_npz(
                data_root / f"processed/v1/samples/{split}/{sample_id}.npz",
                num_frames=3,
                overrides=renderable_pose_overrides(num_frames=3),
            )
            record = processed_record(
                sample_path,
                split=split,
                sample_id=sample_id,
                text=f"{split} text {index}",
                num_frames=3,
            )
            record.update(
                {
                    "source_sentence_id": f"{split}-sentence-{index}",
                    "source_sentence_name": f"{split}-sentence-name-{index}",
                    "source_video_id": f"{split}-video-{index}",
                    "source_video_path": (
                        f"data/raw/how2sign/bfh_keypoints/{split}_2D_keypoints/"
                        f"openpose_output/video/{sample_id}.mp4"
                    ),
                }
            )
            records.append(record)
            if with_source_videos:
                write_tiny_decodable_mp4(
                    data_root
                    / "raw/how2sign/bfh_keypoints"
                    / f"{split}_2D_keypoints/openpose_output/video/{sample_id}.mp4"
                )
        write_jsonl_records(data_root / f"processed/v1/manifests/{split}.jsonl", records)
        filtered_entry = normalized_manifest_entry(
            split=split,
            sample_id=str(records[0]["sample_id"]),
            sample_path=str(records[0]["sample_path"]),
            source_keypoints_dir=str(records[0]["source_keypoints_dir"]),
        )
        filtered_entry.start_time = 1.25
        filtered_entry.end_time = 2.5
        filtered_entry.fps = 30.0
        filtered_entry.num_frames = 3
        filtered_entry.source_sentence_id = str(records[0]["source_sentence_id"])
        filtered_entry.source_sentence_name = str(records[0]["source_sentence_name"])
        filtered_entry.source_video_id = str(records[0]["source_video_id"])
        filtered_entry.source_video_path = str(records[0]["source_video_path"])
        filtered_entry.video_width = 1280
        filtered_entry.video_height = 720
        write_jsonl_records(
            data_root / f"interim/filtered_manifests/filtered_{split}.jsonl",
            [filtered_entry.to_record()],
        )
        write_jsonl_records(
            data_root / f"interim/normalized_manifests/normalized_{split}.jsonl",
            [],
        )
        write_jsonl_records(data_root / f"interim/raw_manifests/raw_{split}.jsonl", [])
    return data_root
