"""Shared assertions for manifests, samples, and reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from text_to_sign_production.data.constants import PROCESSED_SCHEMA_VERSION

REQUIRED_PROCESSED_SAMPLE_KEYS = {
    "processed_schema_version",
    "body",
    "body_confidence",
    "left_hand",
    "left_hand_confidence",
    "right_hand",
    "right_hand_confidence",
    "face",
    "face_confidence",
    "people_per_frame",
    "selected_person_index",
    "frame_valid_mask",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def assert_jsonl_record_count(path: Path, expected_count: int) -> list[dict[str, Any]]:
    records = load_jsonl(path)
    assert len(records) == expected_count
    return records


def assert_processed_sample_payload(path: Path, *, expected_num_frames: int = 2) -> None:
    assert path.exists()
    with np.load(path) as sample:
        assert set(sample.files) >= REQUIRED_PROCESSED_SAMPLE_KEYS
        assert str(sample["processed_schema_version"]) == PROCESSED_SCHEMA_VERSION
        assert sample["body"].shape == (expected_num_frames, 25, 2)


def assert_report_exists(root: Path, relative_path: str) -> None:
    assert (root / relative_path).exists()
