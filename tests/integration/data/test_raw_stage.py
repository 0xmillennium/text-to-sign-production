"""Raw manifest stage integration tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import text_to_sign_production.data.raw as raw_mod
from tests.support.assertions import assert_jsonl_record_count
from tests.support.builders.openpose import person_payload, write_openpose_frame

pytestmark = pytest.mark.integration


def test_build_raw_manifest_for_split_writes_expected_manifest(
    tiny_dataset_workspace: Path,
) -> None:
    raw_mod.build_raw_manifests(splits=("train",))

    raw_records = assert_jsonl_record_count(
        tiny_dataset_workspace / "data/interim/raw_manifests/raw_train.jsonl",
        2,
    )
    assert raw_records[0]["sample_id"] == "train_sample_0-1-rgb_front"
    assert raw_records[0]["num_frames"] == 2
    assert raw_records[0]["video_fps"] == 24.0


def test_build_raw_manifest_for_split_uses_deterministic_first_frame_without_sorting(
    tiny_dataset_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_paths = raw_mod.get_split_paths("train")
    sentence_name = "train_sample_0-1-rgb_front"
    keypoints_dir = split_paths.keypoints_json_root / sentence_name
    later_frame = keypoints_dir / f"{sentence_name}_000000000001_keypoints.json"
    earlier_frame = keypoints_dir / f"{sentence_name}_000000000000_keypoints.json"
    write_openpose_frame(later_frame, people=[person_payload()])
    write_openpose_frame(
        earlier_frame,
        people=[person_payload(), person_payload()],
    )

    original_glob = keypoints_dir.__class__.glob

    def fake_glob(self: Path, pattern: str) -> Iterator[Path]:
        if self == keypoints_dir and pattern == "*.json":
            return iter([later_frame, earlier_frame])
        return original_glob(self, pattern)

    monkeypatch.setattr(keypoints_dir.__class__, "glob", fake_glob)

    manifest_entries, report = raw_mod.build_raw_manifest_for_split("train")

    matched_entry = next(entry for entry in manifest_entries if entry.sample_id == sentence_name)
    assert matched_entry.num_frames == 2
    assert report["first_frame_people_counter"] == {"2": 1}


def test_read_translation_rows_reports_empty_files_before_delimiter_errors(
    tmp_path: Path,
) -> None:
    translation_path = tmp_path / "translations.csv"
    translation_path.write_text("", encoding="utf-8")
    keypoints_root = tmp_path / "json"
    video_root = tmp_path / "video"
    keypoints_root.mkdir(parents=True)
    video_root.mkdir(parents=True)

    with pytest.raises(ValueError, match="is empty"):
        raw_mod.read_translation_rows(
            raw_mod.SplitPaths(
                split="train",
                translation_path=translation_path,
                keypoints_json_root=keypoints_root,
                video_root=video_root,
            )
        )


def test_build_raw_manifests_removes_stale_unrequested_split_outputs(
    tiny_dataset_workspace: Path,
) -> None:
    raw_root = tiny_dataset_workspace / "data/interim/raw_manifests"
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "raw_val.jsonl").write_text("stale\n", encoding="utf-8")
    (raw_root / "raw_test.jsonl").write_text("stale\n", encoding="utf-8")

    raw_mod.build_raw_manifests(splits=("train",))

    assert (raw_root / "raw_train.jsonl").exists()
    assert not (raw_root / "raw_val.jsonl").exists()
    assert not (raw_root / "raw_test.jsonl").exists()
