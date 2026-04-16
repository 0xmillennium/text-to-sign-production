"""Local end-to-end Dataset Build workflow test."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

import scripts.view_sample as view_sample_script
import text_to_sign_production.ops.colab_workflow as colab_workflow_mod
import text_to_sign_production.workflows.dataset_build as dataset_build_workflow_mod
from tests.support.assertions import (
    assert_jsonl_record_count,
    assert_processed_sample_payload,
    assert_report_exists,
    load_jsonl,
)
from tests.support.builders.archives import list_tar_zst_members
from tests.support.builders.media import write_minimal_mp4
from tests.support.builders.openpose import build_channel, person_payload, write_openpose_frame
from tests.support.builders.translations import translation_row, write_translation_file
from text_to_sign_production.data.constants import (
    PROCESSED_SCHEMA_VERSION,
    SPLIT_TO_KEYPOINT_DIR,
    SPLIT_TO_TRANSLATION_FILE,
)

pytestmark = pytest.mark.e2e

requires_tar_and_zstd = pytest.mark.skipif(
    shutil.which("tar") is None or shutil.which("zstd") is None,
    reason="tar and zstd are required for archive tests",
)


def test_run_dataset_build_end_to_end(
    tiny_dataset_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside_dir = tiny_dataset_workspace / "outside"
    outside_dir.mkdir()
    monkeypatch.chdir(outside_dir)

    result = dataset_build_workflow_mod.run_dataset_build(
        splits=("train", "val", "test"),
        filter_config_path=tiny_dataset_workspace / "configs/data/filter-v1.yaml",
        output_mode="none",
    )

    assert result.splits == ("train", "val", "test")
    assert result.output_paths == ()

    raw_manifest = tiny_dataset_workspace / "data/interim/raw_manifests/raw_train.jsonl"
    raw_records = assert_jsonl_record_count(raw_manifest, 2)
    assert raw_records[0]["sample_id"] == "train_sample_0-1-rgb_front"
    assert raw_records[0]["num_frames"] == 2
    assert raw_records[0]["video_fps"] == 24.0

    train_manifest = tiny_dataset_workspace / "data/processed/v1/manifests/train.jsonl"
    processed_records = assert_jsonl_record_count(train_manifest, 1)
    record = processed_records[0]
    assert record["processed_schema_version"] == PROCESSED_SCHEMA_VERSION
    assert record["selected_person_index"] == 0
    assert record["multi_person_frame_count"] == 0

    sample_path = tiny_dataset_workspace / record["sample_path"]
    assert_processed_sample_payload(sample_path, expected_num_frames=2)

    filter_report = json.loads(
        (tiny_dataset_workspace / "data/interim/reports/filter-report.json").read_text(
            encoding="utf-8"
        )
    )
    assert filter_report["config_path"] == "configs/data/filter-v1.yaml"
    assert filter_report["splits"]["train"]["dropped_samples"] == 1
    assert filter_report["splits"]["train"]["drop_reason_counts"]["missing_keypoints_dir"] == 1

    assumption_report = json.loads(
        (tiny_dataset_workspace / "data/interim/reports/assumption-report.json").read_text(
            encoding="utf-8"
        )
    )
    assert assumption_report["splits"]["train"]["matched_sample_count"] == 1
    assert assumption_report["splits"]["train"]["unmatched_sample_count"] == 1

    monkeypatch.setattr(
        sys,
        "argv",
        ["view_sample.py", "--split", "train", "--sample-id", record["sample_id"]],
    )
    assert view_sample_script.main() == 0

    assert_report_exists(tiny_dataset_workspace, "data/processed/v1/reports/assumption-report.md")
    assert_report_exists(tiny_dataset_workspace, "data/processed/v1/reports/data-quality-report.md")
    assert_report_exists(tiny_dataset_workspace, "data/processed/v1/reports/split-report.md")


@requires_tar_and_zstd
def test_dataset_build_sample_archive_matches_processed_manifest_after_filtering(
    tiny_dataset_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dropped_sample_id = "train_dropped_left_hand_0-1-rgb_front"
    _add_train_sample_with_unusable_left_hand(tiny_dataset_workspace, dropped_sample_id)

    outside_dir = tiny_dataset_workspace / "outside"
    outside_dir.mkdir()
    monkeypatch.chdir(outside_dir)

    result = dataset_build_workflow_mod.run_dataset_build(
        splits=("train",),
        filter_config_path=tiny_dataset_workspace / "configs/data/filter-v1.yaml",
        output_mode="none",
    )

    train_filter_report = result.filter_report["splits"]["train"]
    assert train_filter_report["drop_reason_counts"]["unusable_core_channel:left_hand"] >= 1
    assert any(
        example["sample_id"] == dropped_sample_id
        and "unusable_core_channel:left_hand" in example["drop_reasons"]
        for example in train_filter_report["dropped_examples"]
    )

    normalized_records = load_jsonl(
        tiny_dataset_workspace / "data/interim/normalized_manifests/normalized_train.jsonl"
    )
    normalized_dropped = next(
        record for record in normalized_records if record["sample_id"] == dropped_sample_id
    )
    assert normalized_dropped["sample_path"] == (
        f"data/processed/v1/samples/train/{dropped_sample_id}.npz"
    )
    assert (tiny_dataset_workspace / normalized_dropped["sample_path"]).is_file()

    filtered_records = load_jsonl(
        tiny_dataset_workspace / "data/interim/filtered_manifests/filtered_train.jsonl"
    )
    assert dropped_sample_id not in {record["sample_id"] for record in filtered_records}

    processed_manifest = tiny_dataset_workspace / "data/processed/v1/manifests/train.jsonl"
    processed_records = load_jsonl(processed_manifest)
    processed_sample_paths = {record["sample_path"] for record in processed_records}
    assert normalized_dropped["sample_path"] not in processed_sample_paths

    colab_workflow_mod.package_dataset_build_outputs(
        project_root=tiny_dataset_workspace,
        splits=("train",),
    )

    archive_members = {
        member
        for member in list_tar_zst_members(
            tiny_dataset_workspace / "data/archives/dataset_build_samples_train.tar.zst"
        ).splitlines()
        if member.endswith(".npz")
    }
    assert archive_members == processed_sample_paths
    assert normalized_dropped["sample_path"] not in archive_members


def _add_train_sample_with_unusable_left_hand(root: Path, sample_id: str) -> None:
    write_translation_file(
        root / "data/raw/how2sign/translations" / SPLIT_TO_TRANSLATION_FILE["train"],
        [
            translation_row(split="train", sentence_name="train_sample_0-1-rgb_front"),
            translation_row(
                split="train",
                sentence_name="train_unmatched_1-1-rgb_front",
                sentence_index=1,
                text="train unmatched text",
                start_time="0.5",
                end_time="1.0",
            ),
            translation_row(
                split="train",
                sentence_name=sample_id,
                sentence_index=2,
                text="train sample with unusable left hand",
                start_time="1.0",
                end_time="1.5",
            ),
        ],
    )

    dropped_person = person_payload()
    dropped_person["hand_left_keypoints_2d"] = build_channel(
        21,
        x_offset=400.0,
        y_offset=350.0,
        confidence=0.0,
    )
    keypoints_root = (
        root
        / "data/raw/how2sign/bfh_keypoints"
        / SPLIT_TO_KEYPOINT_DIR["train"]
        / "openpose_output"
    )
    json_dir = keypoints_root / "json" / sample_id
    for frame_index in range(2):
        write_openpose_frame(
            json_dir / f"{sample_id}_{frame_index:012d}_keypoints.json",
            people=[dropped_person],
        )
    write_minimal_mp4(keypoints_root / "video" / f"{sample_id}.mp4")
