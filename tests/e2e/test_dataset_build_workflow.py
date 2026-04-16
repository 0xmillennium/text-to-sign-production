"""Local end-to-end Dataset Build workflow test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import scripts.view_sample as view_sample_script
import text_to_sign_production.workflows.dataset_build as dataset_build_workflow_mod
from tests.support.assertions import (
    assert_jsonl_record_count,
    assert_processed_sample_payload,
    assert_report_exists,
)
from text_to_sign_production.data.constants import PROCESSED_SCHEMA_VERSION

pytestmark = pytest.mark.e2e


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
