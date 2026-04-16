"""Filtering stage integration tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import text_to_sign_production.data.filtering as filtering_mod
import text_to_sign_production.data.normalize as normalize_mod
import text_to_sign_production.data.raw as raw_mod
from tests.support.assertions import load_jsonl
from tests.support.builders.manifests import normalized_manifest_entry, write_jsonl_records
from text_to_sign_production.data.constants import DEFAULT_FILTER_CONFIG_RELATIVE_PATH

pytestmark = pytest.mark.integration


def test_filter_all_splits_removes_stale_unrequested_split_outputs(
    tiny_dataset_workspace: Path,
) -> None:
    raw_mod.build_raw_manifests(splits=("train",))
    normalize_mod.normalize_all_splits(splits=("train",))
    filtered_root = tiny_dataset_workspace / "data/interim/filtered_manifests"
    filtered_root.mkdir(parents=True, exist_ok=True)
    (filtered_root / "filtered_val.jsonl").write_text("stale\n", encoding="utf-8")
    (filtered_root / "filtered_test.jsonl").write_text("stale\n", encoding="utf-8")

    filtering_mod.filter_all_splits(
        tiny_dataset_workspace / DEFAULT_FILTER_CONFIG_RELATIVE_PATH,
        splits=("train",),
    )

    assert (filtered_root / "filtered_train.jsonl").exists()
    assert not (filtered_root / "filtered_val.jsonl").exists()
    assert not (filtered_root / "filtered_test.jsonl").exists()


def test_filter_all_splits_applies_body_and_at_least_one_hand_policy(
    tiny_dataset_workspace: Path,
) -> None:
    normalized_root = tiny_dataset_workspace / "data/interim/normalized_manifests"
    write_jsonl_records(
        normalized_root / "normalized_train.jsonl",
        [
            _normalized_record(
                "keep_right_hand",
                body=2,
                left_hand=0,
                right_hand=2,
            ),
            _normalized_record(
                "keep_left_hand",
                body=2,
                left_hand=2,
                right_hand=0,
            ),
            _normalized_record(
                "drop_both_hands",
                body=2,
                left_hand=0,
                right_hand=0,
            ),
            _normalized_record(
                "drop_body",
                body=0,
                left_hand=0,
                right_hand=2,
            ),
        ],
    )

    report = filtering_mod.filter_all_splits(
        tiny_dataset_workspace / DEFAULT_FILTER_CONFIG_RELATIVE_PATH,
        splits=("train",),
    )

    filtered_records = load_jsonl(
        tiny_dataset_workspace / "data/interim/filtered_manifests/filtered_train.jsonl"
    )
    assert [record["sample_id"] for record in filtered_records] == [
        "keep_right_hand",
        "keep_left_hand",
    ]
    assert report["splits"]["train"]["drop_reason_counts"] == {
        "unusable_core_channel:body": 1,
        "unusable_core_channel_group:left_hand|right_hand": 1,
    }


def _normalized_record(
    sample_id: str,
    *,
    body: int,
    left_hand: int,
    right_hand: int,
) -> dict[str, Any]:
    entry = normalized_manifest_entry(sample_id=sample_id)
    entry.core_channel_nonzero_frames = {
        "body": body,
        "left_hand": left_hand,
        "right_hand": right_hand,
    }
    return entry.to_record()
