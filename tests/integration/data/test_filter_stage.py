"""Filtering stage integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import text_to_sign_production.data.filtering as filtering_mod
import text_to_sign_production.data.normalize as normalize_mod
import text_to_sign_production.data.raw as raw_mod

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
        tiny_dataset_workspace / "configs/data/filter-v1.yaml",
        splits=("train",),
    )

    assert (filtered_root / "filtered_train.jsonl").exists()
    assert not (filtered_root / "filtered_val.jsonl").exists()
    assert not (filtered_root / "filtered_test.jsonl").exists()
