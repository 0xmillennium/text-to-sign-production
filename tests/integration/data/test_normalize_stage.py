"""Normalization stage integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import text_to_sign_production.data.normalize as normalize_mod
import text_to_sign_production.data.raw as raw_mod

pytestmark = pytest.mark.integration


def test_normalize_all_splits_removes_stale_unrequested_split_outputs(
    tiny_dataset_workspace: Path,
) -> None:
    raw_mod.build_raw_manifests(splits=("train",))
    normalized_root = tiny_dataset_workspace / "data/interim/normalized_manifests"
    normalized_root.mkdir(parents=True, exist_ok=True)
    (normalized_root / "normalized_val.jsonl").write_text("stale\n", encoding="utf-8")
    (normalized_root / "normalized_test.jsonl").write_text("stale\n", encoding="utf-8")

    normalize_mod.normalize_all_splits(splits=("train",))

    assert (normalized_root / "normalized_train.jsonl").exists()
    assert not (normalized_root / "normalized_val.jsonl").exists()
    assert not (normalized_root / "normalized_test.jsonl").exists()
