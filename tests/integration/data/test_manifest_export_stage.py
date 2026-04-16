"""Processed manifest export stage integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import text_to_sign_production.data.manifests as manifests_mod
from tests.support.builders.manifests import (
    assumption_report_for_splits,
    filter_report_for_splits,
    normalized_manifest_entry,
    raw_manifest_entry,
)
from tests.support.builders.samples import write_processed_sample_npz

pytestmark = pytest.mark.integration


def test_export_final_manifests_supports_subset_splits(
    patched_dataset_roots: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_manifest_root = patched_dataset_roots / "data/processed/v1/manifests"
    processed_manifest_root.mkdir(parents=True, exist_ok=True)
    (processed_manifest_root / "val.jsonl").write_text("stale\n", encoding="utf-8")
    (processed_manifest_root / "test.jsonl").write_text("stale\n", encoding="utf-8")
    write_processed_sample_npz(patched_dataset_roots / "data/processed/v1/samples/train/sample.npz")
    monkeypatch.setattr(
        manifests_mod,
        "_load_raw_records",
        lambda split: [raw_manifest_entry(split=split)] if split == "train" else [],
    )
    monkeypatch.setattr(
        manifests_mod,
        "_load_filtered_records",
        lambda split: [normalized_manifest_entry(split=split)] if split == "train" else [],
    )

    result = manifests_mod.export_final_manifests(
        assumption_report=assumption_report_for_splits(("train",)),
        filter_report=filter_report_for_splits(("train",)),
        splits=("train",),
    )

    assert result["split_report"]["official_split_names"] == ["train"]
    assert result["split_report"]["video_id_overlap"] == {}
    assert set(result["split_report"]["splits"]) == {"train"}
    assert result["split_report"]["splits"]["train"]["sample_id_overlap_with_other_splits"] == {}
    assert (processed_manifest_root / "train.jsonl").exists()
    assert not (processed_manifest_root / "val.jsonl").exists()
    assert not (processed_manifest_root / "test.jsonl").exists()
