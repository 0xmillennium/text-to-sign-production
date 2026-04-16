"""Report and manifest export validation tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import text_to_sign_production.data.manifests as manifests_mod
import text_to_sign_production.data.reports as reports_mod
import text_to_sign_production.data.utils as utils_mod
from tests.support.builders.manifests import (
    assumption_report_for_splits,
    filter_report_for_splits,
    normalized_manifest_entry,
)
from tests.support.builders.samples import write_processed_sample_npz

pytestmark = pytest.mark.unit


def _patch_manifest_roots(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(
        manifests_mod, "PROCESSED_MANIFESTS_ROOT", root / "data/processed/manifests"
    )
    monkeypatch.setattr(manifests_mod, "RAW_MANIFESTS_ROOT", root / "data/interim/raw_manifests")
    monkeypatch.setattr(
        manifests_mod,
        "FILTERED_MANIFESTS_ROOT",
        root / "data/interim/filtered_manifests",
    )
    monkeypatch.setattr(manifests_mod, "PROCESSED_REPORTS_ROOT", root / "data/processed/reports")
    monkeypatch.setattr(reports_mod, "PROCESSED_REPORTS_ROOT", root / "data/processed/reports")
    monkeypatch.setattr(manifests_mod, "_load_raw_records", lambda split: [])


def test_export_final_manifests_rejects_missing_source_keypoints_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_manifest_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(
        manifests_mod,
        "_load_filtered_records",
        lambda split: [normalized_manifest_entry(split=split, source_keypoints_dir=None)],
    )

    with pytest.raises(ValueError, match="missing source_keypoints_dir"):
        manifests_mod.export_final_manifests(
            assumption_report=assumption_report_for_splits(("train",)),
            filter_report=filter_report_for_splits(("train",)),
            splits=("train",),
        )


def test_export_final_manifests_rejects_missing_sample_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_manifest_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(
        manifests_mod,
        "_load_filtered_records",
        lambda split: [normalized_manifest_entry(split=split, sample_path=None)],
    )

    with pytest.raises(ValueError, match="missing sample_path"):
        manifests_mod.export_final_manifests(
            assumption_report=assumption_report_for_splits(("train",)),
            filter_report=filter_report_for_splits(("train",)),
            splits=("train",),
        )


def test_export_final_manifests_rejects_sample_paths_outside_processed_samples_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_manifest_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    write_processed_sample_npz(tmp_path / "data/interim/sample.npz")
    monkeypatch.setattr(
        manifests_mod,
        "_load_filtered_records",
        lambda split: [
            normalized_manifest_entry(
                split=split,
                sample_path="data/interim/sample.npz",
            )
        ],
    )

    with pytest.raises(ValueError, match="data/processed/v1/samples"):
        manifests_mod.export_final_manifests(
            assumption_report=assumption_report_for_splits(("train",)),
            filter_report=filter_report_for_splits(("train",)),
            splits=("train",),
        )


def test_export_final_manifests_rejects_missing_requested_report_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_manifest_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(manifests_mod, "_load_filtered_records", lambda split: [])

    with pytest.raises(ValueError, match="assumption_report is missing requested splits: train"):
        manifests_mod.export_final_manifests(
            assumption_report={"generated_at": "2026-04-07T00:00:00+00:00", "splits": {}},
            filter_report=filter_report_for_splits(("train",)),
            splits=("train",),
        )

    with pytest.raises(ValueError, match="filter_report is missing requested splits: train"):
        manifests_mod.export_final_manifests(
            assumption_report=assumption_report_for_splits(("train",)),
            filter_report={"generated_at": "2026-04-07T00:00:00+00:00", "splits": {}},
            splits=("train",),
        )


@pytest.mark.parametrize(
    "report",
    [
        {"generated_at": "2026-04-07T00:00:00+00:00"},
        {"generated_at": "2026-04-07T00:00:00+00:00", "splits": None},
        {"generated_at": "2026-04-07T00:00:00+00:00", "splits": []},
    ],
)
def test_validate_report_splits_rejects_missing_or_invalid_splits_mapping(
    report: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="filter_report is missing a splits mapping"):
        manifests_mod._validate_report_splits("filter_report", report, ("train",))


def test_build_quality_report_rejects_missing_final_records_split() -> None:
    with pytest.raises(
        ValueError,
        match="final_records_by_split is missing requested split: train",
    ):
        reports_mod.build_quality_report(
            final_records_by_split={},
            filter_report=filter_report_for_splits(("train",)),
            generated_at="2026-04-07T00:00:00+00:00",
            splits=("train",),
        )


@pytest.mark.parametrize(
    ("filter_report", "message"),
    [
        (
            {"generated_at": "2026-04-07T00:00:00+00:00"},
            "filter_report is missing a splits mapping",
        ),
        (
            {"generated_at": "2026-04-07T00:00:00+00:00", "splits": {}},
            "filter_report is missing requested split: train",
        ),
    ],
)
def test_build_quality_report_rejects_missing_filter_report_split_mapping(
    filter_report: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        reports_mod.build_quality_report(
            final_records_by_split={"train": []},
            filter_report=filter_report,
            generated_at="2026-04-07T00:00:00+00:00",
            splits=("train",),
        )


def test_build_split_report_rejects_missing_requested_split_records() -> None:
    with pytest.raises(
        ValueError,
        match="raw_records_by_split is missing requested split: train",
    ):
        reports_mod.build_split_report(
            raw_records_by_split={},
            final_records_by_split={"train": []},
            generated_at="2026-04-07T00:00:00+00:00",
            splits=("train",),
        )

    with pytest.raises(
        ValueError,
        match="final_records_by_split is missing requested split: train",
    ):
        reports_mod.build_split_report(
            raw_records_by_split={"train": []},
            final_records_by_split={},
            generated_at="2026-04-07T00:00:00+00:00",
            splits=("train",),
        )
