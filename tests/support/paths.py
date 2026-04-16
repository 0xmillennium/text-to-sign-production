"""Path and filesystem helpers shared by tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import scripts.view_sample as view_sample_script
import text_to_sign_production.data.filtering as filtering_mod
import text_to_sign_production.data.manifests as manifests_mod
import text_to_sign_production.data.normalize as normalize_mod
import text_to_sign_production.data.raw as raw_mod
import text_to_sign_production.data.reports as reports_mod
import text_to_sign_production.data.utils as utils_mod


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def patch_dataset_paths(monkeypatch: Any, root: Path) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", root)
    monkeypatch.setattr(raw_mod, "TRANSLATIONS_DIR", root / "data/raw/how2sign/translations")
    monkeypatch.setattr(raw_mod, "BFH_KEYPOINTS_ROOT", root / "data/raw/how2sign/bfh_keypoints")
    monkeypatch.setattr(raw_mod, "RAW_MANIFESTS_ROOT", root / "data/interim/raw_manifests")
    monkeypatch.setattr(raw_mod, "INTERIM_REPORTS_ROOT", root / "data/interim/reports")
    monkeypatch.setattr(normalize_mod, "RAW_MANIFESTS_ROOT", root / "data/interim/raw_manifests")
    monkeypatch.setattr(
        normalize_mod,
        "NORMALIZED_MANIFESTS_ROOT",
        root / "data/interim/normalized_manifests",
    )
    monkeypatch.setattr(
        normalize_mod,
        "PROCESSED_SAMPLES_ROOT",
        root / "data/processed/v1/samples",
    )
    monkeypatch.setattr(
        filtering_mod,
        "NORMALIZED_MANIFESTS_ROOT",
        root / "data/interim/normalized_manifests",
    )
    monkeypatch.setattr(
        filtering_mod,
        "FILTERED_MANIFESTS_ROOT",
        root / "data/interim/filtered_manifests",
    )
    monkeypatch.setattr(filtering_mod, "INTERIM_REPORTS_ROOT", root / "data/interim/reports")
    monkeypatch.setattr(manifests_mod, "RAW_MANIFESTS_ROOT", root / "data/interim/raw_manifests")
    monkeypatch.setattr(
        manifests_mod,
        "FILTERED_MANIFESTS_ROOT",
        root / "data/interim/filtered_manifests",
    )
    monkeypatch.setattr(
        manifests_mod,
        "PROCESSED_MANIFESTS_ROOT",
        root / "data/processed/v1/manifests",
    )
    monkeypatch.setattr(
        manifests_mod,
        "PROCESSED_REPORTS_ROOT",
        root / "data/processed/v1/reports",
    )
    monkeypatch.setattr(reports_mod, "PROCESSED_REPORTS_ROOT", root / "data/processed/v1/reports")
    monkeypatch.setattr(
        view_sample_script,
        "PROCESSED_MANIFESTS_ROOT",
        root / "data/processed/v1/manifests",
    )
