"""View-sample CLI wrapper tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import scripts.view_sample as view_sample_script
import text_to_sign_production.data.utils as utils_mod
from tests.support.builders.manifests import processed_record
from tests.support.builders.samples import write_processed_sample_npz

pytestmark = pytest.mark.integration


def _write_train_manifest(manifest_root: Path, record: dict[str, object] | str) -> None:
    manifest_root.mkdir(parents=True, exist_ok=True)
    content = record if isinstance(record, str) else json.dumps(record)
    (manifest_root / "train.jsonl").write_text(content + "\n", encoding="utf-8")


def _set_view_sample_argv(monkeypatch: pytest.MonkeyPatch, sample_id: str = "sample") -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["view_sample.py", "--split", "train", "--sample-id", sample_id],
    )


def test_view_sample_rejects_invalid_processed_sample_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_root = tmp_path / "data/processed/v1/manifests"
    _write_train_manifest(
        manifest_root,
        processed_record("data/processed/v1/samples/train/sample.txt"),
    )
    monkeypatch.setattr(view_sample_script, "PROCESSED_MANIFESTS_ROOT", manifest_root)
    _set_view_sample_argv(monkeypatch)

    assert view_sample_script.main() == 1


def test_view_sample_rejects_processed_sample_path_outside_processed_samples_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_root = tmp_path / "data/processed/v1/manifests"
    write_processed_sample_npz(tmp_path / "data/interim/sample.npz")
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    _write_train_manifest(manifest_root, processed_record("data/interim/sample.npz"))
    monkeypatch.setattr(view_sample_script, "PROCESSED_MANIFESTS_ROOT", manifest_root)
    _set_view_sample_argv(monkeypatch)

    assert view_sample_script.main() == 1


def test_view_sample_rejects_unreadable_processed_sample_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_root = tmp_path / "data/processed/v1/manifests"
    sample_path = tmp_path / "data/processed/v1/samples/train/sample.npz"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text("not-an-npz", encoding="utf-8")
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    _write_train_manifest(
        manifest_root,
        processed_record("data/processed/v1/samples/train/sample.npz"),
    )
    monkeypatch.setattr(view_sample_script, "PROCESSED_MANIFESTS_ROOT", manifest_root)
    _set_view_sample_argv(monkeypatch)

    assert view_sample_script.main() == 1


def test_view_sample_reports_missing_sample_id_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_root = tmp_path / "data/processed/v1/manifests"
    _write_train_manifest(
        manifest_root,
        processed_record("data/processed/v1/samples/train/sample.npz"),
    )
    monkeypatch.setattr(view_sample_script, "PROCESSED_MANIFESTS_ROOT", manifest_root)
    _set_view_sample_argv(monkeypatch, sample_id="missing")

    assert view_sample_script.main() == 1


def test_view_sample_reports_malformed_manifest_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_root = tmp_path / "data/processed/v1/manifests"
    _write_train_manifest(manifest_root, "{broken")
    monkeypatch.setattr(view_sample_script, "PROCESSED_MANIFESTS_ROOT", manifest_root)
    _set_view_sample_argv(monkeypatch)

    assert view_sample_script.main() == 1
