"""Sprint 3 processed modeling manifest contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.support.builders.manifests import processed_record
from tests.support.modeling import (
    patch_modeling_repo_root,
    processed_sample_relative_path,
    write_processed_modeling_manifest,
    write_processed_modeling_sample,
)
from text_to_sign_production.modeling.data import (
    ProcessedModelingDataError,
    read_processed_modeling_manifest,
)

pytestmark = pytest.mark.unit


def test_manifest_reader_rejects_unknown_requested_split(tmp_path: Path) -> None:
    with pytest.raises(ProcessedModelingDataError, match="unknown split"):
        read_processed_modeling_manifest(tmp_path / "missing.jsonl", split="dev")


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("{broken\n", "line 1.*not valid JSON"),
        ("[]\n", "line 1.*JSON object"),
    ],
)
def test_manifest_reader_wraps_jsonl_line_errors_with_context(
    tmp_path: Path,
    content: str,
    match: str,
) -> None:
    manifest_path = tmp_path / "data/processed/v1/manifests/train.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(content, encoding="utf-8")

    with pytest.raises(ProcessedModelingDataError, match=match):
        read_processed_modeling_manifest(manifest_path, split="train")


@pytest.mark.parametrize(
    ("record", "split", "error_type", "match"),
    [
        (
            processed_record(processed_sample_relative_path("dev"), split="dev"),
            None,
            ProcessedModelingDataError,
            "unknown split",
        ),
        (
            processed_record(processed_sample_relative_path()),
            "val",
            ProcessedModelingDataError,
            "expected 'val'",
        ),
        (
            processed_record("/tmp/sample.npz"),
            "train",
            ProcessedModelingDataError,
            "invalid sample_path",
        ),
        (
            processed_record(processed_sample_relative_path()),
            "train",
            FileNotFoundError,
            "does not exist",
        ),
    ],
)
def test_manifest_reader_rejects_invalid_record_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record: dict[str, Any],
    split: str | None,
    error_type: type[Exception],
    match: str,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
    manifest_path = write_processed_modeling_manifest(tmp_path, [record])

    with pytest.raises(error_type, match=match):
        read_processed_modeling_manifest(manifest_path, split=split)


@pytest.mark.parametrize(
    ("records", "match"),
    [
        (
            [processed_record(processed_sample_relative_path(), sample_id=" sample ")],
            "whitespace in sample_id",
        ),
        (
            [
                processed_record(processed_sample_relative_path()),
                processed_record(processed_sample_relative_path()),
            ],
            "Duplicate processed sample_id",
        ),
    ],
)
def test_manifest_reader_rejects_sample_identity_violations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    records: list[dict[str, Any]],
    match: str,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
    write_processed_modeling_sample(tmp_path)
    manifest_path = write_processed_modeling_manifest(tmp_path, records)

    with pytest.raises(ProcessedModelingDataError, match=match):
        read_processed_modeling_manifest(manifest_path, split="train")
