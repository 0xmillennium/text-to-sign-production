"""Local end-to-end Dataset Build CLI test."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import scripts.dataset_build as dataset_build_script
from tests.support.assertions import assert_jsonl_record_count
from text_to_sign_production.data.constants import (
    DEFAULT_FILTER_CONFIG_RELATIVE_PATH,
    PROCESSED_SCHEMA_VERSION,
)

pytestmark = pytest.mark.e2e


def test_dataset_build_cli_end_to_end(
    tiny_dataset_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside_dir = tiny_dataset_workspace / "outside"
    outside_dir.mkdir()
    monkeypatch.chdir(outside_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset_build.py",
            "--config",
            str(tiny_dataset_workspace / DEFAULT_FILTER_CONFIG_RELATIVE_PATH),
            "--no-package",
        ],
    )

    assert dataset_build_script.main() == 0

    train_manifest = tiny_dataset_workspace / "data/processed/v1/manifests/train.jsonl"
    processed_records = assert_jsonl_record_count(train_manifest, 1)
    assert processed_records[0]["processed_schema_version"] == PROCESSED_SCHEMA_VERSION
