"""Validate-manifest CLI wrapper tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import scripts.validate_manifest as validate_manifest_script

pytestmark = pytest.mark.integration


def test_validate_manifest_cli_accepts_static_raw_manifest(
    fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_manifest.py",
            "--manifest",
            str(fixtures_dir / "manifests/raw_train.jsonl"),
            "--kind",
            "raw",
        ],
    )

    assert validate_manifest_script.main() == 0
