"""Pytest wiring shared across the repository test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from tests.support.paths import patch_dataset_paths  # noqa: E402
from tests.support.scenarios import create_tiny_dataset_workspace  # noqa: E402


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def tmp_project_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    return root


@pytest.fixture
def patched_dataset_roots(
    tmp_project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    patch_dataset_paths(monkeypatch, tmp_project_root)
    return tmp_project_root


@pytest.fixture
def tiny_dataset_workspace(patched_dataset_roots: Path) -> Path:
    return create_tiny_dataset_workspace(patched_dataset_roots)
