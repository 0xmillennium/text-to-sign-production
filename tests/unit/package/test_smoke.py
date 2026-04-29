"""Package import and repository path smoke tests."""

from __future__ import annotations

import os
from importlib import import_module, reload
from pathlib import Path
from typing import Any

import pytest

from text_to_sign_production import __version__, smoke_check

pytestmark = pytest.mark.unit


def test_package_is_importable() -> None:
    package = import_module("text_to_sign_production")
    assert package.__name__ == "text_to_sign_production"


def test_version_string_is_exposed() -> None:
    assert __version__ == "0.1.0"


def test_smoke_check_returns_expected_sentinel() -> None:
    assert smoke_check() == "t2sp-smoke-ok"


def test_repo_root_respects_env_override(tmp_path: Path, monkeypatch: Any) -> None:
    import text_to_sign_production.core.paths as paths_mod

    original_repo_root = os.environ.get("T2SP_REPO_ROOT")
    monkeypatch.setenv("T2SP_REPO_ROOT", str(tmp_path))
    reload(paths_mod)
    try:
        assert paths_mod.DEFAULT_REPO_ROOT == tmp_path.resolve()
        assert paths_mod.ProjectLayout(tmp_path).how2sign_root == tmp_path / "data/raw/how2sign"
        assert paths_mod.resolve_repo_path("data/example.txt") == tmp_path / "data/example.txt"
        assert paths_mod.repo_relative_path(tmp_path / "data/example.txt") == "data/example.txt"
    finally:
        if original_repo_root is None:
            monkeypatch.delenv("T2SP_REPO_ROOT", raising=False)
        else:
            monkeypatch.setenv("T2SP_REPO_ROOT", original_repo_root)
        reload(paths_mod)


def test_resolve_repo_path_rejects_relative_escape(tmp_path: Path, monkeypatch: Any) -> None:
    import text_to_sign_production.core.paths as paths_mod

    monkeypatch.setattr(paths_mod, "DEFAULT_REPO_ROOT", tmp_path.resolve())

    with pytest.raises(ValueError, match="must stay under"):
        paths_mod.resolve_repo_path("../../outside.txt")
