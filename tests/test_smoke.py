"""Smoke and environment override tests for the repository scaffold."""

import os
from importlib import import_module, reload
from pathlib import Path
from typing import Any

import pytest

from text_to_sign_production import __version__, smoke_check


def test_package_is_importable() -> None:
    """The src-layout package should be importable after installation."""

    package = import_module("text_to_sign_production")
    assert package.__name__ == "text_to_sign_production"


def test_version_string_is_exposed() -> None:
    """The package should expose an explicit version string."""

    assert __version__ == "0.1.0"


def test_smoke_check_returns_expected_sentinel() -> None:
    """The smoke helper should remain deterministic."""

    assert smoke_check() == "t2sp-smoke-ok"


def test_repo_root_respects_env_override(tmp_path: Path, monkeypatch: Any) -> None:
    """The repo root can be overridden for non-editable package installs."""

    import text_to_sign_production.data.constants as constants_mod
    import text_to_sign_production.data.utils as utils_mod

    original_repo_root = os.environ.get("T2SP_REPO_ROOT")
    monkeypatch.setenv("T2SP_REPO_ROOT", str(tmp_path))
    reload(constants_mod)
    reload(utils_mod)
    try:
        assert constants_mod.REPO_ROOT == tmp_path.resolve()
        assert utils_mod.resolve_repo_path("data/example.txt") == tmp_path / "data/example.txt"
        assert utils_mod.repo_relative_path(tmp_path / "data/example.txt") == "data/example.txt"
    finally:
        if original_repo_root is None:
            monkeypatch.delenv("T2SP_REPO_ROOT", raising=False)
        else:
            monkeypatch.setenv("T2SP_REPO_ROOT", original_repo_root)
        reload(constants_mod)
        reload(utils_mod)


def test_resolve_repo_path_rejects_relative_escape(tmp_path: Path, monkeypatch: Any) -> None:
    """Repo-relative paths must stay within the repo root."""

    import text_to_sign_production.data.utils as utils_mod

    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path.resolve())

    with pytest.raises(ValueError, match="escapes repo root"):
        utils_mod.resolve_repo_path("../../outside.txt")
