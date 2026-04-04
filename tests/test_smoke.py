"""Smoke tests for the Sprint 1 repository scaffold."""

from importlib import import_module

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
