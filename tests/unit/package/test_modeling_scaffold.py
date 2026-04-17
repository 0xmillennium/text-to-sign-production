"""Smoke tests for the Sprint 3 modeling scaffold."""

from __future__ import annotations

import sys
from importlib import import_module

import pytest

pytestmark = pytest.mark.unit


def test_modeling_scaffold_packages_are_importable() -> None:
    package_names = (
        "text_to_sign_production.modeling",
        "text_to_sign_production.modeling.backbones",
        "text_to_sign_production.modeling.config",
        "text_to_sign_production.modeling.data",
        "text_to_sign_production.modeling.inference",
        "text_to_sign_production.modeling.models",
        "text_to_sign_production.modeling.training",
    )

    for package_name in package_names:
        package = import_module(package_name)
        assert package.__name__ == package_name


def test_baseline_placeholder_scripts_fail_cleanly(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.evaluate_baseline as evaluate_baseline_script
    import scripts.export_qualitative_panel as export_qualitative_panel_script
    import scripts.train_baseline as train_baseline_script

    script_modules = (
        train_baseline_script,
        evaluate_baseline_script,
        export_qualitative_panel_script,
    )

    for script_module in script_modules:
        monkeypatch.setattr(sys, "argv", [f"{script_module.__name__}.py"])
        assert script_module.main() == 1
        captured = capsys.readouterr()
        assert "not implemented yet" in captured.err
        assert "Phase 1 placeholder" in captured.err
