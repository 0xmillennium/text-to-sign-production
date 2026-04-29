"""Smoke tests for the Sprint 3 modeling scaffold."""

from __future__ import annotations

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


def test_workflow_package_is_importable_without_script_surface() -> None:
    package = import_module("text_to_sign_production.workflows")

    assert package.__all__ == ["base", "dataset", "visualization"]
