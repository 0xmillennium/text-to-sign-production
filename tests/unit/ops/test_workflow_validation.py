"""Small workflow validation tests."""

from __future__ import annotations

import pytest

import text_to_sign_production.workflows.dataset_build as dataset_build_workflow_mod

pytestmark = pytest.mark.unit


def test_run_dataset_build_rejects_duplicate_splits() -> None:
    with pytest.raises(ValueError, match="Duplicate Dataset Build split"):
        dataset_build_workflow_mod.run_dataset_build(
            splits=("train", "train"),
            output_mode="none",
        )
