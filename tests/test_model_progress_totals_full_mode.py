from __future__ import annotations

from pathlib import Path

import pytest

from text_to_sign_production.workflows.model.contracts import ModelWorkflowInvariantError
from text_to_sign_production.workflows.model.processing.provider_progress import (
    ModelProviderProgress,
)


ROOT = Path(__file__).resolve().parents[1]


def test_full_materialization_progress_uses_effective_manifest_totals() -> None:
    for relative in (
        "src/text_to_sign_production/modeling/candidates/learned_pose_token/provider.py",
        "src/text_to_sign_production/modeling/candidates/latent_diffusion/provider.py",
        "src/text_to_sign_production/modeling/candidates/articulator_aware/provider.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")

        assert "resolve_effective_manifest_count" in source
        assert "total=caps[\"limit_train_samples\"]" not in source
        assert "total=caps[\"limit_validation_samples\"]" not in source


def test_full_mode_provider_progress_rejects_unknown_total() -> None:
    progress = ModelProviderProgress(
        progress_session=None,
        provider_key="learned_pose_token",
        provider_stage_id="train_text_to_token",
        run_mode="full",
    )

    with pytest.raises(ModelWorkflowInvariantError, match="unknown total"):
        progress.task(
            operation="train_epoch",
            label="train epoch",
            unit="batch",
            total=None,
        )
