from __future__ import annotations

import pytest

from text_to_sign_production.workflows.model.contracts import ModelWorkflowInvariantError
from text_to_sign_production.workflows.model.processing.provider_progress import (
    ModelProviderProgress,
)


def test_full_mode_provider_progress_rejects_unknown_total() -> None:
    progress = ModelProviderProgress(
        progress_session=None,
        provider_key="learned_pose_token",
        provider_stage_id="fit_representation",
        run_mode="full",
    )

    with pytest.raises(ModelWorkflowInvariantError, match="unknown total"):
        progress.task(
            operation="materialize_train",
            label="materialize train",
            unit="sample",
            total=None,
        )


def test_debug_mode_provider_progress_allows_unknown_total() -> None:
    progress = ModelProviderProgress(
        progress_session=None,
        provider_key="learned_pose_token",
        provider_stage_id="fit_representation",
        run_mode="debug",
    )

    handle = progress.task(
        operation="materialize_train",
        label="materialize train",
        unit="sample",
        total=None,
    )

    assert handle is not None
