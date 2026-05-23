from __future__ import annotations

import pytest

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.workflows.test_model.constants import TEST_MODEL_WORKFLOW_NAME
from text_to_sign_production.workflows.test_model.progress import (
    test_model_progress_stage as build_test_model_progress_stage,
    visible_test_model_progress_session,
)


@pytest.mark.unit
def test_visible_test_model_progress_session_defaults_to_progress_session() -> None:
    progress = visible_test_model_progress_session(None)

    assert isinstance(progress, ProgressSession)
    assert progress.workflow_id == TEST_MODEL_WORKFLOW_NAME


@pytest.mark.unit
def test_test_model_progress_stage_allows_counter_contract() -> None:
    spec = build_test_model_progress_stage(
        stage_id="test_model.visualization.render",
        label="test_model visualization render",
        unit="artifact",
        owner_module=__name__,
        operation_kind="visualization_render",
        total_semantics="visualization artifacts rendered",
        allowed_counters=("created",),
    )

    assert spec.allowed_counters == ("created",)
