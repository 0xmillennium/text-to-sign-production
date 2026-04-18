"""Baseline-modeling step execution contracts with fake collaborators."""

from __future__ import annotations

from pathlib import Path

import pytest

import text_to_sign_production.workflows.baseline_modeling as baseline_workflow_mod
from tests.support.modeling import (
    fake_create_archive,
    write_baseline_modeling_config,
    write_fake_training_outputs,
)

pytestmark = pytest.mark.unit


def test_non_prepare_step_skips_processed_validation_when_config_already_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_baseline_modeling_config(tmp_path)
    layout = baseline_workflow_mod.resolve_baseline_run_layout(
        run_name="baseline-default",
        artifact_runs_root=tmp_path / "runs",
    )
    baseline_workflow_mod.prepare_baseline_modeling_run(
        config_path,
        layout=layout,
        validate_processed_inputs=False,
    )

    def fake_training_runner(config: Path, *, checkpoint_output_dir: Path | None = None) -> None:
        assert checkpoint_output_dir is not None
        write_fake_training_outputs(checkpoint_output_dir)

    monkeypatch.setattr(
        baseline_workflow_mod,
        "create_tar_zst_archive",
        fake_create_archive,
    )

    result = baseline_workflow_mod.run_baseline_modeling(
        step="train",
        config_path=config_path,
        run_name="baseline-default",
        artifact_runs_root=tmp_path / "runs",
        training_runner=fake_training_runner,
    )

    assert result.steps[-1].step == "train"
    assert result.steps[-1].action == "run_step"
