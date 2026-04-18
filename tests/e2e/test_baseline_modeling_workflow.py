"""CPU-safe e2e-like Sprint 3 baseline-modeling workflow test."""

from __future__ import annotations

from pathlib import Path

import pytest

import text_to_sign_production.workflows.baseline_modeling as baseline_workflow_mod
from tests.support.modeling import (
    fake_create_archive,
    patch_modeling_repo_root,
    write_fake_qualitative_outputs,
    write_fake_training_outputs,
    write_tiny_baseline_modeling_workspace,
)

pytestmark = pytest.mark.e2e


def test_baseline_modeling_workflow_runs_on_tiny_processed_data_without_model_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
    monkeypatch.setattr(baseline_workflow_mod, "create_tar_zst_archive", fake_create_archive)
    workspace = write_tiny_baseline_modeling_workspace(tmp_path, run_name="tiny-e2e")

    result = baseline_workflow_mod.run_baseline_modeling(
        step="all",
        config_path=workspace.config_path,
        run_name=workspace.run_name,
        artifact_runs_root=workspace.artifact_runs_root,
        panel_size=1,
        training_runner=_fake_training_runner,
        qualitative_runner=_fake_qualitative_runner,
    )

    assert result.run_name == "tiny-e2e"
    assert (result.layout.run_root / "config/baseline.yaml").is_file()
    assert (result.layout.run_root / "checkpoints/run_summary.json").is_file()
    assert (result.layout.run_root / "qualitative/panel_summary.json").is_file()
    assert (result.layout.run_root / "record/baseline_modeling_package.json").is_file()
    assert {step.action for step in result.steps} == {"prepared", "run_step"}


def _fake_training_runner(
    config: Path,
    *,
    checkpoint_output_dir: Path | None = None,
) -> None:
    assert checkpoint_output_dir is not None
    write_fake_training_outputs(checkpoint_output_dir)


def _fake_qualitative_runner(
    config: Path,
    *,
    output_dir: Path,
    checkpoint_path: Path | None = None,
    panel_definition_path: Path | None = None,
    run_summary_path: Path | None = None,
    panel_size: int = 8,
) -> None:
    assert checkpoint_path is not None
    assert run_summary_path is not None
    assert panel_size == 1
    write_fake_qualitative_outputs(output_dir)
