"""Integration tests for Sprint 3 baseline-modeling workflow orchestration."""

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

pytestmark = pytest.mark.integration


def test_baseline_modeling_all_chains_training_qualitative_and_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
    monkeypatch.setattr(baseline_workflow_mod, "create_tar_zst_archive", fake_create_archive)
    workspace = write_tiny_baseline_modeling_workspace(tmp_path, run_name="integration-run")

    def fake_training_runner(config: Path, *, checkpoint_output_dir: Path | None = None) -> None:
        assert config.name == "baseline.yaml"
        assert checkpoint_output_dir is not None
        write_fake_training_outputs(checkpoint_output_dir)

    def fake_qualitative_runner(
        config: Path,
        *,
        output_dir: Path,
        checkpoint_path: Path | None = None,
        panel_definition_path: Path | None = None,
        run_summary_path: Path | None = None,
        panel_size: int = 8,
    ) -> None:
        assert config.name == "baseline.yaml"
        assert checkpoint_path is not None
        assert run_summary_path is not None
        assert panel_size == 2
        write_fake_qualitative_outputs(output_dir)

    result = baseline_workflow_mod.run_baseline_modeling(
        step="all",
        config_path=workspace.config_path,
        run_name=workspace.run_name,
        artifact_runs_root=workspace.artifact_runs_root,
        panel_size=2,
        training_runner=fake_training_runner,
        qualitative_runner=fake_qualitative_runner,
    )

    assert [step.step for step in result.steps] == [
        "prepare",
        "train",
        "export-panel",
        "package",
    ]
    assert [step.action for step in result.steps] == [
        "prepared",
        "run_step",
        "run_step",
        "run_step",
    ]
    assert result.layout.config_path.is_file()
    assert result.layout.metrics_run_summary_path.is_file()
    assert result.layout.evidence_bundle_path.is_file()
    assert result.layout.package_manifest_path.is_file()
    assert result.layout.training_archive_path.is_file()
    assert result.layout.qualitative_archive_path.is_file()
    assert result.layout.record_archive_path.is_file()


def test_baseline_modeling_reuses_extracted_training_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
    workspace = write_tiny_baseline_modeling_workspace(
        tmp_path,
        run_name="reuse-run",
    )
    layout = workspace.layout
    baseline_workflow_mod.prepare_baseline_modeling_run(workspace.config_path, layout=layout)
    write_fake_training_outputs(layout.checkpoints_dir)
    baseline_workflow_mod._sync_training_metrics(layout)
    layout.metrics_run_summary_path.unlink()

    def unexpected_training_runner(
        config: Path,
        *,
        checkpoint_output_dir: Path | None = None,
    ) -> None:
        raise AssertionError("training runner should not be called")

    result = baseline_workflow_mod.run_baseline_modeling(
        step="train",
        config_path=workspace.config_path,
        run_name=workspace.run_name,
        artifact_runs_root=workspace.artifact_runs_root,
        training_runner=unexpected_training_runner,
    )

    assert result.steps[-1].step == "train"
    assert result.steps[-1].action == "reuse_extracted"
    assert result.steps[-1].archive_path is None
    assert result.layout.metrics_run_summary_path.is_file()
