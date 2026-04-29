"""Integration tests for the notebook-facing baseline workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import text_to_sign_production.workflows.base as base_workflow_mod
from tests.support.modeling import patch_modeling_repo_root, write_tiny_baseline_modeling_workspace
from text_to_sign_production.core.files import OutputExistsPolicy
from text_to_sign_production.modeling.inference.qualitative import QualitativeExportResult
from text_to_sign_production.modeling.training.train import BaselineTrainingRunResult
from text_to_sign_production.workflows.base import BaseWorkflowConfig, run_base_workflow

pytestmark = pytest.mark.integration


def test_base_workflow_runs_training_and_optional_qualitative_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
    workspace = write_tiny_baseline_modeling_workspace(tmp_path)
    run_root = tmp_path / "data/artifacts/base/integration-run"
    training_result = BaselineTrainingRunResult(
        summary_path=run_root / "checkpoints/run_summary.json",
        last_checkpoint_path=run_root / "checkpoints/last.pt",
        best_checkpoint_path=run_root / "checkpoints/best.pt",
        final_train_loss=1.0,
        final_validation_loss=2.0,
        final_validation_metric=3.0,
        best_validation_loss=2.0,
        best_epoch=1,
    )
    best_checkpoint_path = training_result.best_checkpoint_path
    assert best_checkpoint_path is not None
    qualitative_result = QualitativeExportResult(
        output_dir=run_root / "qualitative",
        panel_definition_path=run_root / "qualitative/panel_definition.json",
        records_path=run_root / "qualitative/records.jsonl",
        panel_summary_path=run_root / "qualitative/panel_summary.json",
        evidence_bundle_path=run_root / "qualitative/baseline_evidence_bundle.json",
        checkpoint_path=best_checkpoint_path,
        sample_count=1,
        sample_ids=("val-sample",),
    )
    seen: dict[str, Any] = {}

    def fake_run_baseline_training(*args: Any, **kwargs: Any) -> BaselineTrainingRunResult:
        seen["training_args"] = args
        seen["training_kwargs"] = kwargs
        return training_result

    def fake_export_qualitative_panel(*args: Any, **kwargs: Any) -> QualitativeExportResult:
        seen["qualitative_args"] = args
        seen["qualitative_kwargs"] = kwargs
        return qualitative_result

    monkeypatch.setattr(
        base_workflow_mod,
        "run_baseline_training",
        fake_run_baseline_training,
    )
    monkeypatch.setattr(
        base_workflow_mod,
        "export_qualitative_panel",
        fake_export_qualitative_panel,
    )

    result = run_base_workflow(
        BaseWorkflowConfig(
            project_root=tmp_path,
            run_name="integration-run",
            config_path=workspace.config_path,
            run_qualitative_export=True,
            output_policy=OutputExistsPolicy.OVERWRITE,
        )
    )

    assert seen["training_args"] == (workspace.config_path,)
    assert seen["training_kwargs"] == {
        "checkpoint_output_dir": run_root / "checkpoints",
        "repo_root": tmp_path.resolve(),
    }
    assert seen["qualitative_args"] == (workspace.config_path,)
    assert seen["qualitative_kwargs"]["checkpoint_path"] == training_result.best_checkpoint_path
    assert seen["qualitative_kwargs"]["run_summary_path"] == training_result.summary_path
    assert seen["qualitative_kwargs"]["output_dir"] == run_root / "qualitative"
    assert seen["qualitative_kwargs"]["repo_root"] == tmp_path.resolve()
    assert result.training == training_result
    assert result.qualitative == qualitative_result
    assert result.run_root == run_root.resolve()
