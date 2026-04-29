"""Unit tests for the notebook-facing baseline workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import text_to_sign_production.workflows.base as base_workflow_mod
from tests.support.modeling import patch_modeling_repo_root, write_tiny_baseline_modeling_workspace
from text_to_sign_production.core.files import OutputExistsPolicy
from text_to_sign_production.modeling.inference.qualitative import QualitativeExportResult
from text_to_sign_production.modeling.training.train import BaselineTrainingRunResult

pytestmark = pytest.mark.unit


def test_validate_base_inputs_rejects_missing_config(tmp_path: Path) -> None:
    with pytest.raises(base_workflow_mod.BaseWorkflowInputError, match="does not exist"):
        base_workflow_mod.validate_base_inputs(
            base_workflow_mod.BaseWorkflowConfig(
                project_root=tmp_path,
                config_path=tmp_path / "missing.yaml",
            )
        )


def test_validate_base_inputs_inspects_processed_inputs_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
    workspace = write_tiny_baseline_modeling_workspace(tmp_path)

    def fail_if_training_runs(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("validation must not run training")

    monkeypatch.setattr(base_workflow_mod, "run_baseline_training", fail_if_training_runs)

    base_workflow_mod.validate_base_inputs(
        base_workflow_mod.BaseWorkflowConfig(
            project_root=tmp_path,
            config_path=workspace.config_path,
        )
    )


def test_run_base_workflow_calls_training_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
    workspace = write_tiny_baseline_modeling_workspace(tmp_path)
    training_result = _training_result(tmp_path / "data/artifacts/base/test-run/checkpoints")
    seen: dict[str, Any] = {}

    def fake_run_baseline_training(config_path: Path, **kwargs: Any) -> BaselineTrainingRunResult:
        seen["config_path"] = config_path
        seen["checkpoint_output_dir"] = kwargs["checkpoint_output_dir"]
        seen["repo_root"] = kwargs["repo_root"]
        return training_result

    monkeypatch.setattr(base_workflow_mod, "run_baseline_training", fake_run_baseline_training)

    result = base_workflow_mod.run_base_workflow(
        base_workflow_mod.BaseWorkflowConfig(
            project_root=tmp_path,
            run_name="test-run",
            config_path=workspace.config_path,
            output_policy=OutputExistsPolicy.OVERWRITE,
        )
    )

    assert seen == {
        "config_path": workspace.config_path,
        "checkpoint_output_dir": tmp_path / "data/artifacts/base/test-run/checkpoints",
        "repo_root": tmp_path.resolve(),
    }
    assert result.training == training_result
    assert result.qualitative is None
    assert result.run_root == (tmp_path / "data/artifacts/base/test-run").resolve()


def test_run_base_workflow_dispatches_optional_qualitative_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
    workspace = write_tiny_baseline_modeling_workspace(tmp_path)
    training_result = _training_result(tmp_path / "data/artifacts/base/test-run/checkpoints")
    qualitative_result = _qualitative_result(tmp_path / "data/artifacts/base/test-run/qualitative")
    seen: dict[str, Any] = {}

    monkeypatch.setattr(
        base_workflow_mod,
        "run_baseline_training",
        lambda *args, **kwargs: training_result,
    )

    def fake_export_qualitative_panel(config_path: Path, **kwargs: Any) -> QualitativeExportResult:
        seen["config_path"] = config_path
        seen.update(kwargs)
        return qualitative_result

    monkeypatch.setattr(
        base_workflow_mod,
        "export_qualitative_panel",
        fake_export_qualitative_panel,
    )

    result = base_workflow_mod.run_base_workflow(
        base_workflow_mod.BaseWorkflowConfig(
            project_root=tmp_path,
            run_name="test-run",
            config_path=workspace.config_path,
            run_qualitative_export=True,
            panel_size=2,
            output_policy=OutputExistsPolicy.OVERWRITE,
        )
    )

    assert seen["config_path"] == workspace.config_path
    assert seen["output_dir"] == tmp_path / "data/artifacts/base/test-run/qualitative"
    assert seen["checkpoint_path"] == training_result.best_checkpoint_path
    assert seen["run_summary_path"] == training_result.summary_path
    assert seen["panel_size"] == 2
    assert seen["repo_root"] == tmp_path.resolve()
    assert result.qualitative == qualitative_result


def _training_result(checkpoint_dir: Path) -> BaselineTrainingRunResult:
    return BaselineTrainingRunResult(
        summary_path=checkpoint_dir / "run_summary.json",
        last_checkpoint_path=checkpoint_dir / "last.pt",
        best_checkpoint_path=checkpoint_dir / "best.pt",
        final_train_loss=1.0,
        final_validation_loss=2.0,
        final_validation_metric=3.0,
        best_validation_loss=2.0,
        best_epoch=1,
    )


def _qualitative_result(output_dir: Path) -> QualitativeExportResult:
    return QualitativeExportResult(
        output_dir=output_dir,
        panel_definition_path=output_dir / "panel_definition.json",
        records_path=output_dir / "records.jsonl",
        panel_summary_path=output_dir / "panel_summary.json",
        evidence_bundle_path=output_dir / "baseline_evidence_bundle.json",
        checkpoint_path=output_dir / "best.pt",
        sample_count=1,
        sample_ids=("val-sample",),
    )
