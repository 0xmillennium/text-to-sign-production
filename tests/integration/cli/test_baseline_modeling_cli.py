"""Baseline-modeling public CLI wrapper tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import scripts.baseline_modeling as baseline_modeling_script
from text_to_sign_production.workflows.baseline_modeling import (
    BaselineModelingStepResult,
    BaselineModelingWorkflowResult,
    resolve_baseline_run_layout,
)

pytestmark = pytest.mark.integration


def test_baseline_modeling_cli_calls_stage_workflow(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "baseline.yaml"
    artifact_runs_root = tmp_path / "runs"
    seen_args: dict[str, object] = {}

    def fake_run_baseline_modeling(**kwargs: object) -> BaselineModelingWorkflowResult:
        seen_args.update(kwargs)
        layout = resolve_baseline_run_layout(
            run_name=str(kwargs["run_name"]),
            artifact_runs_root=artifact_runs_root,
        )
        return BaselineModelingWorkflowResult(
            run_name=layout.run_name,
            layout=layout,
            steps=(
                BaselineModelingStepResult(
                    step="train",
                    action="reuse_extracted",
                    output_path=layout.checkpoints_dir,
                    archive_path=layout.training_archive_path,
                    paths={},
                ),
            ),
        )

    monkeypatch.setattr(
        baseline_modeling_script,
        "run_baseline_modeling",
        fake_run_baseline_modeling,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "baseline_modeling.py",
            "train",
            "--config",
            str(config_path),
            "--run-name",
            "demo-run",
            "--artifact-runs-root",
            str(artifact_runs_root),
            "--panel-size",
            "3",
        ],
    )

    assert baseline_modeling_script.main() == 0
    captured = capsys.readouterr()
    assert seen_args == {
        "step": "train",
        "config_path": config_path,
        "run_name": "demo-run",
        "artifact_runs_root": artifact_runs_root,
        "panel_size": 3,
    }
    assert "Baseline Modeling run: demo-run" in captured.out
    assert "train: reuse_extracted" in captured.out


def test_baseline_modeling_cli_reports_workflow_errors(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_baseline_modeling(**kwargs: object) -> None:
        raise ValueError("bad run")

    monkeypatch.setattr(
        baseline_modeling_script,
        "run_baseline_modeling",
        fake_run_baseline_modeling,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "baseline_modeling.py",
            "all",
            "--config",
            str(tmp_path / "baseline.yaml"),
        ],
    )

    assert baseline_modeling_script.main() == 1
    captured = capsys.readouterr()
    assert "error: bad run" in captured.err
