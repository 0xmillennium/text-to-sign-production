"""Baseline-modeling resume and package-input contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

import text_to_sign_production.workflows.baseline_modeling as baseline_workflow_mod
from tests.support.modeling import touch_file

pytestmark = pytest.mark.unit


def test_training_resume_decision_prefers_extracted_then_archive_then_run(
    tmp_path: Path,
) -> None:
    layout = baseline_workflow_mod.resolve_baseline_run_layout(
        run_name="baseline-default",
        artifact_runs_root=tmp_path / "runs",
    )
    touch_file(layout.config_path)
    touch_file(layout.source_config_path)
    touch_file(layout.checkpoint_run_summary_path)
    touch_file(layout.last_checkpoint_path)
    touch_file(layout.training_archive_path)

    decision = baseline_workflow_mod.resolve_training_resume_decision(layout)

    assert decision.action == "reuse_extracted"

    layout.last_checkpoint_path.unlink()
    decision = baseline_workflow_mod.resolve_training_resume_decision(layout)

    assert decision.action == "extract_archive"

    layout.training_archive_path.unlink()
    decision = baseline_workflow_mod.resolve_training_resume_decision(layout)

    assert decision.action == "run_step"


def test_training_reuse_regenerates_missing_metrics_summary(tmp_path: Path) -> None:
    layout = baseline_workflow_mod.resolve_baseline_run_layout(
        run_name="baseline-default",
        artifact_runs_root=tmp_path / "runs",
    )
    touch_file(layout.config_path)
    touch_file(layout.source_config_path)
    touch_file(layout.checkpoint_run_summary_path)
    touch_file(layout.last_checkpoint_path)

    def unexpected_training_runner(
        config: Path,
        *,
        checkpoint_output_dir: Path | None = None,
    ) -> None:
        raise AssertionError("training runner should not be called")

    result = baseline_workflow_mod.ensure_baseline_training_outputs(
        layout=layout,
        training_runner=unexpected_training_runner,
    )

    assert result.action == "reuse_extracted"
    assert layout.metrics_run_summary_path.is_file()
    assert layout.metrics_run_summary_path.read_bytes() == (
        layout.checkpoint_run_summary_path.read_bytes()
    )


def test_training_resume_decision_rejects_required_artifact_directories(
    tmp_path: Path,
) -> None:
    layout = baseline_workflow_mod.resolve_baseline_run_layout(
        run_name="baseline-default",
        artifact_runs_root=tmp_path / "runs",
    )
    touch_file(layout.config_path)
    touch_file(layout.source_config_path)
    touch_file(layout.checkpoint_run_summary_path)
    touch_file(layout.metrics_run_summary_path)
    layout.last_checkpoint_path.mkdir(parents=True)
    touch_file(layout.training_archive_path)

    decision = baseline_workflow_mod.resolve_training_resume_decision(layout)

    assert decision.action == "extract_archive"


def test_require_decision_outputs_rejects_required_artifact_directories(
    tmp_path: Path,
) -> None:
    layout = baseline_workflow_mod.resolve_baseline_run_layout(
        run_name="baseline-default",
        artifact_runs_root=tmp_path / "runs",
    )
    touch_file(layout.config_path)
    touch_file(layout.source_config_path)
    touch_file(layout.checkpoint_run_summary_path)
    touch_file(layout.metrics_run_summary_path)
    layout.last_checkpoint_path.mkdir(parents=True)
    decision = baseline_workflow_mod.resolve_training_resume_decision(layout)

    with pytest.raises(FileNotFoundError, match="required file outputs"):
        baseline_workflow_mod._require_decision_outputs(decision)


def test_require_package_inputs_rejects_required_artifact_directories(tmp_path: Path) -> None:
    layout = baseline_workflow_mod.resolve_baseline_run_layout(
        run_name="baseline-default",
        artifact_runs_root=tmp_path / "runs",
    )
    for path in (
        layout.config_path,
        layout.source_config_path,
        layout.metrics_run_summary_path,
        layout.last_checkpoint_path,
        layout.panel_definition_path,
        layout.qualitative_records_path,
        layout.panel_summary_path,
    ):
        touch_file(path)
    layout.evidence_bundle_path.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="required file outputs"):
        baseline_workflow_mod._require_package_inputs(layout)
