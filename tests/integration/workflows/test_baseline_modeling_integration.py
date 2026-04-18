"""Integration tests for Sprint 3 baseline-modeling workflow orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

import text_to_sign_production.data.utils as utils_mod
import text_to_sign_production.workflows.baseline_modeling as baseline_workflow_mod
from tests.support.modeling import (
    write_baseline_modeling_config,
    write_fake_qualitative_outputs,
    write_fake_training_outputs,
    write_processed_modeling_split,
)

pytestmark = pytest.mark.integration


def test_baseline_modeling_all_chains_training_qualitative_and_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    _patch_archive_creation(monkeypatch)
    write_processed_modeling_split(tmp_path, split="train", sample_ids=("train-sample",))
    write_processed_modeling_split(tmp_path, split="val", sample_ids=("val-sample",))
    config_path = write_baseline_modeling_config(tmp_path)

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
        config_path=config_path,
        run_name="integration-run",
        artifact_runs_root=tmp_path / "runs",
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
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    write_processed_modeling_split(tmp_path, split="train", sample_ids=("train-sample",))
    write_processed_modeling_split(tmp_path, split="val", sample_ids=("val-sample",))
    config_path = write_baseline_modeling_config(tmp_path)
    layout = baseline_workflow_mod.resolve_baseline_run_layout(
        run_name="reuse-run",
        artifact_runs_root=tmp_path / "runs",
    )
    baseline_workflow_mod.prepare_baseline_modeling_run(config_path, layout=layout)
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
        config_path=config_path,
        run_name="reuse-run",
        artifact_runs_root=tmp_path / "runs",
        training_runner=unexpected_training_runner,
    )

    assert result.steps[-1].step == "train"
    assert result.steps[-1].action == "reuse_extracted"
    assert result.steps[-1].archive_path is None
    assert result.layout.metrics_run_summary_path.is_file()


def _patch_archive_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_create_archive(
        *,
        archive_path: Path,
        members: tuple[Path, ...],
        cwd: Path,
        label: str,
    ) -> Path:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(b"archive")
        return archive_path

    monkeypatch.setattr(
        baseline_workflow_mod,
        "create_tar_zst_archive",
        fake_create_archive,
    )
