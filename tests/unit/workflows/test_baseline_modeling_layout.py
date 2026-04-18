"""Baseline-modeling layout, config, and input validation contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

import text_to_sign_production.workflows.baseline_modeling as baseline_workflow_mod
from tests.support.modeling import (
    patch_modeling_repo_root,
    write_baseline_modeling_config,
    write_processed_modeling_split,
    write_tiny_baseline_modeling_workspace,
)
from text_to_sign_production.modeling.training.config import load_baseline_training_config

pytestmark = pytest.mark.unit


def test_resolve_baseline_run_layout_uses_stable_directory_names(tmp_path: Path) -> None:
    layout = baseline_workflow_mod.resolve_baseline_run_layout(
        run_name="baseline-default",
        artifact_runs_root=tmp_path / "runs",
    )

    assert layout.run_root == (tmp_path / "runs" / "baseline-default").resolve()
    assert layout.config_dir == layout.run_root / "config"
    assert layout.checkpoints_dir == layout.run_root / "checkpoints"
    assert layout.metrics_dir == layout.run_root / "metrics"
    assert layout.qualitative_dir == layout.run_root / "qualitative"
    assert layout.record_dir == layout.run_root / "record"
    assert layout.archives_dir == layout.run_root / "archives"
    assert layout.training_archive_path.name == "baseline_training_outputs.tar.zst"
    assert layout.qualitative_archive_path.name == "baseline_qualitative_outputs.tar.zst"
    assert layout.record_archive_path.name == "baseline_record_package.tar.zst"


@pytest.mark.parametrize("run_name", ["", " with-space", "../bad", "bad/name", ".hidden"])
def test_validate_baseline_run_name_rejects_unsafe_names(run_name: str) -> None:
    with pytest.raises(ValueError, match="run_name"):
        baseline_workflow_mod.validate_baseline_run_name(run_name)


def test_validate_baseline_processed_inputs_requires_manifest_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
    workspace = write_tiny_baseline_modeling_workspace(tmp_path)

    baseline_workflow_mod.validate_baseline_processed_inputs(workspace.config_path)


def test_validate_baseline_processed_inputs_rejects_empty_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
    write_processed_modeling_split(tmp_path, split="train", sample_ids=("train-sample",))
    val_manifest = tmp_path / "data/processed/v1/manifests/val.jsonl"
    val_manifest.parent.mkdir(parents=True, exist_ok=True)
    val_manifest.write_text("", encoding="utf-8")
    config_path = write_baseline_modeling_config(tmp_path)

    with pytest.raises(ValueError, match="val manifest has no records"):
        baseline_workflow_mod.validate_baseline_processed_inputs(config_path)


def test_prepare_writes_effective_config_and_preserves_source_config(tmp_path: Path) -> None:
    config_path = write_baseline_modeling_config(
        tmp_path,
        checkpoint_output_dir="outputs/modeling/original",
    )
    layout = baseline_workflow_mod.resolve_baseline_run_layout(
        run_name="baseline-default",
        artifact_runs_root=tmp_path / "runs",
    )

    result = baseline_workflow_mod.prepare_baseline_modeling_run(
        config_path,
        layout=layout,
        validate_processed_inputs=False,
    )
    effective_config = load_baseline_training_config(layout.config_path, validate_paths=False)
    source_config = load_baseline_training_config(
        layout.source_config_path,
        validate_paths=False,
    )

    assert result.paths["config_path"] == layout.config_path
    assert result.paths["source_config_path"] == layout.source_config_path
    assert effective_config.checkpoint.output_dir == layout.checkpoints_dir
    assert source_config.checkpoint.output_dir.name == "original"


def test_prepare_rejects_existing_run_config_with_unexpected_content(tmp_path: Path) -> None:
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
    layout.config_path.write_text("unexpected: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="different content"):
        baseline_workflow_mod.prepare_baseline_modeling_run(
            config_path,
            layout=layout,
            validate_processed_inputs=False,
        )
