"""Unit tests for Sprint 3 baseline-modeling workflow logic."""

from __future__ import annotations

from pathlib import Path

import pytest

import text_to_sign_production.data.utils as utils_mod
import text_to_sign_production.workflows.baseline_modeling as baseline_workflow_mod
from tests.support.modeling import (
    write_baseline_modeling_config,
    write_fake_training_outputs,
    write_processed_modeling_split,
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
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    write_processed_modeling_split(tmp_path, split="train", sample_ids=("train-sample",))
    write_processed_modeling_split(tmp_path, split="val", sample_ids=("val-sample",))
    config_path = write_baseline_modeling_config(tmp_path)

    baseline_workflow_mod.validate_baseline_processed_inputs(config_path)


def test_validate_baseline_processed_inputs_rejects_empty_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
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


def test_training_resume_decision_prefers_extracted_then_archive_then_run(
    tmp_path: Path,
) -> None:
    layout = baseline_workflow_mod.resolve_baseline_run_layout(
        run_name="baseline-default",
        artifact_runs_root=tmp_path / "runs",
    )
    _touch(layout.config_path)
    _touch(layout.source_config_path)
    _touch(layout.checkpoint_run_summary_path)
    _touch(layout.last_checkpoint_path)
    _touch(layout.training_archive_path)

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
    _touch(layout.config_path)
    _touch(layout.source_config_path)
    _touch(layout.checkpoint_run_summary_path)
    _touch(layout.last_checkpoint_path)

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
    _touch(layout.config_path)
    _touch(layout.source_config_path)
    _touch(layout.checkpoint_run_summary_path)
    _touch(layout.metrics_run_summary_path)
    layout.last_checkpoint_path.mkdir(parents=True)
    _touch(layout.training_archive_path)

    decision = baseline_workflow_mod.resolve_training_resume_decision(layout)

    assert decision.action == "extract_archive"


def test_require_decision_outputs_rejects_required_artifact_directories(
    tmp_path: Path,
) -> None:
    layout = baseline_workflow_mod.resolve_baseline_run_layout(
        run_name="baseline-default",
        artifact_runs_root=tmp_path / "runs",
    )
    _touch(layout.config_path)
    _touch(layout.source_config_path)
    _touch(layout.checkpoint_run_summary_path)
    _touch(layout.metrics_run_summary_path)
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
        _touch(path)
    layout.evidence_bundle_path.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="required file outputs"):
        baseline_workflow_mod._require_package_inputs(layout)


def test_non_prepare_step_skips_processed_validation_when_config_already_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_baseline_modeling_config(tmp_path)
    layout = baseline_workflow_mod.resolve_baseline_run_layout(
        run_name="baseline-default",
        artifact_runs_root=tmp_path / "runs",
    )
    layout.config_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_workflow_mod.prepare_baseline_modeling_run(
        config_path,
        layout=layout,
        validate_processed_inputs=False,
    )

    def fail_validation(config: Path | str) -> None:
        raise AssertionError("processed validation should be skipped")

    def fake_training_runner(config: Path, *, checkpoint_output_dir: Path | None = None) -> None:
        assert checkpoint_output_dir is not None
        write_fake_training_outputs(checkpoint_output_dir)

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
        "validate_baseline_processed_inputs",
        fail_validation,
    )
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


def test_archive_training_outputs_reports_baseline_specific_missing_members(
    tmp_path: Path,
) -> None:
    layout = baseline_workflow_mod.resolve_baseline_run_layout(
        run_name="baseline-default",
        artifact_runs_root=tmp_path / "runs",
    )
    _touch(layout.config_path)
    _touch(layout.last_checkpoint_path)

    with pytest.raises(FileNotFoundError) as exc_info:
        baseline_workflow_mod._archive_training_outputs(layout)

    message = str(exc_info.value)
    assert "baseline training outputs" in message
    assert "baseline-modeling run artifacts" in message
    assert str(layout.metrics_dir) in message
    assert "Dataset Build" not in message


def test_archive_qualitative_outputs_reports_baseline_specific_missing_members(
    tmp_path: Path,
) -> None:
    layout = baseline_workflow_mod.resolve_baseline_run_layout(
        run_name="baseline-default",
        artifact_runs_root=tmp_path / "runs",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        baseline_workflow_mod._archive_qualitative_outputs(layout)

    message = str(exc_info.value)
    assert "baseline qualitative outputs" in message
    assert "baseline-modeling run artifacts" in message
    assert str(layout.qualitative_dir) in message
    assert "Dataset Build" not in message


def test_archive_record_package_reports_baseline_specific_missing_members(
    tmp_path: Path,
) -> None:
    layout = baseline_workflow_mod.resolve_baseline_run_layout(
        run_name="baseline-default",
        artifact_runs_root=tmp_path / "runs",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        baseline_workflow_mod._archive_record_package(layout)

    message = str(exc_info.value)
    assert "baseline record package" in message
    assert "baseline-modeling run artifacts" in message
    assert str(layout.record_dir) in message
    assert "Dataset Build" not in message


def test_archive_helper_file_not_found_is_reworded_for_baseline_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = baseline_workflow_mod.resolve_baseline_run_layout(
        run_name="baseline-default",
        artifact_runs_root=tmp_path / "runs",
    )
    layout.qualitative_dir.mkdir(parents=True)

    def raise_dataset_build_error(
        *,
        archive_path: Path,
        members: tuple[Path, ...],
        cwd: Path,
        label: str,
    ) -> Path:
        raise FileNotFoundError("Missing required Dataset Build outputs:\n- stale")

    monkeypatch.setattr(
        baseline_workflow_mod,
        "create_tar_zst_archive",
        raise_dataset_build_error,
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        baseline_workflow_mod._archive_qualitative_outputs(layout)

    message = str(exc_info.value)
    assert "baseline qualitative outputs" in message
    assert "baseline-modeling archive member validation failed" in message
    assert "Dataset Build" not in message


def test_extract_archive_into_run_root_preserves_existing_archives_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = baseline_workflow_mod.resolve_baseline_run_layout(
        run_name="baseline-default",
        artifact_runs_root=tmp_path / "runs",
    )
    _touch(layout.archives_dir / "keep.tar.zst")
    archive_path = tmp_path / "archive.tar.zst"
    archive_path.write_bytes(b"archive")

    def fake_extract(archive: Path, destination: Path, *, label: str) -> None:
        assert archive == archive_path
        _touch(destination / "checkpoints" / "run_summary.json")

    monkeypatch.setattr(
        baseline_workflow_mod,
        "extract_tar_zst_with_progress",
        fake_extract,
    )

    baseline_workflow_mod._extract_archive_into_run_root(
        archive_path,
        run_root=layout.run_root,
        label="Extract",
    )

    assert (layout.archives_dir / "keep.tar.zst").is_file()
    assert (layout.checkpoints_dir / "run_summary.json").is_file()


def test_extract_archive_preserves_matching_existing_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_baseline_modeling_config(
        tmp_path,
        checkpoint_output_dir="outputs/modeling/original",
    )
    layout = baseline_workflow_mod.resolve_baseline_run_layout(
        run_name="baseline-default",
        artifact_runs_root=tmp_path / "runs",
    )
    baseline_workflow_mod.prepare_baseline_modeling_run(
        config_path,
        layout=layout,
        validate_processed_inputs=False,
    )
    current_effective_config = layout.config_path.read_bytes()
    archive_path = tmp_path / "archive.tar.zst"
    archive_path.write_bytes(b"archive")

    def fake_extract(archive: Path, destination: Path, *, label: str) -> None:
        assert archive == archive_path
        archived_config_dir = destination / "config"
        archived_config_dir.mkdir(parents=True)
        (archived_config_dir / baseline_workflow_mod.SOURCE_BASELINE_CONFIG_NAME).write_bytes(
            layout.source_config_path.read_bytes()
        )
        (archived_config_dir / "baseline.yaml").write_text(
            "checkpoint:\n  output_dir: /stale/archive/path\n",
            encoding="utf-8",
        )
        _touch(destination / "checkpoints" / "run_summary.json")

    monkeypatch.setattr(
        baseline_workflow_mod,
        "extract_tar_zst_with_progress",
        fake_extract,
    )

    baseline_workflow_mod._extract_archive_into_run_root(
        archive_path,
        run_root=layout.run_root,
        label="Extract",
    )

    assert layout.config_path.read_bytes() == current_effective_config
    assert (layout.checkpoints_dir / "run_summary.json").is_file()


def test_extract_archive_rejects_different_existing_config_before_moving_members(
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
    archive_path = tmp_path / "archive.tar.zst"
    archive_path.write_bytes(b"archive")

    def fake_extract(archive: Path, destination: Path, *, label: str) -> None:
        assert archive == archive_path
        archived_config_dir = destination / "config"
        archived_config_dir.mkdir(parents=True)
        (archived_config_dir / baseline_workflow_mod.SOURCE_BASELINE_CONFIG_NAME).write_text(
            "different: true\n",
            encoding="utf-8",
        )
        _touch(destination / "checkpoints" / "run_summary.json")

    monkeypatch.setattr(
        baseline_workflow_mod,
        "extract_tar_zst_with_progress",
        fake_extract,
    )

    with pytest.raises(ValueError, match="source config differs"):
        baseline_workflow_mod._extract_archive_into_run_root(
            archive_path,
            run_root=layout.run_root,
            label="Extract",
        )

    assert not (layout.checkpoints_dir / "run_summary.json").exists()


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
