"""Baseline-modeling archive and extraction contracts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

import text_to_sign_production.workflows._baseline_archive_ops as baseline_archive_ops_mod
import text_to_sign_production.workflows.baseline_modeling as baseline_workflow_mod
from tests.support.modeling import (
    fake_extract_archive_with,
    raise_dataset_build_archive_error,
    touch_file,
    write_baseline_modeling_config,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("archive_call", "artifact_name", "missing_path_name"),
    [
        (
            baseline_workflow_mod._archive_training_outputs,
            "baseline training outputs",
            "metrics",
        ),
        (
            baseline_workflow_mod._archive_qualitative_outputs,
            "baseline qualitative outputs",
            "qualitative",
        ),
        (
            baseline_workflow_mod._archive_record_package,
            "baseline record package",
            "record",
        ),
    ],
)
def test_archive_outputs_report_baseline_specific_missing_members(
    tmp_path: Path,
    archive_call: Callable[[baseline_workflow_mod.BaselineRunLayout], Path],
    artifact_name: str,
    missing_path_name: str,
) -> None:
    layout = baseline_workflow_mod.resolve_baseline_run_layout(
        run_name="baseline-default",
        artifact_runs_root=tmp_path / "runs",
    )
    touch_file(layout.config_path)
    touch_file(layout.last_checkpoint_path)

    with pytest.raises(FileNotFoundError) as exc_info:
        archive_call(layout)

    message = str(exc_info.value)
    assert artifact_name in message
    assert "baseline-modeling run artifacts" in message
    assert missing_path_name in message
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
    monkeypatch.setattr(
        baseline_archive_ops_mod,
        "create_tar_zst_archive_from_snapshot",
        raise_dataset_build_archive_error,
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
    touch_file(layout.archives_dir / "keep.tar.zst")
    archive_path = tmp_path / "archive.tar.zst"
    archive_path.write_bytes(b"archive")

    def build_extracted_tree(destination: Path) -> None:
        touch_file(destination / "checkpoints" / "run_summary.json")

    monkeypatch.setattr(
        baseline_archive_ops_mod,
        "extract_tar_zst_with_progress",
        fake_extract_archive_with(build_extracted_tree),
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

    def build_extracted_tree(destination: Path) -> None:
        archived_config_dir = destination / "config"
        archived_config_dir.mkdir(parents=True)
        (archived_config_dir / baseline_workflow_mod.SOURCE_BASELINE_CONFIG_NAME).write_bytes(
            layout.source_config_path.read_bytes()
        )
        (archived_config_dir / "baseline.yaml").write_text(
            "checkpoint:\n  output_dir: /stale/archive/path\n",
            encoding="utf-8",
        )
        touch_file(destination / "checkpoints" / "run_summary.json")

    monkeypatch.setattr(
        baseline_archive_ops_mod,
        "extract_tar_zst_with_progress",
        fake_extract_archive_with(build_extracted_tree),
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

    def build_extracted_tree(destination: Path) -> None:
        archived_config_dir = destination / "config"
        archived_config_dir.mkdir(parents=True)
        (archived_config_dir / baseline_workflow_mod.SOURCE_BASELINE_CONFIG_NAME).write_text(
            "different: true\n",
            encoding="utf-8",
        )
        touch_file(destination / "checkpoints" / "run_summary.json")

    monkeypatch.setattr(
        baseline_archive_ops_mod,
        "extract_tar_zst_with_progress",
        fake_extract_archive_with(build_extracted_tree),
    )

    with pytest.raises(ValueError, match="source config differs"):
        baseline_workflow_mod._extract_archive_into_run_root(
            archive_path,
            run_root=layout.run_root,
            label="Extract",
        )

    assert not (layout.checkpoints_dir / "run_summary.json").exists()
