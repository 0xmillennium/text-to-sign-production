"""Unit tests for the notebook-facing dataset workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import text_to_sign_production.workflows.dataset as dataset_workflow_mod
from tests.support.builders.manifests import processed_record, write_jsonl_records
from tests.support.builders.samples import write_processed_sample_npz
from text_to_sign_production.core.paths import ProjectLayout

pytestmark = pytest.mark.unit


def test_validate_dataset_inputs_rejects_missing_raw_inputs(tmp_path: Path) -> None:
    layout = ProjectLayout(tmp_path)
    filter_config_path = tmp_path / "filter.yaml"
    filter_config_path.write_text("schema_version: 2\n", encoding="utf-8")

    with pytest.raises(dataset_workflow_mod.DatasetWorkflowInputError, match="does not exist"):
        dataset_workflow_mod.validate_dataset_inputs(
            dataset_workflow_mod.DatasetWorkflowConfig(
                splits=("train",),
                filter_config_path=filter_config_path,
                project_root=layout.root,
            )
        )


def test_run_dataset_workflow_calls_data_stages_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = ProjectLayout(tmp_path)
    filter_config_path = tmp_path / "filter.yaml"
    filter_config_path.write_text("schema_version: 2\n", encoding="utf-8")
    calls: list[str] = []

    def fake_build_raw_manifests(**kwargs: Any) -> dict[str, Any]:
        calls.append("build_raw_manifests")
        assert kwargs["splits"] == ("train",)
        assert kwargs["data_root"] == layout.data_root
        return {"splits": {"train": {}}, "generated_at": "test"}

    def fake_normalize_all_splits(**kwargs: Any) -> None:
        calls.append("normalize_all_splits")
        assert kwargs["splits"] == ("train",)
        assert kwargs["data_root"] == layout.data_root

    def fake_filter_all_splits(config_path: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append("filter_all_splits")
        assert config_path == filter_config_path.resolve()
        assert kwargs["splits"] == ("train",)
        assert kwargs["data_root"] == layout.data_root
        return {"splits": {"train": {}}, "generated_at": "test"}

    def fake_export_final_manifests(**kwargs: Any) -> dict[str, Any]:
        calls.append("export_final_manifests")
        assert kwargs["splits"] == ("train",)
        assert kwargs["data_root"] == layout.data_root
        return {"quality_report": {}, "split_report": {}}

    def fake_validate_manifest(path: Path, kind: str, **kwargs: Any) -> list[Any]:
        calls.append(f"validate:{kind}:{path.name}")
        assert kwargs["data_root"] == layout.data_root
        return []

    monkeypatch.setattr(dataset_workflow_mod, "validate_dataset_inputs", lambda config: None)
    monkeypatch.setattr(dataset_workflow_mod, "build_raw_manifests", fake_build_raw_manifests)
    monkeypatch.setattr(dataset_workflow_mod, "normalize_all_splits", fake_normalize_all_splits)
    monkeypatch.setattr(dataset_workflow_mod, "filter_all_splits", fake_filter_all_splits)
    monkeypatch.setattr(dataset_workflow_mod, "export_final_manifests", fake_export_final_manifests)
    monkeypatch.setattr(dataset_workflow_mod, "validate_manifest", fake_validate_manifest)
    monkeypatch.setattr(dataset_workflow_mod, "_verify_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(dataset_workflow_mod, "verify_output_dir", lambda *args, **kwargs: None)
    monkeypatch.setattr(dataset_workflow_mod, "verify_output_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(dataset_workflow_mod, "ensure_dir", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        dataset_workflow_mod,
        "_collect_processed_sample_archive_members",
        lambda **kwargs: {"train": (Path("data/processed/v1/samples/train/train-sample.npz"),)},
    )

    result = dataset_workflow_mod.run_dataset_workflow(
        dataset_workflow_mod.DatasetWorkflowConfig(
            splits=("train",),
            filter_config_path=filter_config_path,
            project_root=layout.root,
        )
    )

    assert calls == [
        "build_raw_manifests",
        "validate:raw:raw_train.jsonl",
        "normalize_all_splits",
        "validate:normalized:normalized_train.jsonl",
        "filter_all_splits",
        "validate:normalized:filtered_train.jsonl",
        "export_final_manifests",
        "validate:processed:train.jsonl",
    ]
    assert result.project_root == layout.root
    assert result.data_root == layout.data_root
    assert (
        result.raw_manifest_paths["train"]
        == (tmp_path / "data/interim/raw_manifests/raw_train.jsonl").resolve()
    )
    assert (
        result.processed_manifest_paths["train"]
        == (tmp_path / "data/processed/v1/manifests/train.jsonl").resolve()
    )
    assert result.processed_samples_root == (tmp_path / "data/processed/v1/samples").resolve()
    assert result.processed_sample_archive_members["train"] == (
        Path("data/processed/v1/samples/train/train-sample.npz"),
    )
    assert result.interim_report_paths["assumption"].name == "assumption-report.json"
    assert result.processed_report_paths["data_quality_json"].name == "data-quality-report.json"


def test_processed_sample_archive_members_use_only_final_manifest_records(
    tmp_path: Path,
) -> None:
    layout = ProjectLayout(tmp_path)
    referenced_member = Path("data/processed/v1/samples/train/referenced.npz")
    unreferenced_member = Path("data/processed/v1/samples/train/unreferenced.npz")

    write_processed_sample_npz(layout.root / referenced_member)
    write_processed_sample_npz(layout.root / unreferenced_member)
    write_jsonl_records(
        layout.processed_manifest_path("train"),
        [
            processed_record(
                referenced_member.as_posix(),
                split="train",
                sample_id="referenced",
            )
        ],
    )

    members = dataset_workflow_mod._collect_processed_sample_archive_members(
        layout=layout,
        splits=("train",),
    )

    assert members == {"train": (referenced_member,)}
    assert unreferenced_member not in members["train"]
    assert all(member.as_posix().startswith("data/") for member in members["train"])
