"""Integration tests for the notebook-facing dataset workflow."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.assertions import load_jsonl
from text_to_sign_production.core.paths import ProjectLayout
from text_to_sign_production.data.constants import DEFAULT_FILTER_CONFIG_RELATIVE_PATH
from text_to_sign_production.workflows.dataset import (
    DatasetWorkflowConfig,
    run_dataset_workflow,
)

pytestmark = pytest.mark.integration


def test_dataset_workflow_runs_tiny_split_over_project_root(
    tiny_dataset_workspace: Path,
) -> None:
    layout = ProjectLayout(tiny_dataset_workspace)
    result = run_dataset_workflow(
        DatasetWorkflowConfig(
            splits=("train",),
            filter_config_path=tiny_dataset_workspace / DEFAULT_FILTER_CONFIG_RELATIVE_PATH,
            project_root=layout.root,
        )
    )

    assert result.project_root == layout.root
    assert result.data_root == layout.data_root
    assert result.splits == ("train",)
    assert result.raw_manifest_paths["train"].is_file()
    assert result.normalized_manifest_paths["train"].is_file()
    assert result.filtered_manifest_paths["train"].is_file()
    assert result.processed_manifest_paths["train"].is_file()
    assert result.interim_report_paths["assumption"].is_file()
    assert result.interim_report_paths["filter"].is_file()
    assert result.processed_report_paths["data_quality_json"].is_file()
    assert result.processed_report_paths["split_json"].is_file()
    assert "filtered:train" in result.validation_issues
    assert not any(
        issue.severity == "error"
        for issues in result.validation_issues.values()
        for issue in issues
    )

    processed_records = load_jsonl(result.processed_manifest_paths["train"])
    assert [record["sample_id"] for record in processed_records] == ["train_sample_0-1-rgb_front"]
    assert (result.processed_samples_root / "train/train_sample_0-1-rgb_front.npz").is_file()
    assert result.processed_sample_archive_members["train"] == (
        Path("data/processed/v1/samples/train/train_sample_0-1-rgb_front.npz"),
    )
