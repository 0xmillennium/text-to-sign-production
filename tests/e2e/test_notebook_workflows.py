"""E2E smoke tests for notebook-facing workflows."""

from __future__ import annotations

from pathlib import Path

import pytest

from text_to_sign_production.core.paths import ProjectLayout
from text_to_sign_production.data.constants import DEFAULT_FILTER_CONFIG_RELATIVE_PATH
from text_to_sign_production.visualization.skeleton import SkeletonRenderConfig
from text_to_sign_production.workflows.dataset import (
    DatasetWorkflowConfig,
    run_dataset_workflow,
)
from text_to_sign_production.workflows.visualization import (
    VisualizationSelectionConfig,
    VisualizationWorkflowConfig,
    run_visualization_workflow,
    select_visualization_sample,
)

pytestmark = pytest.mark.e2e


def test_dataset_then_visualization_workflows_run_on_tiny_project_root(
    tiny_dataset_workspace: Path,
) -> None:
    layout = ProjectLayout(tiny_dataset_workspace)
    dataset_result = run_dataset_workflow(
        DatasetWorkflowConfig(
            splits=("train",),
            filter_config_path=tiny_dataset_workspace / DEFAULT_FILTER_CONFIG_RELATIVE_PATH,
            project_root=layout.root,
        )
    )

    selected_sample_path = (
        dataset_result.data_root / "processed/v1/samples/train/train_sample_0-1-rgb_front.npz"
    )
    held_sample_path = tiny_dataset_workspace / "outputs/held/train_sample_0-1-rgb_front.npz"
    held_sample_path.parent.mkdir(parents=True, exist_ok=True)
    selected_sample_path.replace(held_sample_path)

    selected_sample = select_visualization_sample(
        VisualizationSelectionConfig(
            data_root=dataset_result.data_root,
            splits=("train",),
            sample_id="train_sample_0-1-rgb_front",
        )
    )

    assert selected_sample.record.split == "train"
    assert not selected_sample.sample_path.exists()
    held_sample_path.replace(selected_sample.sample_path)

    visualization_result = run_visualization_workflow(
        VisualizationWorkflowConfig(
            selection=VisualizationSelectionConfig(
                data_root=dataset_result.data_root,
                splits=("train",),
                sample_id="train_sample_0-1-rgb_front",
            ),
            output_dir=tiny_dataset_workspace / "outputs/visualization",
            mode="skeleton_only",
            fps=12.0,
            skeleton_config=SkeletonRenderConfig(canvas_width=64, canvas_height=48),
        )
    )

    assert dataset_result.processed_manifest_paths["train"].is_file()
    assert visualization_result.output_path.is_file()
