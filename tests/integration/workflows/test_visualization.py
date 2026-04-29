"""Integration tests for the notebook-facing visualization workflow."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.builders.manifests import processed_record, write_jsonl_records
from tests.support.builders.media import write_tiny_decodable_mp4
from tests.support.builders.samples import write_processed_sample_npz
from tests.support.modeling import write_processed_modeling_split
from text_to_sign_production.visualization.skeleton import SkeletonRenderConfig
from text_to_sign_production.workflows.visualization import (
    VisualizationSelectionConfig,
    VisualizationWorkflowConfig,
    VisualizationWorkflowInputError,
    run_visualization_workflow,
    select_visualization_sample,
    validate_visualization_inputs,
)

pytestmark = pytest.mark.integration


def test_manifest_only_selection_then_selected_sample_validation(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    write_jsonl_records(
        data_root / "processed/v1/manifests/train.jsonl",
        [
            processed_record(
                "data/processed/v1/samples/train/sample.npz",
                split="train",
                sample_id="sample",
            )
        ],
    )
    config = VisualizationWorkflowConfig(
        selection=VisualizationSelectionConfig(
            data_root=data_root,
            splits=("train",),
            sample_id="sample",
        ),
        output_dir=tmp_path / "renders",
        mode="skeleton_only",
        skeleton_config=SkeletonRenderConfig(canvas_width=64, canvas_height=48),
    )

    selected = select_visualization_sample(config.selection)

    assert selected.record.split == "train"
    assert not selected.sample_path.exists()
    with pytest.raises(
        VisualizationWorkflowInputError,
        match="Extract dataset_build_samples_train\\.tar\\.zst",
    ):
        validate_visualization_inputs(config)

    write_processed_sample_npz(selected.sample_path)

    validate_visualization_inputs(config)


def test_visualization_workflow_renders_skeleton_only_video(tmp_path: Path) -> None:
    write_processed_modeling_split(tmp_path, split="train", sample_ids=("sample",))

    result = run_visualization_workflow(
        VisualizationWorkflowConfig(
            selection=VisualizationSelectionConfig(
                data_root=tmp_path / "data",
                splits=("train",),
                sample_id="sample",
            ),
            output_dir=tmp_path / "renders",
            mode="skeleton_only",
            fps=12.0,
            skeleton_config=SkeletonRenderConfig(canvas_width=64, canvas_height=48),
        )
    )

    assert result.output_path.is_file()
    assert result.render_metadata["mode"] == "skeleton_only"
    assert result.render_metadata["skeleton_frames"] == 2


def test_visualization_workflow_renders_side_by_side_with_existing_source_video(
    tmp_path: Path,
) -> None:
    write_processed_modeling_split(tmp_path, split="train", sample_ids=("sample",))
    write_tiny_decodable_mp4(
        tmp_path / "data/raw/how2sign/bfh_keypoints/train/sample.mp4",
        frame_count=2,
        width=32,
        height=24,
        fps=12.0,
    )

    result = run_visualization_workflow(
        VisualizationWorkflowConfig(
            selection=VisualizationSelectionConfig(
                data_root=tmp_path / "data",
                splits=("train",),
                sample_id="sample",
            ),
            output_dir=tmp_path / "renders",
            mode="side_by_side",
            skeleton_config=SkeletonRenderConfig(canvas_width=64, canvas_height=48),
        )
    )

    assert result.output_path.is_file()
    assert result.render_metadata["mode"] == "side_by_side"
    assert result.render_metadata["video_frames"] == 2
