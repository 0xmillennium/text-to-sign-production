"""Unit tests for the notebook-facing visualization workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import text_to_sign_production.workflows.visualization as visualization_workflow_mod
from tests.support.builders.manifests import processed_record, write_jsonl_records
from tests.support.builders.samples import write_processed_sample_npz
from tests.support.modeling import write_processed_modeling_split
from text_to_sign_production.visualization.skeleton import SkeletonRenderConfig

pytestmark = pytest.mark.unit


def test_select_visualization_sample_exact_sample_id_works(tmp_path: Path) -> None:
    write_processed_modeling_split(tmp_path, split="train", sample_ids=("sample-a",))

    selected = visualization_workflow_mod.select_visualization_sample(
        visualization_workflow_mod.VisualizationSelectionConfig(
            data_root=tmp_path / "data",
            splits=("train",),
            sample_id="sample-a",
        )
    )

    assert selected.record.sample_id == "sample-a"
    assert (
        selected.sample_path
        == (tmp_path / "data/processed/v1/samples/train/sample-a.npz").resolve()
    )


def test_select_visualization_sample_uses_manifest_only_by_default(tmp_path: Path) -> None:
    data_root = _write_manifest_only_processed_split(tmp_path, sample_ids=("sample-a",))

    selected = visualization_workflow_mod.select_visualization_sample(
        visualization_workflow_mod.VisualizationSelectionConfig(
            data_root=data_root,
            splits=("train",),
            sample_id="sample-a",
        )
    )

    assert selected.record.sample_id == "sample-a"
    assert selected.record.split == "train"
    assert (
        selected.sample_path
        == (tmp_path / "data/processed/v1/samples/train/sample-a.npz").resolve()
    )
    assert not selected.sample_path.exists()


def test_select_visualization_sample_random_selection_is_deterministic(tmp_path: Path) -> None:
    data_root = _write_manifest_only_processed_split(
        tmp_path,
        sample_ids=("sample-a", "sample-b", "sample-c"),
    )
    config = visualization_workflow_mod.VisualizationSelectionConfig(
        data_root=data_root,
        splits=("train",),
        random_selection=True,
        seed=11,
    )

    first = visualization_workflow_mod.select_visualization_sample(config)
    second = visualization_workflow_mod.select_visualization_sample(config)

    assert first.record.sample_id == second.record.sample_id
    assert first.record.split == "train"
    assert not first.sample_path.exists()


def test_validate_visualization_inputs_rejects_conflicting_selection_modes() -> None:
    with pytest.raises(visualization_workflow_mod.VisualizationWorkflowInputError, match="one"):
        visualization_workflow_mod.validate_visualization_inputs(
            visualization_workflow_mod.VisualizationWorkflowConfig(
                selection=visualization_workflow_mod.VisualizationSelectionConfig(
                    data_root="data",
                    sample_id="sample",
                    random_selection=True,
                ),
                output_dir="outputs",
            )
        )


def test_validate_visualization_inputs_requires_selected_sample_file(tmp_path: Path) -> None:
    data_root = _write_manifest_only_processed_split(tmp_path, sample_ids=("sample-a",))
    config = visualization_workflow_mod.VisualizationWorkflowConfig(
        selection=visualization_workflow_mod.VisualizationSelectionConfig(
            data_root=data_root,
            splits=("train",),
            sample_id="sample-a",
        ),
        output_dir=tmp_path / "renders",
    )

    with pytest.raises(
        visualization_workflow_mod.VisualizationWorkflowInputError,
        match=(
            "Selected processed sample file is missing. "
            "Extract dataset_build_samples_train\\.tar\\.zst"
        ),
    ):
        visualization_workflow_mod.validate_visualization_inputs(config)

    write_processed_sample_npz(data_root / "processed/v1/samples/train/sample-a.npz")

    visualization_workflow_mod.validate_visualization_inputs(config)


def test_skeleton_only_workflow_dispatches_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_processed_modeling_split(tmp_path, split="train", sample_ids=("sample-a",))
    seen: dict[str, Any] = {}

    def fake_render_skeleton_video(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"mp4")
        return {"mode": "skeleton_only", "output_path": str(kwargs["output_path"])}

    monkeypatch.setattr(
        visualization_workflow_mod,
        "render_skeleton_video",
        fake_render_skeleton_video,
    )

    result = visualization_workflow_mod.run_visualization_workflow(
        visualization_workflow_mod.VisualizationWorkflowConfig(
            selection=visualization_workflow_mod.VisualizationSelectionConfig(
                data_root=tmp_path / "data",
                splits=("train",),
                sample_id="sample-a",
            ),
            output_dir=tmp_path / "renders",
            mode="skeleton_only",
            fps=12.0,
            skeleton_config=SkeletonRenderConfig(canvas_width=64, canvas_height=48),
        )
    )

    assert result.output_path == (tmp_path / "renders/sample-a__skeleton_only.mp4").resolve()
    assert seen["fps"] == 12.0
    assert seen["label"] == "sample-a"
    assert result.render_metadata["mode"] == "skeleton_only"


def test_side_by_side_workflow_fails_clearly_when_source_video_is_missing(
    tmp_path: Path,
) -> None:
    write_processed_modeling_split(tmp_path, split="train", sample_ids=("sample-a",))

    with pytest.raises(
        visualization_workflow_mod.VisualizationWorkflowInputError,
        match=(
            "^Source video is missing\\. Extract the required raw BFH archive "
            "in the notebook/runtime first\\.$"
        ),
    ):
        visualization_workflow_mod.run_visualization_workflow(
            visualization_workflow_mod.VisualizationWorkflowConfig(
                selection=visualization_workflow_mod.VisualizationSelectionConfig(
                    data_root=tmp_path / "data",
                    splits=("train",),
                    sample_id="sample-a",
                ),
                output_dir=tmp_path / "renders",
                mode="side_by_side",
                skeleton_config=SkeletonRenderConfig(canvas_width=64, canvas_height=48),
            )
        )


def _write_manifest_only_processed_split(
    root: Path,
    *,
    split: str = "train",
    sample_ids: tuple[str, ...],
) -> Path:
    data_root = root / "data"
    records = [
        processed_record(
            f"data/processed/v1/samples/{split}/{sample_id}.npz",
            split=split,
            sample_id=sample_id,
            text=f"text for {sample_id}",
        )
        for sample_id in sample_ids
    ]
    write_jsonl_records(data_root / f"processed/v1/manifests/{split}.jsonl", records)
    return data_root
