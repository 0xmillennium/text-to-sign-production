"""Sprint 3 qualitative artifact and evidence tests."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch: Any = pytest.importorskip("torch")

import text_to_sign_production.modeling.inference.predict as predict_module  # noqa: E402
import text_to_sign_production.modeling.inference.qualitative as qualitative_module  # noqa: E402
from tests.support.modeling import (  # noqa: E402
    patch_modeling_repo_root,
    write_tiny_baseline_modeling_workspace,
)
from tests.support.modeling_torch import (  # noqa: E402
    build_dummy_baseline_model,
    pose_surface,
    processed_pose_item,
    write_dummy_training_checkpoint,
)
from text_to_sign_production.modeling.inference.evidence import (  # noqa: E402
    BASELINE_EVIDENCE_SCHEMA_VERSION,
    write_baseline_evidence_bundle,
)
from text_to_sign_production.modeling.inference.qualitative import (  # noqa: E402
    export_qualitative_panel,
    write_qualitative_sample_artifacts,
)
from text_to_sign_production.modeling.models import (  # noqa: E402
    BaselinePoseOutput,
)

pytestmark = pytest.mark.unit


def _prediction() -> BaselinePoseOutput:
    return BaselinePoseOutput(
        body=pose_surface("body", fill_value=10.0),
        left_hand=pose_surface("left_hand", fill_value=20.0),
        right_hand=pose_surface("right_hand", fill_value=30.0),
    )


def test_write_qualitative_sample_artifacts_preserves_channel_separated_npz(
    tmp_path: Path,
) -> None:
    record = write_qualitative_sample_artifacts(
        tmp_path,
        index=0,
        item=processed_pose_item(),
        prediction=_prediction(),
    )

    reference_path = Path(str(record["reference_artifact"]))
    prediction_path = Path(str(record["prediction_artifact"]))
    assert reference_path.name == "0000__sample_one.npz"
    assert prediction_path.name == "0000__sample_one.npz"
    assert record["sample_id"] == "sample one"
    assert record["text"] == "A raw English sentence."
    assert record["frame_valid_count"] == 1
    assert record["target_channels"] == ["body", "left_hand", "right_hand"]

    with np.load(reference_path, allow_pickle=False) as reference:
        assert set(reference.files) == {"body", "left_hand", "right_hand", "frame_valid_mask"}
        assert tuple(reference["body"].shape) == (2, 25, 2)
        assert reference["body"].mean() == pytest.approx(1.0)
        assert "face" not in reference.files
    with np.load(prediction_path, allow_pickle=False) as prediction:
        assert set(prediction.files) == {"body", "left_hand", "right_hand", "frame_valid_mask"}
        assert tuple(prediction["left_hand"].shape) == (2, 21, 2)
        assert prediction["left_hand"].mean() == pytest.approx(20.0)
        assert "face" not in prediction.files


def test_write_baseline_evidence_bundle_ties_runtime_outputs_to_run_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
    run_summary_path = tmp_path / "run_summary.json"
    run_summary_path.write_text(
        json.dumps(
            {
                "checkpoint_output_path": "data/artifacts/base/baseline/checkpoints",
                "last_checkpoint_path": "data/artifacts/base/baseline/checkpoints/last.pt",
                "best_checkpoint_path": "data/artifacts/base/baseline/checkpoints/best.pt",
            }
        ),
        encoding="utf-8",
    )

    evidence_path = write_baseline_evidence_bundle(
        tmp_path / "baseline_evidence_bundle.json",
        config_path=tmp_path / "baseline.yaml",
        run_summary_path=run_summary_path,
        checkpoint_path=tmp_path / "best.pt",
        checkpoint_payload={
            "role": "best",
            "epoch": 2,
            "backbone_name": "dummy",
            "seed": 13,
            "train_loss": 1.0,
            "validation_loss": 2.0,
            "validation_metric": 3.0,
        },
        panel_definition_path=tmp_path / "panel_definition.json",
        panel_summary_path=tmp_path / "panel_summary.json",
        records_path=tmp_path / "records.jsonl",
        sample_ids=("sample-a", "sample-b"),
        target_channels=("body", "left_hand", "right_hand"),
    )

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == BASELINE_EVIDENCE_SCHEMA_VERSION
    assert evidence["config_path"] == "baseline.yaml"
    assert evidence["run_summary_path"] == "run_summary.json"
    assert evidence["export_checkpoint"]["path"] == "best.pt"
    assert evidence["export_checkpoint"]["role"] == "best"
    assert evidence["export_checkpoint"]["metrics"]["validation_loss"] == pytest.approx(2.0)
    assert evidence["phase5_checkpoint_references"]["best_checkpoint_path"].endswith("best.pt")
    assert evidence["qualitative_panel"]["panel_definition_path"] == "panel_definition.json"
    assert evidence["qualitative_panel"]["panel_summary_path"] == "panel_summary.json"
    assert evidence["qualitative_panel"]["records_path"] == "records.jsonl"
    assert evidence["qualitative_panel"]["sample_ids"] == ["sample-a", "sample-b"]
    assert evidence["qualitative_panel"]["target_channels"] == [
        "body",
        "left_hand",
        "right_hand",
    ]


def test_export_qualitative_panel_writes_panel_artifacts_with_dummy_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
    workspace = write_tiny_baseline_modeling_workspace(
        tmp_path,
        train_sample_ids=("train-sample",),
        val_sample_ids=("sample-z", "sample-a"),
    )
    config_path = workspace.config_path
    checkpoint_dir = tmp_path / "data/artifacts/base/baseline-test/checkpoints"
    checkpoint_dir.mkdir(parents=True)
    run_summary_path = checkpoint_dir / "run_summary.json"
    run_summary_path.write_text(
        json.dumps(
            {
                "checkpoint_output_path": checkpoint_dir.as_posix(),
                "last_checkpoint_path": (checkpoint_dir / "last.pt").as_posix(),
                "best_checkpoint_path": (checkpoint_dir / "best.pt").as_posix(),
            }
        ),
        encoding="utf-8",
    )
    write_dummy_training_checkpoint(checkpoint_dir / "best.pt")
    monkeypatch.setattr(
        predict_module,
        "build_baseline_model",
        lambda _config: build_dummy_baseline_model(),
    )
    progress_calls: list[dict[str, object]] = []

    def fake_iter_with_progress(
        iterable: Iterable[Any],
        *,
        total: int | None,
        desc: str,
        unit: str,
    ) -> Iterator[Any]:
        progress_calls.append({"total": total, "desc": desc, "unit": unit})
        return iter(iterable)

    monkeypatch.setattr(qualitative_module, "iter_with_progress", fake_iter_with_progress)

    result = export_qualitative_panel(
        config_path,
        output_dir=tmp_path / "exports",
        panel_size=1,
    )

    assert result.sample_ids == ("sample-a",)
    assert result.sample_count == 1
    assert result.panel_definition_path.is_file()
    assert result.records_path.is_file()
    assert result.panel_summary_path.is_file()
    assert result.evidence_bundle_path.is_file()
    records = [
        json.loads(line) for line in result.records_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["sample_id"] == "sample-a"
    assert records[0]["reference_artifact"] == "exports/references/0000__sample-a.npz"
    assert records[0]["prediction_artifact"] == "exports/predictions/0000__sample-a.npz"
    assert (tmp_path / str(records[0]["reference_artifact"])).is_file()
    assert (tmp_path / str(records[0]["prediction_artifact"])).is_file()
    summary = json.loads(result.panel_summary_path.read_text(encoding="utf-8"))
    assert summary["panel_definition_path"] == "exports/panel_definition.json"
    assert summary["records_path"] == "exports/records.jsonl"
    assert summary["checkpoint_path"] == "data/artifacts/base/baseline-test/checkpoints/best.pt"
    assert summary["sample_count"] == 1
    assert summary["sample_ids"] == ["sample-a"]
    evidence = json.loads(result.evidence_bundle_path.read_text(encoding="utf-8"))
    assert evidence["config_path"] == "baseline.yaml"
    assert (
        evidence["run_summary_path"]
        == "data/artifacts/base/baseline-test/checkpoints/run_summary.json"
    )
    assert (
        evidence["export_checkpoint"]["path"]
        == "data/artifacts/base/baseline-test/checkpoints/best.pt"
    )
    assert evidence["qualitative_panel"]["panel_definition_path"] == "exports/panel_definition.json"
    assert evidence["qualitative_panel"]["panel_summary_path"] == "exports/panel_summary.json"
    assert evidence["qualitative_panel"]["records_path"] == "exports/records.jsonl"
    assert evidence["qualitative_panel"]["sample_count"] == 1
    assert progress_calls == [
        {
            "total": 1,
            "desc": "[baseline qualitative] Export validation panel",
            "unit": "sample",
        }
    ]
