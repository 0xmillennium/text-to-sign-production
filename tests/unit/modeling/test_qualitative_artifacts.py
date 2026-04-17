"""Sprint 3 qualitative artifact and evidence tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

torch: Any = pytest.importorskip("torch")

import text_to_sign_production.data.utils as utils_mod  # noqa: E402
import text_to_sign_production.modeling.inference.predict as predict_module  # noqa: E402
from tests.support.builders.manifests import processed_record, write_jsonl_records  # noqa: E402
from tests.support.builders.samples import write_processed_sample_npz  # noqa: E402
from text_to_sign_production.modeling.backbones import TextBackboneOutput  # noqa: E402
from text_to_sign_production.modeling.data import ProcessedPoseItem  # noqa: E402
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
    BaselineTextToPoseModel,
)
from text_to_sign_production.modeling.training.checkpointing import (  # noqa: E402
    CheckpointMetrics,
    save_training_checkpoint,
)

pytestmark = pytest.mark.unit


class _DummyBackbone:
    output_dim = 4

    def __call__(
        self,
        texts: Sequence[str],
        *,
        device: Any | None = None,
    ) -> TextBackboneOutput:
        resolved_device = torch.device("cpu") if device is None else torch.device(device)
        batch_size = len(texts)
        pooled_embedding = torch.ones((batch_size, self.output_dim), device=resolved_device)
        return TextBackboneOutput(
            token_embeddings=pooled_embedding.unsqueeze(1),
            pooled_embedding=pooled_embedding,
            attention_mask=torch.ones((batch_size, 1), dtype=torch.long, device=resolved_device),
        )


def _model() -> BaselineTextToPoseModel:
    return BaselineTextToPoseModel(
        _DummyBackbone(),
        decoder_hidden_dim=8,
        latent_dim=6,
    )


def _item() -> ProcessedPoseItem:
    return ProcessedPoseItem(
        sample_id="sample one",
        split="val",
        text="A raw English sentence.",
        fps=24.0,
        num_frames=2,
        sample_path=Path("/tmp/sample-one.npz"),
        sample_path_value="data/processed/v1/samples/val/sample one.npz",
        body=np.full((2, 25, 2), 1.0, dtype=np.float32),
        left_hand=np.full((2, 21, 2), 2.0, dtype=np.float32),
        right_hand=np.full((2, 21, 2), 3.0, dtype=np.float32),
        frame_valid_mask=np.asarray([True, False], dtype=np.bool_),
    )


def _prediction() -> BaselinePoseOutput:
    return BaselinePoseOutput(
        body=torch.full((1, 2, 25, 2), 10.0, dtype=torch.float32),
        left_hand=torch.full((1, 2, 21, 2), 20.0, dtype=torch.float32),
        right_hand=torch.full((1, 2, 21, 2), 30.0, dtype=torch.float32),
    )


def test_write_qualitative_sample_artifacts_preserves_channel_separated_npz(
    tmp_path: Path,
) -> None:
    record = write_qualitative_sample_artifacts(
        tmp_path,
        index=0,
        item=_item(),
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
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    run_summary_path = tmp_path / "run_summary.json"
    run_summary_path.write_text(
        json.dumps(
            {
                "checkpoint_output_path": "outputs/modeling/baseline",
                "last_checkpoint_path": "outputs/modeling/baseline/last.pt",
                "best_checkpoint_path": "outputs/modeling/baseline/best.pt",
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
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    _write_processed_split(tmp_path, split="train", sample_ids=["train-sample"])
    _write_processed_split(tmp_path, split="val", sample_ids=["sample-z", "sample-a"])
    config_path = _write_baseline_config(tmp_path)
    checkpoint_dir = tmp_path / "outputs/modeling/baseline-test"
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
    _write_checkpoint(checkpoint_dir / "best.pt")
    monkeypatch.setattr(predict_module, "build_baseline_model", lambda _config: _model())

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
    assert summary["checkpoint_path"] == "outputs/modeling/baseline-test/best.pt"
    assert summary["sample_count"] == 1
    assert summary["sample_ids"] == ["sample-a"]
    evidence = json.loads(result.evidence_bundle_path.read_text(encoding="utf-8"))
    assert evidence["config_path"] == "baseline.yaml"
    assert evidence["run_summary_path"] == "outputs/modeling/baseline-test/run_summary.json"
    assert evidence["export_checkpoint"]["path"] == "outputs/modeling/baseline-test/best.pt"
    assert evidence["qualitative_panel"]["panel_definition_path"] == "exports/panel_definition.json"
    assert evidence["qualitative_panel"]["panel_summary_path"] == "exports/panel_summary.json"
    assert evidence["qualitative_panel"]["records_path"] == "exports/records.jsonl"
    assert evidence["qualitative_panel"]["sample_count"] == 1


def _write_processed_split(root: Path, *, split: str, sample_ids: list[str]) -> None:
    records: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        sample_path = f"data/processed/v1/samples/{split}/{sample_id}.npz"
        write_processed_sample_npz(root / sample_path)
        records.append(
            processed_record(
                sample_path,
                split=split,
                sample_id=sample_id,
                text=f"text for {sample_id}",
            )
        )
    write_jsonl_records(root / f"data/processed/v1/manifests/{split}.jsonl", records)


def _write_baseline_config(root: Path) -> Path:
    config_path = root / "baseline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "train_manifest": "data/processed/v1/manifests/train.jsonl",
                    "val_manifest": "data/processed/v1/manifests/val.jsonl",
                    "train_split": "train",
                    "val_split": "val",
                },
                "backbone": {
                    "name": "dummy",
                    "max_length": 8,
                    "local_files_only": True,
                    "freeze": True,
                },
                "model": {
                    "decoder_hidden_dim": 8,
                    "latent_dim": 6,
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 1,
                    "shuffle_train": False,
                    "num_workers": 0,
                    "seed": 5,
                    "device": "cpu",
                },
                "optimizer": {
                    "name": "adamw",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                },
                "checkpoint": {
                    "output_dir": "outputs/modeling/baseline-test",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _write_checkpoint(path: Path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        role="best",
        config_summary={"backbone": {"name": "dummy"}},
        backbone_name="dummy",
        seed=5,
        metrics=CheckpointMetrics(
            train_loss=1.0,
            validation_loss=2.0,
            validation_metric=3.0,
        ),
    )
