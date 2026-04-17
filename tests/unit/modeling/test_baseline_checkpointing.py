"""Sprint 3 baseline checkpointing and provenance tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

torch: Any = pytest.importorskip("torch")

import text_to_sign_production.data.utils as utils_mod  # noqa: E402
import text_to_sign_production.modeling.training.train as train_module  # noqa: E402
from tests.support.builders.manifests import processed_record, write_jsonl_records  # noqa: E402
from tests.support.builders.samples import write_processed_sample_npz  # noqa: E402
from text_to_sign_production.modeling.backbones import TextBackboneOutput  # noqa: E402
from text_to_sign_production.modeling.models import BaselineTextToPoseModel  # noqa: E402
from text_to_sign_production.modeling.training.checkpointing import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointMetrics,
    load_training_checkpoint,
    save_training_checkpoint,
    should_replace_best_checkpoint,
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


def test_save_and_load_training_checkpoint_round_trips_payload(tmp_path: Path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    path = tmp_path / "checkpoint.pt"

    saved_path = save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        role="last",
        config_summary={"backbone": {"name": "dummy"}},
        backbone_name="dummy",
        seed=123,
        metrics=CheckpointMetrics(
            train_loss=1.0,
            validation_loss=2.0,
            validation_metric=3.0,
        ),
    )
    loaded = load_training_checkpoint(saved_path)

    assert loaded["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert loaded["epoch"] == 1
    assert loaded["role"] == "last"
    assert loaded["backbone_name"] == "dummy"
    assert loaded["seed"] == 123
    assert loaded["train_loss"] == pytest.approx(1.0)
    assert loaded["validation_loss"] == pytest.approx(2.0)
    assert loaded["validation_metric"] == pytest.approx(3.0)
    assert loaded["target_channels"] == ("body", "left_hand", "right_hand")
    assert isinstance(loaded["model_state_dict"], dict)
    assert isinstance(loaded["optimizer_state_dict"], dict)


def test_load_training_checkpoint_rejects_bad_payload(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "schema_version": "wrong",
            "epoch": 1,
            "role": "last",
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "config": {},
            "backbone_name": "dummy",
            "seed": None,
            "target_channels": (),
            "train_loss": 0.0,
            "validation_loss": 0.0,
            "validation_metric": 0.0,
        },
        path,
    )

    with pytest.raises(ValueError, match="schema version mismatch"):
        load_training_checkpoint(path)


def test_load_training_checkpoint_uses_weights_only_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoint.pt"
    captured_kwargs: dict[str, Any] = {}

    def fake_torch_load(load_path: Path, **kwargs: Any) -> dict[str, Any]:
        assert load_path == path
        captured_kwargs.update(kwargs)
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "epoch": 1,
            "role": "last",
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "config": {},
            "backbone_name": "dummy",
            "seed": None,
            "target_channels": ("body", "left_hand", "right_hand"),
            "train_loss": 0.0,
            "validation_loss": 0.0,
            "validation_metric": 0.0,
        }

    monkeypatch.setattr(torch, "load", fake_torch_load)

    load_training_checkpoint(path)

    assert captured_kwargs["weights_only"] is True
    assert captured_kwargs["map_location"] == "cpu"


def test_best_checkpoint_selection_prefers_lower_validation_loss_and_keeps_ties() -> None:
    assert should_replace_best_checkpoint(
        candidate_validation_loss=2.0,
        best_validation_loss=None,
    )
    assert should_replace_best_checkpoint(
        candidate_validation_loss=1.0,
        best_validation_loss=2.0,
    )
    assert not should_replace_best_checkpoint(
        candidate_validation_loss=2.0,
        best_validation_loss=2.0,
    )
    assert not should_replace_best_checkpoint(
        candidate_validation_loss=3.0,
        best_validation_loss=2.0,
    )


def test_run_baseline_training_writes_runtime_provenance_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    train_sample_path = "data/processed/v1/samples/train/train-sample.npz"
    val_sample_path = "data/processed/v1/samples/val/val-sample.npz"
    write_processed_sample_npz(tmp_path / train_sample_path)
    write_processed_sample_npz(tmp_path / val_sample_path)
    train_manifest_path = tmp_path / "data/processed/v1/manifests/train.jsonl"
    val_manifest_path = tmp_path / "data/processed/v1/manifests/val.jsonl"
    write_jsonl_records(
        train_manifest_path,
        [
            processed_record(
                train_sample_path,
                split="train",
                sample_id="train-sample",
            )
        ],
    )
    write_jsonl_records(
        val_manifest_path,
        [
            processed_record(
                val_sample_path,
                split="val",
                sample_id="val-sample",
            )
        ],
    )
    config_path = tmp_path / "baseline.yaml"
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
    monkeypatch.setattr(train_module, "build_baseline_model", lambda _config: _model())

    result = train_module.run_baseline_training(config_path)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert result.last_checkpoint_path.is_file()
    assert result.best_checkpoint_path is not None
    assert result.best_checkpoint_path.is_file()
    assert summary["config_path"] == config_path.resolve().as_posix()
    assert summary["config"]["backbone"]["name"] == "dummy"
    assert summary["backbone_name"] == "dummy"
    assert summary["seed"] == 5
    assert summary["checkpoint_output_path"].endswith("outputs/modeling/baseline-test")
    assert summary["last_checkpoint_path"].endswith("last.pt")
    assert summary["best_checkpoint_path"].endswith("best.pt")
    assert isinstance(summary["final_train_loss"], float)
    assert isinstance(summary["final_validation_loss"], float)
    assert isinstance(summary["final_validation_metric"], float)
