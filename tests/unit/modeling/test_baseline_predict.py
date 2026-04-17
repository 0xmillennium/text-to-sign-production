"""Sprint 3 baseline qualitative prediction helper tests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")

import text_to_sign_production.modeling.inference.predict as predict_module  # noqa: E402
from text_to_sign_production.modeling.backbones import TextBackboneOutput  # noqa: E402
from text_to_sign_production.modeling.inference.predict import (  # noqa: E402
    BaselinePredictionError,
    load_baseline_predictor,
    resolve_baseline_checkpoint_path,
)
from text_to_sign_production.modeling.models import BaselineTextToPoseModel  # noqa: E402
from text_to_sign_production.modeling.training.checkpointing import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointMetrics,
    save_training_checkpoint,
)
from text_to_sign_production.modeling.training.config import (  # noqa: E402
    BaselineBackboneConfig,
    BaselineCheckpointConfig,
    BaselineDataConfig,
    BaselineLoopConfig,
    BaselineModelConfig,
    BaselineOptimizerConfig,
    BaselineTrainingConfig,
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


def _config(tmp_path: Path, *, checkpoint_dir: Path) -> BaselineTrainingConfig:
    return BaselineTrainingConfig(
        source_path=tmp_path / "baseline.yaml",
        raw_config={},
        data=BaselineDataConfig(
            train_manifest=tmp_path / "train.jsonl",
            val_manifest=tmp_path / "val.jsonl",
            train_split="train",
            val_split="val",
        ),
        backbone=BaselineBackboneConfig(
            name="dummy",
            max_length=8,
            local_files_only=True,
            freeze=True,
        ),
        model=BaselineModelConfig(
            decoder_hidden_dim=8,
            latent_dim=6,
        ),
        training=BaselineLoopConfig(
            epochs=1,
            batch_size=1,
            shuffle_train=False,
            num_workers=0,
            seed=5,
            device="cpu",
        ),
        optimizer=BaselineOptimizerConfig(
            name="adamw",
            learning_rate=0.001,
            weight_decay=0.0,
        ),
        checkpoint=BaselineCheckpointConfig(output_dir=checkpoint_dir),
    )


def test_resolve_baseline_checkpoint_path_uses_best_checkpoint_by_default(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    best_checkpoint = checkpoint_dir / "best.pt"
    best_checkpoint.write_text("placeholder", encoding="utf-8")
    config = _config(tmp_path, checkpoint_dir=checkpoint_dir)

    assert resolve_baseline_checkpoint_path(config, None) == best_checkpoint.resolve()


def test_resolve_baseline_checkpoint_path_prefers_explicit_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    explicit_checkpoint = tmp_path / "custom.pt"
    explicit_checkpoint.write_text("placeholder", encoding="utf-8")
    config = _config(tmp_path, checkpoint_dir=checkpoint_dir)

    assert (
        resolve_baseline_checkpoint_path(config, explicit_checkpoint)
        == explicit_checkpoint.resolve()
    )


def test_resolve_baseline_checkpoint_path_rejects_missing_checkpoint(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, checkpoint_dir=tmp_path / "missing")

    with pytest.raises(FileNotFoundError, match="Baseline checkpoint"):
        resolve_baseline_checkpoint_path(config, None)


def test_load_baseline_predictor_rejects_incompatible_target_channels(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "bad.pt"
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "epoch": 1,
            "role": "best",
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "config": {},
            "backbone_name": "dummy",
            "seed": None,
            "target_channels": ("body",),
            "train_loss": 0.0,
            "validation_loss": 0.0,
            "validation_metric": 0.0,
        },
        checkpoint_path,
    )
    config = _config(tmp_path, checkpoint_dir=tmp_path)

    with pytest.raises(BaselinePredictionError, match="target_channels"):
        load_baseline_predictor(config, checkpoint_path=checkpoint_path)


def test_load_baseline_predictor_wraps_checkpoint_schema_errors(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "bad-schema.pt"
    torch.save(
        {
            "schema_version": "wrong",
            "epoch": 1,
            "role": "best",
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "config": {},
            "backbone_name": "dummy",
            "seed": None,
            "target_channels": ("body", "left_hand", "right_hand"),
            "train_loss": 0.0,
            "validation_loss": 0.0,
            "validation_metric": 0.0,
        },
        checkpoint_path,
    )
    config = _config(tmp_path, checkpoint_dir=tmp_path)

    with pytest.raises(
        BaselinePredictionError,
        match=f"invalid for baseline prediction export: {checkpoint_path.resolve()}",
    ):
        load_baseline_predictor(config, checkpoint_path=checkpoint_path)


def test_load_baseline_predictor_loads_strict_checkpoint_with_dummy_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "best.pt"
    _write_checkpoint(checkpoint_path)
    config = _config(tmp_path, checkpoint_dir=tmp_path)
    monkeypatch.setattr(predict_module, "build_baseline_model", lambda _config: _model())

    predictor = load_baseline_predictor(config, checkpoint_path=checkpoint_path)

    assert predictor.checkpoint_path == checkpoint_path.resolve()
    assert predictor.device == torch.device("cpu")
    assert not predictor.model.training
    assert predictor.checkpoint_payload["role"] == "best"


def test_load_baseline_predictor_loads_checkpoint_on_cpu_before_device_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_text("placeholder", encoding="utf-8")
    config = _config(tmp_path, checkpoint_dir=tmp_path)
    model = _RecordingModel()
    seen_map_locations: list[object] = []

    def fake_load_training_checkpoint(
        path: Path,
        *,
        map_location: object = "cpu",
    ) -> dict[str, object]:
        assert path == checkpoint_path.resolve()
        seen_map_locations.append(map_location)
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "epoch": 1,
            "role": "best",
            "model_state_dict": {"weight": torch.ones((1,), dtype=torch.float32)},
            "optimizer_state_dict": {},
            "config": {},
            "backbone_name": "dummy",
            "seed": None,
            "target_channels": ("body", "left_hand", "right_hand"),
            "train_loss": 0.0,
            "validation_loss": 0.0,
            "validation_metric": 0.0,
        }

    monkeypatch.setattr(predict_module, "load_training_checkpoint", fake_load_training_checkpoint)
    monkeypatch.setattr(
        predict_module,
        "resolve_training_device",
        lambda _policy: torch.device("cuda"),
    )
    monkeypatch.setattr(predict_module, "build_baseline_model", lambda _config: model)

    predictor = load_baseline_predictor(config, checkpoint_path=checkpoint_path)

    assert seen_map_locations == ["cpu"]
    assert model.loaded_state_dict is not None
    weight = model.loaded_state_dict["weight"]
    assert isinstance(weight, torch.Tensor)
    assert torch.equal(weight, torch.ones((1,), dtype=torch.float32))
    assert model.seen_to_devices == [torch.device("cuda")]
    assert predictor.device == torch.device("cuda")
    assert not predictor.model.training


def test_load_baseline_predictor_rejects_incompatible_model_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "best.pt"
    _write_checkpoint(checkpoint_path)
    config = _config(tmp_path, checkpoint_dir=tmp_path)
    monkeypatch.setattr(
        predict_module,
        "build_baseline_model",
        lambda _config: BaselineTextToPoseModel(
            _DummyBackbone(),
            decoder_hidden_dim=12,
            latent_dim=6,
        ),
    )

    with pytest.raises(BaselinePredictionError, match="incompatible"):
        load_baseline_predictor(config, checkpoint_path=checkpoint_path)


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


class _RecordingModel:
    def __init__(self) -> None:
        self.loaded_state_dict: dict[str, object] | None = None
        self.seen_to_devices: list[torch.device] = []
        self.training = True

    def load_state_dict(
        self,
        state_dict: object,
        strict: bool = True,
        assign: bool = False,
    ) -> object:
        assert strict is True
        assert assign is False
        assert isinstance(state_dict, dict)
        self.loaded_state_dict = state_dict
        return type(
            "_LoadResult",
            (),
            {"missing_keys": [], "unexpected_keys": []},
        )()

    def to(self, *args: object, **kwargs: object) -> _RecordingModel:
        device = kwargs.get("device")
        if isinstance(device, torch.device):
            self.seen_to_devices.append(device)
        return self

    def eval(self) -> _RecordingModel:
        self.training = False
        return self
