"""Sprint 3 baseline qualitative prediction helper tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")

import text_to_sign_production.modeling.inference.predict as predict_module  # noqa: E402
from tests.support.modeling import baseline_training_config  # noqa: E402
from tests.support.modeling_torch import (  # noqa: E402
    build_dummy_baseline_model,
    write_dummy_training_checkpoint,
)
from text_to_sign_production.modeling.inference.predict import (  # noqa: E402
    BaselinePredictionError,
    load_baseline_predictor,
    resolve_baseline_checkpoint_path,
)
from text_to_sign_production.modeling.training.checkpointing import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
)

pytestmark = pytest.mark.unit


def test_resolve_baseline_checkpoint_path_uses_best_checkpoint_by_default(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    best_checkpoint = checkpoint_dir / "best.pt"
    best_checkpoint.write_text("placeholder", encoding="utf-8")
    config = baseline_training_config(tmp_path, checkpoint_dir=checkpoint_dir)

    assert resolve_baseline_checkpoint_path(config, None) == best_checkpoint.resolve()


def test_resolve_baseline_checkpoint_path_prefers_explicit_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    explicit_checkpoint = tmp_path / "custom.pt"
    explicit_checkpoint.write_text("placeholder", encoding="utf-8")
    config = baseline_training_config(tmp_path, checkpoint_dir=checkpoint_dir)

    assert (
        resolve_baseline_checkpoint_path(config, explicit_checkpoint)
        == explicit_checkpoint.resolve()
    )


def test_resolve_baseline_checkpoint_path_rejects_missing_checkpoint(
    tmp_path: Path,
) -> None:
    config = baseline_training_config(tmp_path, checkpoint_dir=tmp_path / "missing")

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
    config = baseline_training_config(tmp_path, checkpoint_dir=tmp_path)

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
    config = baseline_training_config(tmp_path, checkpoint_dir=tmp_path)

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
    write_dummy_training_checkpoint(checkpoint_path)
    config = baseline_training_config(tmp_path, checkpoint_dir=tmp_path)
    monkeypatch.setattr(
        predict_module,
        "build_baseline_model",
        lambda _config: build_dummy_baseline_model(),
    )

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
    config = baseline_training_config(tmp_path, checkpoint_dir=tmp_path)
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
    write_dummy_training_checkpoint(checkpoint_path)
    config = baseline_training_config(tmp_path, checkpoint_dir=tmp_path)
    monkeypatch.setattr(
        predict_module,
        "build_baseline_model",
        lambda _config: build_dummy_baseline_model(decoder_hidden_dim=12),
    )

    with pytest.raises(BaselinePredictionError, match="incompatible"):
        load_baseline_predictor(config, checkpoint_path=checkpoint_path)


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
