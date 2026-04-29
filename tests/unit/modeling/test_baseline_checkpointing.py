"""Sprint 3 baseline checkpointing and provenance tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")

import text_to_sign_production.modeling.training.train as train_module  # noqa: E402
from tests.support.modeling import (  # noqa: E402
    patch_modeling_repo_root,
    write_tiny_baseline_modeling_workspace,
)
from tests.support.modeling_torch import build_dummy_baseline_model  # noqa: E402
from text_to_sign_production.modeling.training.checkpointing import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointMetrics,
    load_training_checkpoint,
    save_training_checkpoint,
    should_replace_best_checkpoint,
)

pytestmark = pytest.mark.unit


def test_save_and_load_training_checkpoint_round_trips_payload(tmp_path: Path) -> None:
    model = build_dummy_baseline_model()
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
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
    workspace = write_tiny_baseline_modeling_workspace(tmp_path)
    config_path = workspace.config_path
    monkeypatch.setattr(
        train_module,
        "build_baseline_model",
        lambda _config: build_dummy_baseline_model(),
    )

    result = train_module.run_baseline_training(config_path)
    captured = capsys.readouterr()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert result.last_checkpoint_path.is_file()
    assert result.best_checkpoint_path is not None
    assert result.best_checkpoint_path.is_file()
    assert summary["config_path"] == config_path.resolve().as_posix()
    assert summary["config"]["backbone"]["name"] == "dummy"
    assert summary["backbone_name"] == "dummy"
    assert summary["seed"] == 5
    assert summary["checkpoint_output_path"].endswith(
        "data/artifacts/base/baseline-test/checkpoints"
    )
    assert summary["last_checkpoint_path"].endswith("last.pt")
    assert summary["best_checkpoint_path"].endswith("best.pt")
    assert isinstance(summary["final_train_loss"], float)
    assert isinstance(summary["final_validation_loss"], float)
    assert isinstance(summary["final_validation_metric"], float)
    assert "[baseline train] epoch 1/1 start" in captured.out
    assert "[baseline train] epoch 1/1 summary:" in captured.out
    assert "best_checkpoint_updated=yes" in captured.out
