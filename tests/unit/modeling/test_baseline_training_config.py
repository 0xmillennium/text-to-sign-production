"""Sprint 3 baseline training config tests."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

import text_to_sign_production.data.utils as utils_mod
from text_to_sign_production.modeling.config import DEFAULT_BASELINE_CONFIG_PATH
from text_to_sign_production.modeling.training.config import (
    BaselineTrainingConfigError,
    load_baseline_training_config,
)

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _valid_config() -> dict[str, Any]:
    return {
        "data": {
            "train_manifest": "data/processed/v1/manifests/train.jsonl",
            "val_manifest": "data/processed/v1/manifests/val.jsonl",
            "train_split": "train",
            "val_split": "val",
        },
        "backbone": {
            "name": "google/flan-t5-base",
            "max_length": 16,
            "local_files_only": True,
            "freeze": True,
        },
        "model": {
            "decoder_hidden_dim": 8,
            "latent_dim": 6,
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "shuffle_train": False,
            "num_workers": 0,
            "seed": 123,
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
    }


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _touch_manifest(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_load_baseline_training_config_resolves_repo_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    config_path = tmp_path / "config.yaml"
    config_payload = _valid_config()
    _write_yaml(config_path, config_payload)
    _touch_manifest(tmp_path, "data/processed/v1/manifests/train.jsonl")
    _touch_manifest(tmp_path, "data/processed/v1/manifests/val.jsonl")

    config = load_baseline_training_config(config_path)

    assert config.source_path == config_path.resolve()
    assert (
        config.data.train_manifest
        == (tmp_path / "data/processed/v1/manifests/train.jsonl").resolve()
    )
    assert (
        config.data.val_manifest == (tmp_path / "data/processed/v1/manifests/val.jsonl").resolve()
    )
    assert config.backbone.name == "google/flan-t5-base"
    assert config.training.seed == 123
    assert config.checkpoint.output_dir == (tmp_path / "outputs/modeling/baseline-test").resolve()


def test_load_baseline_training_config_rejects_missing_required_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    config_payload = _valid_config()
    del config_payload["data"]["train_manifest"]
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, config_payload)

    with pytest.raises(BaselineTrainingConfigError, match="data.train_manifest"):
        load_baseline_training_config(config_path)


def test_load_baseline_training_config_allows_null_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    config_payload = _valid_config()
    config_payload["training"]["seed"] = None
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, config_payload)

    config = load_baseline_training_config(config_path, validate_paths=False)

    assert config.training.seed is None


def test_load_baseline_training_config_requires_seed_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    config_payload = _valid_config()
    del config_payload["training"]["seed"]
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, config_payload)

    with pytest.raises(BaselineTrainingConfigError, match="training.seed"):
        load_baseline_training_config(config_path, validate_paths=False)


def test_load_baseline_training_config_rejects_malformed_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("data: [broken\n", encoding="utf-8")

    with pytest.raises(BaselineTrainingConfigError, match="Could not parse YAML config"):
        load_baseline_training_config(config_path, validate_paths=False)


def test_load_baseline_training_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(BaselineTrainingConfigError, match="root must be a mapping"):
        load_baseline_training_config(config_path, validate_paths=False)


def test_default_baseline_config_parses_without_path_validation() -> None:
    config = load_baseline_training_config(
        DEFAULT_BASELINE_CONFIG_PATH,
        validate_paths=False,
    )

    assert config.backbone.name == "google/flan-t5-base"
    assert config.data.train_split == "train"
    assert config.data.val_split == "val"
    assert config.optimizer.name == "adamw"


def test_package_data_includes_default_baseline_yaml() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    package_data = pyproject["tool"]["setuptools"]["package-data"]

    assert "*.yaml" in package_data["text_to_sign_production.modeling.config"]
