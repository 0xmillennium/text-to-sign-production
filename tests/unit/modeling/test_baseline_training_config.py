"""Sprint 3 baseline training config tests."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

from tests.support.modeling import (
    baseline_config_payload,
    patch_modeling_repo_root,
    write_baseline_config_payload,
)
from text_to_sign_production.modeling.config import DEFAULT_BASELINE_CONFIG_PATH
from text_to_sign_production.modeling.training.config import (
    BaselineTrainingConfigError,
    load_baseline_training_config,
)

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _valid_config() -> dict[str, Any]:
    payload = baseline_config_payload(backbone_name="google/flan-t5-base")
    payload["backbone"]["max_length"] = 16
    payload["training"]["batch_size"] = 2
    payload["training"]["seed"] = 123
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    write_baseline_config_payload(path, payload)


def _touch_manifest(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_load_baseline_training_config_resolves_repo_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
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
    assert config.training.pin_memory is False
    assert config.training.persistent_workers is False
    assert config.training.prefetch_factor is None
    assert config.training.non_blocking_transfers is False
    assert (
        config.checkpoint.output_dir
        == (tmp_path / "data/artifacts/base/baseline-test/checkpoints").resolve()
    )


def test_load_baseline_training_config_rejects_missing_required_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
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
    patch_modeling_repo_root(monkeypatch, tmp_path)
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
    patch_modeling_repo_root(monkeypatch, tmp_path)
    config_payload = _valid_config()
    del config_payload["training"]["seed"]
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, config_payload)

    with pytest.raises(BaselineTrainingConfigError, match="training.seed"):
        load_baseline_training_config(config_path, validate_paths=False)


def test_load_baseline_training_config_rejects_persistent_workers_without_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
    config_payload = _valid_config()
    config_payload["training"]["persistent_workers"] = True
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, config_payload)

    with pytest.raises(BaselineTrainingConfigError, match="persistent_workers"):
        load_baseline_training_config(config_path, validate_paths=False)


def test_load_baseline_training_config_rejects_prefetch_factor_without_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
    config_payload = _valid_config()
    config_payload["training"]["prefetch_factor"] = 2
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, config_payload)

    with pytest.raises(BaselineTrainingConfigError, match="prefetch_factor"):
        load_baseline_training_config(config_path, validate_paths=False)


def test_load_baseline_training_config_rejects_invalid_prefetch_factor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
    config_payload = _valid_config()
    config_payload["training"]["num_workers"] = 2
    config_payload["training"]["prefetch_factor"] = 0
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, config_payload)

    with pytest.raises(BaselineTrainingConfigError, match="prefetch_factor"):
        load_baseline_training_config(config_path, validate_paths=False)


def test_load_baseline_training_config_accepts_worker_tuning_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
    config_payload = _valid_config()
    config_payload["training"].update(
        {
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "non_blocking_transfers": True,
        }
    )
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, config_payload)

    config = load_baseline_training_config(config_path, validate_paths=False)

    assert config.training.num_workers == 2
    assert config.training.pin_memory is True
    assert config.training.persistent_workers is True
    assert config.training.prefetch_factor == 2
    assert config.training.non_blocking_transfers is True


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
    assert config.backbone.freeze is True
    assert config.training.pin_memory is False
    assert config.training.persistent_workers is False
    assert config.training.prefetch_factor is None
    assert config.training.non_blocking_transfers is False


def test_package_data_includes_default_baseline_yaml() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    package_data = pyproject["tool"]["setuptools"]["package-data"]

    assert "*.yaml" in package_data["text_to_sign_production.modeling.config"]
