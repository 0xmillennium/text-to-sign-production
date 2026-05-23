from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from text_to_sign_production.artifacts.store import build_artifact_topology
from text_to_sign_production.core import build_repo_roots
from text_to_sign_production.modeling.candidates import ModelStageExecutionError
from text_to_sign_production.modeling.candidates.base_direct import exporter
from text_to_sign_production.modeling.training.config import (
    BaselineTrainingConfigError,
    load_baseline_training_config,
)


@pytest.mark.unit
def test_baseline_training_config_validate_paths_flag_preserves_training_strictness(
    tmp_path: Path,
) -> None:
    source = Path("src/text_to_sign_production/modeling/config/baseline.yaml")
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        source.read_text(encoding="utf-8")
        .replace("data/manifests/untiered/passed/train.json", "missing/train.json")
        .replace("data/manifests/untiered/passed/val.json", "missing/val.json"),
        encoding="utf-8",
    )

    with pytest.raises(BaselineTrainingConfigError, match="data.train_manifest"):
        load_baseline_training_config(
            config_path,
            validate_paths=True,
            repo_root=tmp_path,
        )

    loaded = load_baseline_training_config(
        config_path,
        validate_paths=False,
        checkpoint_output_dir=tmp_path / "checkpoints",
        repo_root=tmp_path,
    )

    assert loaded.data.train_manifest.name == "train.json"
    assert loaded.data.val_manifest.name == "val.json"


@pytest.mark.unit
def test_base_direct_single_sample_inference_loads_predictor_without_config_path_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    topology = build_artifact_topology(build_repo_roots(tmp_path / "runtime"))
    context = SimpleNamespace(
        checkpoint_path=checkpoint_path,
        topology=topology,
        request=SimpleNamespace(run_name="run001"),
        output_root=tmp_path / "out",
    )
    calls: list[dict[str, object]] = []

    def fake_load_base_direct_predictor(**kwargs):
        calls.append(kwargs)
        raise ModelStageExecutionError("stop after predictor load")

    monkeypatch.setattr(
        exporter,
        "load_base_direct_predictor",
        fake_load_base_direct_predictor,
    )

    with pytest.raises(ModelStageExecutionError, match="stop after predictor load"):
        exporter.infer_base_direct_single_sample(
            context=context,
            config=object(),
        )

    assert calls
    assert calls[0]["validate_config_paths"] is False
