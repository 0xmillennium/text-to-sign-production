from __future__ import annotations

import json
from pathlib import Path

import pytest

from text_to_sign_production.modeling.candidates import (
    MODEL_RUNTIME_SUPPORT_SCHEMA_VERSION,
    ModelRuntimeSupportManifest,
    read_runtime_support_manifest,
    support_artifact_from_model_run_file,
    write_runtime_support_manifest,
)


@pytest.mark.unit
def test_runtime_support_artifact_hash_and_size_match_file(tmp_path: Path) -> None:
    run_root = tmp_path / "models" / "base_direct" / "run001"
    artifact_path = run_root / "intermediates" / "config" / "baseline_training_compat.yaml"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("batch_size: 1\n", encoding="utf-8")

    artifact = support_artifact_from_model_run_file(
        model_run_root=run_root,
        path=artifact_path,
        role="base_direct_compatibility_config",
        provider_key="base_direct",
    )

    assert artifact.relative_path.as_posix() == (
        "intermediates/config/baseline_training_compat.yaml"
    )
    assert artifact.size_bytes == artifact_path.stat().st_size
    assert len(artifact.sha256) == 64


@pytest.mark.unit
def test_runtime_support_manifest_round_trips(tmp_path: Path) -> None:
    run_root = tmp_path / "models" / "learned_pose_token" / "run001"
    stats_path = run_root / "intermediates" / "representation" / "standardization_stats.json"
    stats_path.parent.mkdir(parents=True)
    stats_path.write_text(json.dumps({"schema_version": "test"}) + "\n", encoding="utf-8")
    artifact = support_artifact_from_model_run_file(
        model_run_root=run_root,
        path=stats_path,
        role="learned_pose_token_standardization_stats",
        provider_key="learned_pose_token",
    )
    manifest = ModelRuntimeSupportManifest(
        schema_version=MODEL_RUNTIME_SUPPORT_SCHEMA_VERSION,
        model_key="learned_pose_token",
        model_run_name="run001",
        manifest_family="tiered:clean:included",
        provider_config_kind="learned_pose_token.effective_config",
        artifacts=(artifact,),
    )
    path = run_root / "runtime_support_manifest.json"

    write_runtime_support_manifest(path, manifest)
    loaded = read_runtime_support_manifest(path)

    assert loaded == manifest


@pytest.mark.unit
def test_missing_required_support_artifact_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        support_artifact_from_model_run_file(
            model_run_root=tmp_path / "models" / "latent_diffusion" / "run001",
            path=tmp_path / "models" / "latent_diffusion" / "run001" / "missing.json",
            role="latent_diffusion_target_spec",
            provider_key="latent_diffusion",
        )
