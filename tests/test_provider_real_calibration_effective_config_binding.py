from __future__ import annotations

from dataclasses import replace
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from text_to_sign_production.modeling.backbones.base import TextBackboneOutput
from text_to_sign_production.modeling.candidates import (
    ModelProviderLoadedConfig,
    ModelRunMode,
    ModelRunRequest,
)
from text_to_sign_production.modeling.candidates.articulator_aware.provider import (
    ArticulatorAwareProvider,
)
from text_to_sign_production.modeling.candidates.base_direct.provider import (
    BaseDirectProvider,
)
from text_to_sign_production.modeling.candidates.bootstrap import ensure_model_provider_registered
from text_to_sign_production.modeling.candidates.latent_diffusion.provider import (
    LatentDiffusionProvider,
)
from text_to_sign_production.modeling.candidates.registry import DEFAULT_MODEL_PROVIDER_REGISTRY
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.model.processing.provider import load_model_provider_config
from text_to_sign_production.workflows.model.contracts.config import ModelWorkflowInvariantError
from text_to_sign_production.workflows.model.contracts.compute_profiles import (
    load_model_compute_profile,
    provider_candidate_overrides,
)
from text_to_sign_production.workflows.model.processing.calibration import (
    build_provider_real_calibration_context,
    inspect_provider_real_calibration_artifacts,
    provider_config_hash,
    provider_model_build_spec_hash,
    run_compute_calibration,
    run_provider_real_calibration,
    validate_provider_real_trace,
    _provider_calibration_surface_schema,
    _provider_calibration_model_build_spec,
    _trace_from_mapping,
)
from helpers.real_modeling_topology import write_real_modeling_topology


ROOT = Path(__file__).resolve().parents[1]


def test_provider_real_hook_requires_context() -> None:
    with pytest.raises(TypeError):
        run_provider_real_calibration(  # type: ignore[call-arg]
            candidates={"denoiser_batch_size": [2]},
            max_batches_per_candidate=1,
        )


def test_base_direct_full_a100_effective_config_contains_compute_profile() -> None:
    request = ModelRunRequest(
        model_key=ModelKey.BASE_DIRECT,
        run_name="test-base-direct-full-a100",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_mode=ModelRunMode.FULL,
        config_path=ROOT / "configs/modeling/base_direct.yaml",
        compute_profile=load_model_compute_profile(ROOT, "colab_a100_80gb").to_dict(),
    )

    loaded = BaseDirectProvider().load_config(request)

    assert isinstance(loaded.effective_config["compute_profile"], dict)


def test_articulator_aware_full_a100_effective_config_contains_compute_profile() -> None:
    request = ModelRunRequest(
        model_key=ModelKey.ARTICULATOR_AWARE,
        run_name="test-articulator-aware-full-a100",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_mode=ModelRunMode.FULL,
        config_path=ROOT / "configs/modeling/articulator_aware.yaml",
        compute_profile=load_model_compute_profile(ROOT, "colab_a100_80gb").to_dict(),
    )

    loaded = ArticulatorAwareProvider().load_config(request)

    assert isinstance(loaded.effective_config["compute_profile"], dict)


def test_latent_diffusion_calibration_policy_matches_profile_candidates() -> None:
    request = ModelRunRequest(
        model_key=ModelKey.LATENT_DIFFUSION,
        run_name="test-latent-diffusion-full-a100",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_mode=ModelRunMode.FULL,
        config_path=ROOT / "configs/modeling/latent_diffusion.yaml",
        compute_profile=load_model_compute_profile(ROOT, "colab_a100_80gb").to_dict(),
    )
    provider = LatentDiffusionProvider()
    loaded = provider.load_config(request)

    policy = provider.calibration_policy(loaded)
    candidates = provider_candidate_overrides(
        request.compute_profile,
        provider_key="latent_diffusion",
    )

    assert set(candidates) == set(policy.candidate_keys)
    assert "generation_batch_size" not in policy.candidate_keys


def test_provider_real_calibration_context_contains_effective_config_and_build_spec_hashes() -> None:
    loaded, plan = _loaded_and_plan("latent_diffusion")
    context = build_provider_real_calibration_context(
        loaded_config=loaded,
        stage_plan=plan,
        working_dir=Path("."),
        topology=write_real_modeling_topology(Path(tempfile.mkdtemp()) / "runtime"),
    )

    assert context.provider_config_hash == provider_config_hash(loaded.effective_config)
    assert context.model_build_spec_hash == provider_model_build_spec_hash(
        "latent_diffusion",
        effective_config=loaded.effective_config,
    )


def test_provider_real_trace_rejects_context_hash_mismatches() -> None:
    context = _context("latent_diffusion")
    row = run_provider_real_calibration(
        context=context,
        candidates={"denoiser_batch_size": [2]},
        max_batches_per_candidate=1,
        warmup_batches=0,
    )[0]
    trace = _trace_from_mapping(row["provider_real_trace"])

    with pytest.raises(ModelWorkflowInvariantError, match="effective_config_hash"):
        validate_provider_real_trace(
            replace(trace, effective_config_hash="0" * 64, surface_provider_config_sha256="0" * 64),
            context=context,
        )
    with pytest.raises(ModelWorkflowInvariantError, match="model_build_spec_hash"):
        validate_provider_real_trace(replace(trace, model_build_spec_hash="1" * 64), context=context)
    with pytest.raises(ModelWorkflowInvariantError, match="surface provider config hash"):
        validate_provider_real_trace(replace(trace, surface_provider_config_sha256="2" * 64), context=context)
    with pytest.raises(ModelWorkflowInvariantError, match="surface/model input dimension"):
        validate_provider_real_trace(replace(trace, model_expected_input_dim=999), context=context)


@pytest.mark.parametrize(
    ("provider_key", "candidates"),
    (
        ("learned_pose_token", {"tokenizer_batch_size": [2], "text_to_token_batch_size": [2]}),
        ("latent_diffusion", {"denoiser_batch_size": [2]}),
        ("articulator_aware", {"frame_batch_size": [2]}),
        ("base_direct", {"batch_size": [2]}),
    ),
)
def test_provider_real_uses_mutated_effective_config(
    monkeypatch,
    provider_key: str,
    candidates: dict[str, list[int]],
) -> None:
    if provider_key == "base_direct":
        _patch_base_direct_flan(monkeypatch)
    context = _context(provider_key)
    rows = run_provider_real_calibration(
        context=context,
        candidates=candidates,
        max_batches_per_candidate=1,
        warmup_batches=0,
    )

    assert rows
    for row in rows:
        assert row["effective_config_hash"] == context.provider_config_hash
        assert row["model_build_spec_hash"] == context.model_build_spec_hash
        assert row["surface_provider_config_sha256"] == context.provider_config_hash
        assert row["surface_schema_hash"]
        assert row["benchmark_type"] == "provider_real"
        assert row["model_factory"]
        if row["benchmark_kind"] == "training":
            assert row["training_step_helper"]
        spec_values = set(_flatten_scalars(row["model_build_spec"]))
        raw_values = set(_flatten_scalars(_raw_yaml(provider_key)))
        for forbidden in (13, 37, 11, 23, 5, 64, 32, 8, 16):
            if forbidden not in raw_values:
                assert forbidden not in spec_values


def test_provider_real_model_build_spec_changes_when_config_changes() -> None:
    config = _effective_config("learned_pose_token")
    changed = dict(config)
    changed["tokenizer"] = {**config["tokenizer"], "hidden_dim": 41}  # type: ignore[index]

    assert provider_model_build_spec_hash("learned_pose_token", effective_config=config) != (
        provider_model_build_spec_hash("learned_pose_token", effective_config=changed)
    )


def test_readiness_rejects_stale_calibration_config_and_build_spec(tmp_path: Path) -> None:
    loaded, plan = _loaded_and_plan("learned_pose_token")
    run_compute_calibration(
        provider_config=loaded,
        stage_plan=plan,
        output_root=tmp_path,
        max_batches_per_candidate=1,
        max_samples=2,
        topology=write_real_modeling_topology(tmp_path / "runtime"),
    )

    stale = inspect_provider_real_calibration_artifacts(
        calibration_root=tmp_path,
        provider_key="learned_pose_token",
        expected_provider_config_hash="0" * 64,
        expected_model_build_spec_hash="1" * 64,
    )

    assert "calibration_effective_config_hash_mismatch" in stale.blocking_issues
    assert "calibration_model_build_spec_hash_mismatch" in stale.blocking_issues
    assert "calibration_not_bound_to_effective_config" in stale.blocking_issues
    assert "selected_overrides_trace_id_mismatch" in stale.blocking_issues


def test_learned_provider_real_spec_uses_real_yaml_text_encoder() -> None:
    yaml_config = _raw_yaml("learned_pose_token")
    spec = _build_spec("learned_pose_token")

    assert spec["text_to_token"]["text_embedding_dim"] == yaml_config["text_encoder"]["embedding_dim"]  # type: ignore[index]
    assert spec["text_to_token"]["max_text_length"] == yaml_config["text_encoder"]["max_length"]  # type: ignore[index]


def test_learned_provider_real_spec_uses_real_yaml_tokenizer_and_codebook() -> None:
    yaml_config = _raw_yaml("learned_pose_token")
    spec = _build_spec("learned_pose_token")

    assert spec["tokenizer"]["tokenizer_hidden_dim"] == yaml_config["tokenizer"]["hidden_dim"]  # type: ignore[index]
    assert spec["tokenizer"]["tokenizer_latent_dim"] == yaml_config["tokenizer"]["latent_dim"]  # type: ignore[index]
    assert spec["tokenizer"]["codebook_size"] == yaml_config["codebook"]["size"]  # type: ignore[index]


def test_learned_provider_real_spec_missing_required_key_fails_fast() -> None:
    config = _without_path(_effective_config("learned_pose_token"), "text_encoder.embedding_dim")
    with pytest.raises(ModelWorkflowInvariantError, match="text_encoder.embedding_dim"):
        _build_spec_from_effective("learned_pose_token", config)


def test_latent_provider_real_spec_uses_real_yaml_text_encoder() -> None:
    yaml_config = _raw_yaml("latent_diffusion")
    spec = _build_spec("latent_diffusion")

    assert spec["denoiser"]["text_embedding_dim"] == yaml_config["text_encoder"]["embedding_dim"]  # type: ignore[index]


def test_latent_provider_real_spec_uses_real_yaml_denoiser() -> None:
    yaml_config = _raw_yaml("latent_diffusion")
    spec = _build_spec("latent_diffusion")

    assert spec["denoiser"]["denoiser_hidden_dim"] == yaml_config["denoiser"]["hidden_dim"]  # type: ignore[index]
    assert spec["denoiser"]["timestep_embedding_dim"] == yaml_config["denoiser"]["timestep_embedding_dim"]  # type: ignore[index]
    assert spec["denoiser"]["position_embedding_dim"] == yaml_config["denoiser"]["position_embedding_dim"]  # type: ignore[index]
    assert spec["denoiser"]["max_positions"] == yaml_config["denoiser"]["max_positions"]  # type: ignore[index]


def test_latent_provider_real_spec_uses_real_yaml_diffusion_and_latent_target() -> None:
    yaml_config = _raw_yaml("latent_diffusion")
    spec = _build_spec("latent_diffusion")

    assert spec["denoiser"]["diffusion_steps"] == yaml_config["diffusion"]["timesteps"]  # type: ignore[index]
    assert spec["autoencoder"]["window_size"] == yaml_config["latent_target"]["window_size"]  # type: ignore[index]


def test_latent_provider_real_spec_missing_required_key_fails_fast() -> None:
    config = _without_path(_effective_config("latent_diffusion"), "provider_config.diffusion.timesteps")
    with pytest.raises(ModelWorkflowInvariantError, match="provider_config.diffusion.timesteps"):
        _build_spec_from_effective("latent_diffusion", config)


def test_articulator_provider_real_spec_uses_real_yaml_structure_variant() -> None:
    yaml_config = _raw_yaml("articulator_aware")
    spec = _build_spec("articulator_aware")

    assert spec["frame_model"]["architecture"] == yaml_config["structure_variant"]["architecture"]  # type: ignore[index]
    assert spec["frame_model"]["hidden_dim"] == yaml_config["structure_variant"]["hidden_dim"]  # type: ignore[index]


def test_articulator_provider_real_spec_uses_real_yaml_text_embedding_policy() -> None:
    yaml_config = _raw_yaml("articulator_aware")
    spec = _build_spec("articulator_aware")

    assert spec["frame_model"]["text_embedding_dim"] == yaml_config["text_encoder"]["embedding_dim"]  # type: ignore[index]


def test_articulator_provider_real_spec_missing_required_key_fails_fast() -> None:
    config = _without_path(_effective_config("articulator_aware"), "provider_config.structure_variant.hidden_dim")
    with pytest.raises(ModelWorkflowInvariantError, match="provider_config.structure_variant.hidden_dim"):
        _build_spec_from_effective("articulator_aware", config)


def test_base_direct_provider_real_uses_real_build_base_direct_model(monkeypatch) -> None:
    _patch_base_direct_flan(monkeypatch)
    context = _context("base_direct")
    rows = run_provider_real_calibration(
        context=context,
        candidates={"batch_size": [2]},
        max_batches_per_candidate=1,
        warmup_batches=0,
    )

    assert rows
    assert all(row["benchmark_type"] == "provider_real" for row in rows)
    assert all(row["model_factory"].endswith("build_base_direct_model") for row in rows)


def test_base_direct_rows_are_authoritative_only_from_provider_real_trace(monkeypatch) -> None:
    _patch_base_direct_flan(monkeypatch)
    context = _context("base_direct")
    row = run_provider_real_calibration(
        context=context,
        candidates={"batch_size": [2]},
        max_batches_per_candidate=1,
        warmup_batches=0,
    )[0]

    assert row["benchmark_type"] == "provider_real"
    assert row["trace_validation_passed"] is True


def test_provider_real_acceptance_tests_load_real_config_files() -> None:
    assert all(_config_path(provider_key).is_file() for provider_key in _CONFIG_PATHS)


def test_provider_real_tests_do_not_define_local_authority_configs() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    keys = ("text_embedding_dim", "hidden_dim", "latent_dim", "codebook_size", "window_size")
    values = ("13", "37", "11", "23", "5")
    forbidden = tuple(f'"{key}": {value}' for key, value in zip(keys, values, strict=True))
    assert [item for item in forbidden if item in source] == []


def test_provider_real_calibration_no_hardcoded_tiny_model_specs() -> None:
    forbidden = (
        "input_dim=" + "64",
        "hidden_dim=" + "32",
        "latent_dim=" + "8",
        "codebook_size=" + "16",
        "text_embedding_dim=32",
        "provider_config_sha256=\"b\"",
        "writer.append_units({\"values\"",
    )
    matches: list[str] = []
    for path in Path("src/text_to_sign_production/modeling/candidates").glob("*/calibration.py"):
        text = path.read_text(encoding="utf-8")
        matches.extend(f"{path}:{item}" for item in forbidden if item in text)

    assert matches == []


def _patch_base_direct_flan(monkeypatch) -> None:
    class FlanT5TextBackbone(torch.nn.Module):
        output_dim = 8

        def __init__(self, **_kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((1,)))

        def forward(self, texts, *, device=None):
            resolved = torch.device("cpu" if device is None else device)
            batch = len(texts)
            pooled = self.weight.expand(batch, self.output_dim).to(resolved)
            return TextBackboneOutput(
                token_embeddings=pooled.unsqueeze(1),
                pooled_embedding=pooled,
                attention_mask=torch.ones((batch, 1), dtype=torch.bool, device=resolved),
            )

    FlanT5TextBackbone.__module__ = "text_to_sign_production.modeling.backbones.flan_t5"
    monkeypatch.setattr(
        "text_to_sign_production.modeling.backbones.flan_t5.FlanT5TextBackbone",
        FlanT5TextBackbone,
    )


def _context(provider_key: str):
    loaded, plan = _loaded_and_plan(provider_key)
    root = Path(tempfile.mkdtemp(prefix="t2sp-test-real-topology-"))
    topology = write_real_modeling_topology(root / "runtime")
    return build_provider_real_calibration_context(
        loaded_config=loaded,
        stage_plan=plan,
        working_dir=root,
        topology=topology,
    )


_CONFIG_PATHS = {
    "learned_pose_token": Path("configs/modeling/learned_pose_token.yaml"),
    "latent_diffusion": Path("configs/modeling/latent_diffusion.yaml"),
    "articulator_aware": Path("configs/modeling/articulator_aware.yaml"),
    "base_direct": Path("configs/modeling/base_direct.yaml"),
}


def _loaded_and_plan(provider_key: str):
    model_key = ModelKey(provider_key)
    request = ModelRunRequest(
        model_key=model_key,
        run_name="run",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        config_path=_config_path(provider_key),
        compute_profile={
            "name": "test",
            "provider_overrides": {provider_key: {"candidates": _candidates(provider_key)}},
        },
    )
    ensure_model_provider_registered(model_key)
    provider = DEFAULT_MODEL_PROVIDER_REGISTRY.require(model_key)
    result = load_model_provider_config(provider, request)
    effective = dict(result.loaded_config.effective_config)
    effective["compute_profile"] = request.compute_profile
    loaded = ModelProviderLoadedConfig(
        model_key=model_key,
        source_path=result.loaded_config.source_path,
        raw_config=result.loaded_config.raw_config,
        effective_config=effective,
    )
    return loaded, provider.plan_stages(request, loaded)


def _effective_config(provider_key: str) -> dict[str, object]:
    loaded, _plan = _loaded_and_plan(provider_key)
    return dict(loaded.effective_config)


def _config_path(provider_key: str) -> Path:
    return _CONFIG_PATHS[provider_key]


def _raw_yaml(provider_key: str) -> dict[str, object]:
    payload = yaml.safe_load(_config_path(provider_key).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _build_spec(provider_key: str) -> dict[str, object]:
    return dict(_build_spec_from_effective(provider_key, _effective_config(provider_key)))


def _build_spec_from_effective(provider_key: str, effective_config: dict[str, object]):
    surface_schema = _provider_calibration_surface_schema(
        provider_key,
        effective_config=effective_config,
    )
    return _provider_calibration_model_build_spec(
        provider_key,
        effective_config=effective_config,
        surface_schema=surface_schema,
    )


def _without_path(config: dict[str, object], path: str) -> dict[str, object]:
    changed = _mutable_jsonable(config)
    current: Any = changed
    parts = path.split(".")
    for part in parts[:-1]:
        current = current[part]
    del current[parts[-1]]
    return changed


def _mutable_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _mutable_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mutable_jsonable(item) for item in value]
    if hasattr(value, "items"):
        return {str(key): _mutable_jsonable(item) for key, item in value.items()}
    return value


def _candidates(provider_key: str) -> dict[str, list[int]]:
    return {
        "learned_pose_token": {
            "tokenizer_batch_size": [2],
            "text_to_token_batch_size": [2],
            "reconstruction_batch_size": [2],
            "decode_batch_size": [2],
        },
        "latent_diffusion": {"denoiser_batch_size": [2]},
        "articulator_aware": {"frame_batch_size": [2]},
        "base_direct": {"batch_size": [2]},
    }[provider_key]


def _flatten_scalars(value: Any) -> tuple[object, ...]:
    if isinstance(value, dict):
        return tuple(item for child in value.values() for item in _flatten_scalars(child))
    if isinstance(value, (list, tuple)):
        return tuple(item for child in value for item in _flatten_scalars(child))
    return (value,)
