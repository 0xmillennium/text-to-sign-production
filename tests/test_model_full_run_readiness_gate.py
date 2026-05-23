from __future__ import annotations

from pathlib import Path

import pytest
import torch

from text_to_sign_production.modeling.backbones.base import TextBackboneOutput
from text_to_sign_production.modeling.candidates import (
    ModelProviderLoadedConfig,
    ModelRunMode,
)
from text_to_sign_production.modeling.candidates.bootstrap import ensure_model_provider_registered
from text_to_sign_production.modeling.candidates.configs import ModelRunRequest
from text_to_sign_production.modeling.candidates.registry import DEFAULT_MODEL_PROVIDER_REGISTRY
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.model.processing.stages import (
    _guard_full_run_data_surface_readiness,
)
from text_to_sign_production.workflows.model.contracts.config import ModelWorkflowInvariantError
from text_to_sign_production.workflows.model.processing.calibration import run_compute_calibration
from text_to_sign_production.workflows.model.processing.provider import load_model_provider_config
from helpers.real_modeling_topology import write_real_modeling_topology


def test_full_run_guard_requires_runtime_calibration_artifacts() -> None:
    ensure_model_provider_registered(ModelKey.LEARNED_POSE_TOKEN)
    request = ModelRunRequest(
        model_key=ModelKey.LEARNED_POSE_TOKEN,
        run_name="full",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_mode=ModelRunMode.FULL,
    )

    with pytest.raises(ModelWorkflowInvariantError, match="provider_real_calibration_missing"):
        _guard_full_run_data_surface_readiness(request)


def test_full_run_guard_allows_verified_provider_artifacts(tmp_path) -> None:
    for model_key in ModelKey:
        if model_key is ModelKey.BASE_DIRECT:
            continue
        ensure_model_provider_registered(model_key)
        root = tmp_path / model_key.value
        root.mkdir()
        _write_provider_real_artifacts(root, model_key.value)
        request = ModelRunRequest(
            model_key=model_key,
            run_name="full",
            manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
            run_mode=ModelRunMode.FULL,
        )

        _guard_full_run_data_surface_readiness(request, calibration_root=root)


def test_full_run_guard_allows_authoritative_base_direct_artifacts(monkeypatch, tmp_path) -> None:
    _patch_base_direct_flan(monkeypatch)
    root = tmp_path / "base_direct"
    root.mkdir()
    _write_provider_real_artifacts(root, "base_direct")
    request = ModelRunRequest(
        model_key=ModelKey.BASE_DIRECT,
        run_name="full",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_mode=ModelRunMode.FULL,
    )

    _guard_full_run_data_surface_readiness(request, calibration_root=root)


def _write_provider_real_artifacts(root, provider_key: str) -> None:
    model_key = ModelKey(provider_key)
    candidates = {
        "base_direct": {"batch_size": [2]},
        "learned_pose_token": {
            "tokenizer_batch_size": [2],
            "text_to_token_batch_size": [2],
            "reconstruction_batch_size": [2],
            "decode_batch_size": [2],
        },
        "latent_diffusion": {"denoiser_batch_size": [2]},
        "articulator_aware": {"source_batch_size": [2], "frame_batch_size": [2]},
    }[provider_key]
    compute_profile = {
        "name": "test",
        "provider_overrides": {provider_key: {"candidates": candidates}},
    }
    request = ModelRunRequest(
        model_key=model_key,
        run_name="full",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        config_path=Path(f"configs/modeling/{provider_key}.yaml"),
        compute_profile=compute_profile,
        run_mode=ModelRunMode.FULL,
    )
    ensure_model_provider_registered(model_key)
    provider = DEFAULT_MODEL_PROVIDER_REGISTRY.require(model_key)
    loaded_result = load_model_provider_config(provider, request).loaded_config
    effective = dict(loaded_result.effective_config)
    effective["compute_profile"] = compute_profile
    loaded = ModelProviderLoadedConfig(
        model_key=loaded_result.model_key,
        source_path=loaded_result.source_path,
        raw_config=loaded_result.raw_config,
        effective_config=effective,
    )
    run_compute_calibration(
        provider_config=loaded,
        stage_plan=provider.plan_stages(request, loaded),
        output_root=root,
        max_batches_per_candidate=1,
        max_samples=2,
        topology=write_real_modeling_topology(root / "runtime"),
    )


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
