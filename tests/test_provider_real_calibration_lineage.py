from __future__ import annotations

from dataclasses import replace
import tempfile
from pathlib import Path

import pytest
import torch

from text_to_sign_production.modeling.backbones.base import TextBackboneOutput
from text_to_sign_production.modeling.candidates import ModelRunRequest
from text_to_sign_production.modeling.candidates.bootstrap import ensure_model_provider_registered
from text_to_sign_production.modeling.candidates.registry import DEFAULT_MODEL_PROVIDER_REGISTRY
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.model.contracts.config import ModelWorkflowInvariantError
from text_to_sign_production.workflows.model.processing.calibration import (
    build_provider_real_calibration_context,
    run_provider_real_calibration,
    validate_provider_real_trace,
    _trace_from_mapping,
)
from text_to_sign_production.workflows.model.processing.provider import load_model_provider_config
from helpers.real_modeling_topology import write_real_modeling_topology


def test_learned_tokenizer_calibration_uses_surface_batch_values() -> None:
    row = _row("learned_pose_token", {"tokenizer_batch_size": [2]})
    assert row["surface_batch_fingerprint"]
    assert row["representative_source_manifest_sha256"]
    assert row["representative_provider_config_sha256"] == row["effective_config_hash"]


def test_learned_text_to_token_calibration_uses_surface_batch_values() -> None:
    row = _row("learned_pose_token", {"text_to_token_batch_size": [2]})
    assert row["surface_batch_fingerprint"]
    assert row["representative_surface_kind"] == "text_to_token_sequences"
    assert row["model_input_fingerprint"]


def test_latent_denoiser_calibration_uses_surface_batch_values() -> None:
    row = _row("latent_diffusion", {"denoiser_batch_size": [2]})
    assert row["surface_derived_primary_input"] is True
    assert row["random_used"] is True
    assert row["random_role"] == "noise"
    assert row["representative_surface_kind"] == "latent_sequences"
    assert row["target_fingerprint"]


def test_articulator_calibration_uses_frame_surface_batch_values() -> None:
    row = _row("articulator_aware", {"frame_batch_size": [2]})
    assert row["representative_surface_kind"] == "articulator_frame_units"
    assert row["model_input_fingerprint"]
    assert row["target_fingerprint"]


def test_base_direct_calibration_uses_lazy_loader_batch_values(monkeypatch) -> None:
    _patch_base_direct_flan(monkeypatch)
    row = _row("base_direct", {"batch_size": [2]})
    assert row["benchmark_type"] == "provider_real"
    assert row["batch_adapter"] == "base_direct.calibration.lazy_dataloader_batch"


def test_provider_real_rejects_random_primary_input() -> None:
    trace = _trace("latent_diffusion", {"denoiser_batch_size": [2]})
    with pytest.raises(ModelWorkflowInvariantError, match="primary input"):
        validate_provider_real_trace(
            replace(trace, surface_derived_primary_input=False, random_used=True, random_role=None)
        )


def test_provider_real_rejects_missing_surface_batch_fingerprint() -> None:
    trace = _trace("learned_pose_token", {"tokenizer_batch_size": [2]})
    with pytest.raises(ModelWorkflowInvariantError, match="surface_batch_fingerprint"):
        validate_provider_real_trace(replace(trace, surface_batch_fingerprint=""))


def test_provider_real_rejects_missing_gradient_for_training() -> None:
    trace = _trace("learned_pose_token", {"text_to_token_batch_size": [2]})
    with pytest.raises(ModelWorkflowInvariantError, match="finite gradients"):
        validate_provider_real_trace(replace(trace, gradient_observed=False))


def test_provider_real_rejects_proxy_model_module() -> None:
    trace = _trace("latent_diffusion", {"denoiser_batch_size": [2]})
    with pytest.raises(ModelWorkflowInvariantError, match="provider-owned"):
        validate_provider_real_trace(
            replace(trace, model_module="text_to_sign_production.workflows.model.processing.calibration")
        )


def _row(provider_key: str, candidates: dict[str, list[int]]) -> dict[str, object]:
    return run_provider_real_calibration(
        context=_context(provider_key),
        candidates=candidates,
        max_batches_per_candidate=1,
        warmup_batches=0,
    )[0]


def _context(provider_key: str):
    root = Path(tempfile.mkdtemp(prefix="t2sp-test-real-topology-"))
    topology = write_real_modeling_topology(root / "runtime")
    model_key = ModelKey(provider_key)
    request = ModelRunRequest(
        model_key=model_key,
        run_name="run",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        config_path=Path(f"configs/modeling/{provider_key}.yaml"),
    )
    ensure_model_provider_registered(model_key)
    provider = DEFAULT_MODEL_PROVIDER_REGISTRY.require(model_key)
    loaded = load_model_provider_config(provider, request).loaded_config
    return build_provider_real_calibration_context(
        loaded_config=loaded,
        stage_plan=provider.plan_stages(request, loaded),
        working_dir=root,
        topology=topology,
    )


def _trace(provider_key: str, candidates: dict[str, list[int]]):
    return _trace_from_mapping(_row(provider_key, candidates)["provider_real_trace"])


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
