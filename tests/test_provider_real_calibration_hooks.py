from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from text_to_sign_production.modeling.backbones.base import TextBackboneOutput
from text_to_sign_production.modeling.candidates import (
    ModelRunRequest,
)
from text_to_sign_production.modeling.candidates.bootstrap import ensure_model_provider_registered
from text_to_sign_production.modeling.candidates.registry import DEFAULT_MODEL_PROVIDER_REGISTRY
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.model.processing.provider import load_model_provider_config
from text_to_sign_production.workflows.model.processing.calibration import (
    build_provider_real_calibration_context,
    run_provider_real_calibration,
    validate_provider_real_trace,
    _trace_from_mapping,
)
from helpers.real_modeling_topology import write_real_modeling_topology


def test_provider_real_calibration_hooks_execute_actual_provider_paths(monkeypatch) -> None:
    _patch_base_direct_flan(monkeypatch)
    cases = {
        "learned_pose_token": {
            "tokenizer_batch_size": [2],
            "text_to_token_batch_size": [2],
            "decode_batch_size": [2],
        },
        "latent_diffusion": {
            "denoiser_batch_size": [2],
        },
        "articulator_aware": {"frame_batch_size": [2]},
        "base_direct": {"batch_size": [2]},
    }

    rows = tuple(
        row
        for provider_key, candidates in cases.items()
        for row in run_provider_real_calibration(
            context=_context(provider_key),
            candidates=candidates,
            max_batches_per_candidate=1,
            warmup_batches=0,
        )
    )

    assert rows
    assert all(row["benchmark_type"] == "provider_real" for row in rows)
    assert all(row["surface_reader_used"] is True for row in rows)
    assert all(row["forward_executed"] is True for row in rows)
    training_rows = tuple(row for row in rows if "generation" not in row["benchmark_path"] and "decode" not in row["benchmark_path"])
    assert all(row["backward_executed"] is True for row in training_rows)
    assert all(row["gradient_observed"] is True for row in training_rows)
    assert all(row["finite_loss_observed"] is True for row in training_rows)
    forward_only = tuple(row for row in rows if "generation" in row["benchmark_path"] or "decode" in row["benchmark_path"])
    assert forward_only
    assert all(row["backward_executed"] is False for row in forward_only)
    assert all(row["forward_only_reason"] == "generation" for row in forward_only)
    assert all(row["surface_batch_fingerprint"] for row in rows)
    assert all(row["model_input_fingerprint"] for row in rows)
    assert all(row["surface_derived_primary_input"] is True for row in rows)
    for row in rows:
        validate_provider_real_trace(_trace_from_mapping(row["provider_real_trace"]))


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
