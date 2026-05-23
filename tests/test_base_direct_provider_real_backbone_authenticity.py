from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from text_to_sign_production.modeling.candidates import ModelRunMode, ModelRunRequest
from text_to_sign_production.modeling.candidates.base_direct.config import (
    load_base_direct_config,
)
from text_to_sign_production.modeling.candidates.base_direct.model import (
    build_base_direct_model,
)
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.model.contracts.compute_profiles import (
    load_model_compute_profile,
)
from text_to_sign_production.workflows.model.processing.calibration import (
    ProviderRealCalibrationTrace,
    _is_authoritative_provider_real_row,
    calibration_row_from_provider_real_trace,
    provider_real_trace_id,
    stable_config_hash,
)

ROOT = Path(__file__).resolve().parents[1]


def _config():
    request = ModelRunRequest(
        model_key=ModelKey.BASE_DIRECT,
        run_name="base-direct-authenticity",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_mode=ModelRunMode.FULL,
        config_path=ROOT / "configs/modeling/base_direct.yaml",
        compute_profile=load_model_compute_profile(ROOT, "colab_a100_80gb").to_dict(),
    )
    return load_base_direct_config(request.config_path, request=request)


def test_build_base_direct_model_does_not_fallback_to_calibration_backbone(monkeypatch) -> None:
    class FailingBackbone:
        def __init__(self, **_kwargs):
            raise OSError("missing local model")

    monkeypatch.setattr(
        "text_to_sign_production.modeling.backbones.flan_t5.FlanT5TextBackbone",
        FailingBackbone,
    )
    config = _config()

    with pytest.raises(OSError, match="missing local model"):
        build_base_direct_model(config)


def test_base_direct_provider_real_calibration_fails_when_flan_backbone_unavailable(monkeypatch) -> None:
    class FailingBackbone:
        def __init__(self, **_kwargs):
            raise RuntimeError("flan unavailable")

    monkeypatch.setattr(
        "text_to_sign_production.modeling.backbones.flan_t5.FlanT5TextBackbone",
        FailingBackbone,
    )

    with pytest.raises(RuntimeError, match="flan unavailable"):
        build_base_direct_model(_config())


def _trace(**overrides) -> ProviderRealCalibrationTrace:
    spec = {"provider_key": "base_direct"}
    values = {
        "provider_key": "base_direct",
        "provider_hook": "BaseDirectProviderRealBenchmark",
        "benchmark_name": "base_direct.lazy_dataloader.forward_backward",
        "benchmark_type": "provider_real",
        "benchmark_kind": "training",
        "forward_only_reason": None,
        "surface_kind": "processed_pose_dataset",
        "surface_metadata_path": ROOT / "configs/modeling/base_direct.yaml",
        "surface_metadata_sha256": "a" * 64,
        "surface_batch_fingerprint": "b" * 64,
        "batch_adapter": "base_direct.calibration.lazy_dataloader_batch",
        "model_input_fingerprint": "c" * 64,
        "target_fingerprint": "d" * 64,
        "model_class": "BaselineTextToPoseModel",
        "model_module": "text_to_sign_production.modeling.models.baseline",
        "model_config_hash": "e" * 64,
        "effective_config_hash": "f" * 64,
        "model_build_spec_hash": stable_config_hash(spec),
        "surface_schema_hash": "2" * 64,
        "surface_provider_config_sha256": "f" * 64,
        "surface_feature_dim": 128,
        "model_expected_input_dim": 128,
        "model_factory": "text_to_sign_production.modeling.candidates.base_direct.model.build_base_direct_model",
        "model_build_spec": spec,
        "loss_name": "compute_base_direct_loss",
        "loss_module": "text_to_sign_production.modeling.candidates.base_direct.losses",
        "training_step_helper": "text_to_sign_production.modeling.candidates.base_direct.calibration._base_direct_provider_real_step",
        "forward_executed": True,
        "backward_executed": True,
        "gradient_observed": True,
        "finite_loss_observed": True,
        "surface_derived_primary_input": True,
        "random_used": False,
        "random_role": None,
        "units_processed": 1,
        "elapsed_seconds": 0.1,
        "peak_cuda_memory_reserved_fraction": None,
    }
    values.update(overrides)
    return ProviderRealCalibrationTrace(**values)
