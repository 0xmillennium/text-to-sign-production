from __future__ import annotations

from pathlib import Path

import pytest

from text_to_sign_production.modeling.candidates import (
    ModelExecutionResult,
    ModelRunMode,
    ModelRunRequest,
    ModelStageResult,
    ModelStageStatus,
    default_stage_plan_for_request,
)
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.model.contracts.compute_profiles import (
    load_model_compute_profile,
)
from text_to_sign_production.workflows.model.processing.calibration import (
    ProviderRealCalibrationArtifactStatus,
)
from text_to_sign_production.workflows.model.processing.reports import (
    _a100_readiness_summary_payload,
)

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FIELDS = {
    ModelKey.BASE_DIRECT: ("batch_size", "num_workers"),
    ModelKey.LEARNED_POSE_TOKEN: (
        "tokenizer_batch_size",
        "text_to_token_batch_size",
        "reconstruction_batch_size",
        "decode_batch_size",
        "cache_materialized_sources",
        "num_workers",
    ),
    ModelKey.LATENT_DIFFUSION: ("denoiser_batch_size", "num_workers"),
    ModelKey.ARTICULATOR_AWARE: ("source_batch_size", "frame_batch_size", "num_workers"),
}
ACTIVE = {
    ModelKey.BASE_DIRECT: {"batch_size": 8},
    ModelKey.LEARNED_POSE_TOKEN: {
        "tokenizer_batch_size": 512,
        "text_to_token_batch_size": 256,
        "reconstruction_batch_size": 256,
        "decode_batch_size": 256,
        "cache_materialized_sources": True,
    },
    ModelKey.LATENT_DIFFUSION: {"denoiser_batch_size": 128},
    ModelKey.ARTICULATOR_AWARE: {"source_batch_size": 64, "frame_batch_size": 2048},
}
MISSING_FIELD = {
    ModelKey.BASE_DIRECT: "batch_size",
    ModelKey.LEARNED_POSE_TOKEN: "decode_batch_size",
    ModelKey.LATENT_DIFFUSION: "denoiser_batch_size",
    ModelKey.ARTICULATOR_AWARE: "frame_batch_size",
}


def _application(model_key: ModelKey) -> dict[str, object]:
    return {
        "active_requested": dict(ACTIVE[model_key]),
        "active_applied": dict(ACTIVE[model_key]),
        "active_not_applicable": {},
        "active_unsupported": {},
        "telemetry_required_fields": list(REQUIRED_FIELDS[model_key]),
    }


def _performance(model_key: ModelKey) -> dict[str, object]:
    payload = {
        "stage_elapsed_seconds": 1.0,
        "unit_type": "sample",
        "unit_count": 10,
        "units_per_second": 10.0,
        "peak_cuda_memory_allocated_gb": 0.1,
        "peak_cuda_memory_reserved_gb": 0.2,
        "peak_cuda_memory_allocated_fraction": 0.001,
        "peak_cuda_memory_reserved_fraction": 0.002,
        "device_name": "NVIDIA A100-SXM4-80GB",
        "cuda_total_memory_gb": 79.25,
        "precision_policy": "bf16",
        "autocast_enabled": True,
        "precision_applied": True,
        "gpu_utilization_class": "nominal",
    }
    payload.update(ACTIVE[model_key])
    payload["num_workers"] = 4
    return payload


def _result(model_key: ModelKey, performance: dict[str, object]) -> ModelExecutionResult:
    request = ModelRunRequest(
        model_key=model_key,
        run_name="run",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_mode=ModelRunMode.FULL,
    )
    stage = default_stage_plan_for_request(request).stages[0]
    return ModelExecutionResult(
        model_key=model_key,
        run_name="run",
        stages=(
            ModelStageResult(
                stage=stage,
                status=ModelStageStatus.COMPLETED,
                metadata={"performance": performance},
            ),
        ),
    )


@pytest.fixture(autouse=True)
def _device_and_calibration(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "text_to_sign_production.workflows.model.processing.reports.collect_model_device_telemetry",
        lambda: {
            "device_name": "NVIDIA A100-SXM4-80GB",
            "cuda_total_memory_gb": 79.25,
            "bf16_supported": True,
        },
    )
    monkeypatch.setattr(
        "text_to_sign_production.workflows.model.processing.reports.inspect_provider_real_calibration_artifacts",
        lambda **_kwargs: ProviderRealCalibrationArtifactStatus(
            provider_real_calibration_required=True,
            provider_real_calibration_found=True,
            selected_overrides_found=True,
            calibration_artifact_path=tmp_path / "calibration.json",
            selected_overrides_path=tmp_path / "selected.json",
            calibration_authoritative=True,
            calibration_benchmark_type="provider_real",
        ),
    )


@pytest.mark.parametrize("model_key", tuple(REQUIRED_FIELDS))
def test_a100_readiness_uses_provider_specific_required_fields(
    tmp_path: Path,
    model_key: ModelKey,
) -> None:
    payload = _a100_readiness_summary_payload(
        results=_result(model_key, _performance(model_key)),
        effective_config={"compute_profile_application": _application(model_key)},
        compute_profile=load_model_compute_profile(ROOT, "colab_a100_80gb").to_dict(),
        run_mode="full",
        workflow_completed=True,
        calibration_root=tmp_path,
    )

    assert payload["provider_batch_fields_complete"] is True
    assert "provider_batch_fields_missing" not in payload["blocking_issues"]


@pytest.mark.parametrize("model_key", tuple(REQUIRED_FIELDS))
def test_a100_readiness_reports_missing_provider_specific_fields(
    tmp_path: Path,
    model_key: ModelKey,
) -> None:
    performance = _performance(model_key)
    removed = MISSING_FIELD[model_key]
    performance.pop(removed)

    payload = _a100_readiness_summary_payload(
        results=_result(model_key, performance),
        effective_config={"compute_profile_application": _application(model_key)},
        compute_profile=load_model_compute_profile(ROOT, "colab_a100_80gb").to_dict(),
        run_mode="full",
        workflow_completed=True,
        calibration_root=tmp_path,
    )

    assert payload["provider_batch_fields_complete"] is False
    assert removed in payload["provider_batch_fields_missing"]
    assert "provider_batch_fields_missing" in payload["blocking_issues"]


def test_a100_readiness_blocks_unapplied_active_ledger(tmp_path: Path) -> None:
    app = _application(ModelKey.BASE_DIRECT)
    app["active_applied"] = {}

    payload = _a100_readiness_summary_payload(
        results=_result(ModelKey.BASE_DIRECT, _performance(ModelKey.BASE_DIRECT)),
        effective_config={"compute_profile_application": app},
        compute_profile=load_model_compute_profile(ROOT, "colab_a100_80gb").to_dict(),
        run_mode="full",
        workflow_completed=True,
        calibration_root=tmp_path,
    )

    assert payload["recommended_for_full_run"] is False
    assert "A100 active provider overrides are not applied." in payload["blocking_issues"]
