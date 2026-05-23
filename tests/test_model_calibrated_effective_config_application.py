from __future__ import annotations

from pathlib import Path

import pytest

from text_to_sign_production.modeling.candidates import (
    ModelExecutionResult,
    ModelRunMode,
    ModelRunRequest,
    ModelStageResult,
    ModelStageStatus,
    ProviderCalibrationPolicy,
    default_stage_plan_for_request,
)
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.model.contracts.compute_profiles import (
    load_model_compute_profile,
)
from text_to_sign_production.workflows.model.processing.calibration import (
    ProviderRealCalibrationArtifactStatus,
    apply_selected_overrides_to_effective_config,
)
from text_to_sign_production.workflows.model.processing.reports import (
    _a100_readiness_summary_payload,
)

ROOT = Path(__file__).resolve().parents[1]


def test_calibrated_effective_config_contains_compute_calibration_application() -> None:
    calibrated, applied = apply_selected_overrides_to_effective_config(
        effective_config={
            "training": {"batch_size": 8},
            "compute_profile_application": {
                "schema_version": "model.compute_profile_application.v1",
                "active_applied": {"batch_size": 8},
            },
        },
        selected_overrides={"batch_size": 16},
        policy=ProviderCalibrationPolicy(
            provider_key="base_direct",
            supports_provider_real=True,
            required_run_modes=("full",),
            required_compute_profiles=("colab_a100_80gb",),
            candidate_keys=("batch_size",),
            override_targets={"batch_size": "training.batch_size"},
            representative_surface_kinds={"batch_size": "base_direct_lazy_batch"},
            representative_split="train",
            max_samples=1,
            warmup_batches=0,
            max_batches_per_candidate=1,
        ),
    )

    application = calibrated["compute_calibration_application"]
    assert application["source"] == "provider_real_calibration"
    assert application["selected_overrides"] == {"batch_size": 16}
    assert application["applied_override_paths"] == {"training.batch_size": 16}
    assert application["base_compute_profile_application"]["active_applied"] == {"batch_size": 8}
    assert applied == {"training.batch_size": 16}


@pytest.fixture(autouse=True)
def _device_and_calibration(monkeypatch, tmp_path: Path):
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


def test_readiness_detects_calibrated_value_mismatch(tmp_path: Path) -> None:
    request = ModelRunRequest(
        model_key=ModelKey.BASE_DIRECT,
        run_name="mismatch",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_mode=ModelRunMode.FULL,
    )
    stage = default_stage_plan_for_request(request).stages[0]
    performance = {
        "batch_size": 8,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "stage_elapsed_seconds": 1.0,
        "unit_type": "sample",
        "unit_count": 1,
        "units_per_second": 1.0,
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
    results = ModelExecutionResult(
        model_key=ModelKey.BASE_DIRECT,
        run_name="mismatch",
        stages=(
            ModelStageResult(
                stage=stage,
                status=ModelStageStatus.COMPLETED,
                metadata={"performance": performance},
            ),
        ),
    )
    effective = {
        "compute_profile_application": {
            "active_requested": {"batch_size": 8},
            "active_applied": {"batch_size": 8},
            "active_not_applicable": {},
            "active_unsupported": {},
            "dataloader_applied": {"num_workers": 4},
            "telemetry_required_fields": ["batch_size", "num_workers"],
        },
        "compute_calibration_application": {
            "source": "provider_real_calibration",
            "selected_overrides": {"batch_size": 16},
        },
        "runtime_truth_contract": {
            "schema_version": "model.runtime_truth_contract.v1",
            "provider_key": "base_direct",
            "required_runtime_evidence": {"batch_size": ["batch_size"]},
        },
    }

    payload = _a100_readiness_summary_payload(
        results=results,
        effective_config=effective,
        compute_profile=load_model_compute_profile(ROOT, "colab_a100_80gb").to_dict(),
        run_mode="full",
        workflow_completed=True,
        calibration_root=tmp_path,
    )

    assert "provider_batch_fields_mismatched" in payload["blocking_issues"]
    assert payload["provider_batch_fields_mismatched"][0]["expected"] == 16
