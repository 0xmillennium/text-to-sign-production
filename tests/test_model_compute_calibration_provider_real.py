from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from text_to_sign_production.modeling.candidates import (
    ModelRunRequest,
)
from text_to_sign_production.modeling.candidates.bootstrap import ensure_model_provider_registered
from text_to_sign_production.modeling.candidates.registry import DEFAULT_MODEL_PROVIDER_REGISTRY
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.model.processing.provider import load_model_provider_config
from text_to_sign_production.workflows.model.processing.calibration import (
    benchmark_compute_candidates,
    benchmark_provider_compute_candidates,
    build_provider_real_calibration_context,
    inspect_provider_real_calibration_artifacts,
    provider_config_hash,
    provider_model_build_spec_hash,
    run_provider_real_calibration,
    run_compute_calibration,
    select_compute_calibration_overrides,
)
from text_to_sign_production.workflows.model.contracts.config import ModelWorkflowInvariantError
from helpers.real_modeling_topology import write_real_modeling_topology


def test_provider_calibration_invokes_provider_specific_benchmark_path() -> None:
    context = _context_for("latent_diffusion")
    rows = benchmark_provider_compute_candidates(
        context,
        {"denoiser_batch_size": [2]},
        max_batches_per_candidate=1,
    )

    assert rows[0]["provider_key"] == "latent_diffusion"
    assert rows[0]["benchmark_path"] == "latent_diffusion.denoiser.forward_backward"
    assert rows[0]["synthetic_device_smoke_fallback"] is False
    assert rows[0]["benchmark_type"] == "provider_real"


def test_synthetic_calibration_is_marked_non_authoritative() -> None:
    rows = benchmark_compute_candidates(
        {"tokenizer_batch_size": [2]},
        max_batches_per_candidate=1,
    )

    assert rows[0]["benchmark_type"] == "synthetic_device_smoke"
    assert rows[0]["synthetic_device_smoke_fallback"] is True


def test_synthetic_calibration_does_not_write_selected_overrides(tmp_path: Path) -> None:
    loaded, plan = _loaded_and_plan()
    run_compute_calibration(
        provider_config=loaded,
        stage_plan=plan,
        output_root=tmp_path,
        measurements=(
            {
                "parameter": "denoiser_batch_size",
                "value": 2,
                "units_per_second": 10.0,
                "benchmark_type": "synthetic_device_smoke",
            },
        ),
    )

    assert not (tmp_path / "selected_overrides.json").exists()


def test_provider_named_synthetic_rows_are_rejected(tmp_path: Path) -> None:
    loaded, plan = _loaded_and_plan()

    with pytest.raises(ModelWorkflowInvariantError, match="provider-named synthetic"):
        run_compute_calibration(
            provider_config=loaded,
            stage_plan=plan,
            output_root=tmp_path,
            measurements=(
                {
                    "provider_key": "latent_diffusion",
                    "parameter": "denoiser_batch_size",
                    "value": 2,
                    "units_per_second": 10.0,
                    "benchmark_type": "synthetic_device_smoke",
                },
            ),
        )


def test_provider_real_calibration_writes_selected_overrides(tmp_path: Path) -> None:
    loaded, plan = _loaded_and_plan()
    topology = write_real_modeling_topology(tmp_path / "runtime")
    context = build_provider_real_calibration_context(
        loaded_config=loaded,
        stage_plan=plan,
        working_dir=tmp_path,
        topology=topology,
    )
    rows = run_provider_real_calibration(
        context=context,
        candidates={"denoiser_batch_size": [2]},
        max_batches_per_candidate=1,
        warmup_batches=0,
    )
    run_compute_calibration(
        provider_config=loaded,
        stage_plan=plan,
        output_root=tmp_path,
        measurements=rows,
    )

    assert (tmp_path / "selected_overrides.json").is_file()


def test_readiness_artifact_status_requires_provider_real_and_selected_overrides(tmp_path: Path) -> None:
    loaded, plan = _loaded_and_plan()
    run_compute_calibration(
        provider_config=loaded,
        stage_plan=plan,
        output_root=tmp_path,
        measurements=(
            {
                "parameter": "denoiser_batch_size",
                "value": 2,
                "units_per_second": 10.0,
                "benchmark_type": "synthetic_device_smoke",
            },
        ),
    )
    synthetic = inspect_provider_real_calibration_artifacts(
        calibration_root=tmp_path,
        provider_key="latent_diffusion",
        expected_provider_config_hash=provider_config_hash(loaded.effective_config),
    )
    assert synthetic.provider_real_calibration_found is False
    assert "provider_real_calibration_missing" in synthetic.blocking_issues
    assert "selected_overrides_missing" in synthetic.blocking_issues

    rows = run_provider_real_calibration(
        context=build_provider_real_calibration_context(
            loaded_config=loaded,
            stage_plan=plan,
            working_dir=tmp_path,
            topology=write_real_modeling_topology(tmp_path / "runtime-real-1"),
        ),
        candidates={"denoiser_batch_size": [2]},
        max_batches_per_candidate=1,
        warmup_batches=0,
    )
    run_compute_calibration(
        provider_config=loaded,
        stage_plan=plan,
        output_root=tmp_path,
        measurements=rows,
    )
    provider_real = inspect_provider_real_calibration_artifacts(
        calibration_root=tmp_path,
        provider_key="latent_diffusion",
        expected_provider_config_hash=provider_config_hash(loaded.effective_config),
        expected_model_build_spec_hash=provider_model_build_spec_hash(
            "latent_diffusion",
            effective_config=loaded.effective_config,
        ),
    )
    assert provider_real.provider_real_calibration_found is True
    assert provider_real.selected_overrides_found is True
    assert provider_real.blocking_issues == ()


def test_readiness_rejects_selected_overrides_without_provider_real_trace(tmp_path: Path) -> None:
    loaded, plan = _loaded_and_plan()
    run_compute_calibration(
        provider_config=loaded,
        stage_plan=plan,
        output_root=tmp_path,
        measurements=(
            {
                "parameter": "denoiser_batch_size",
                "value": 2,
                "units_per_second": 10.0,
                "benchmark_type": "provider_real",
                "surface_reader_used": True,
                "actual_provider_model_used": True,
            },
        ),
    )
    (tmp_path / "selected_overrides.json").write_text(
        json.dumps({"source": "manual", "selected_candidate_trace_ids": ["fake"]}) + "\n",
        encoding="utf-8",
    )

    status = inspect_provider_real_calibration_artifacts(
        calibration_root=tmp_path,
        provider_key="latent_diffusion",
        expected_provider_config_hash=provider_config_hash(loaded.effective_config),
    )

    assert status.provider_real_calibration_found is False
    assert "provider_real_calibration_missing" in status.blocking_issues
    assert "selected_overrides_not_provider_real" in status.blocking_issues


def test_readiness_accepts_selected_overrides_with_valid_provider_real_trace(tmp_path: Path) -> None:
    loaded, plan = _loaded_and_plan()
    rows = run_provider_real_calibration(
        context=build_provider_real_calibration_context(
            loaded_config=loaded,
            stage_plan=plan,
            working_dir=tmp_path,
            topology=write_real_modeling_topology(tmp_path / "runtime-real-2"),
        ),
        candidates={"denoiser_batch_size": [2]},
        max_batches_per_candidate=1,
        warmup_batches=0,
    )
    run_compute_calibration(
        provider_config=loaded,
        stage_plan=plan,
        output_root=tmp_path,
        measurements=rows,
    )

    status = inspect_provider_real_calibration_artifacts(
        calibration_root=tmp_path,
        provider_key="latent_diffusion",
        expected_provider_config_hash=provider_config_hash(loaded.effective_config),
        expected_model_build_spec_hash=provider_model_build_spec_hash(
            "latent_diffusion",
            effective_config=loaded.effective_config,
        ),
    )

    assert status.blocking_issues == ()


def test_readiness_rejects_trace_id_mismatch(tmp_path: Path) -> None:
    loaded, plan = _loaded_and_plan()
    rows = run_provider_real_calibration(
        context=build_provider_real_calibration_context(
            loaded_config=loaded,
            stage_plan=plan,
            working_dir=tmp_path,
            topology=write_real_modeling_topology(tmp_path / "runtime-real-3"),
        ),
        candidates={"denoiser_batch_size": [2]},
        max_batches_per_candidate=1,
        warmup_batches=0,
    )
    run_compute_calibration(
        provider_config=loaded,
        stage_plan=plan,
        output_root=tmp_path,
        measurements=rows,
    )
    selected_path = tmp_path / "selected_overrides.json"
    payload = json.loads(selected_path.read_text(encoding="utf-8"))
    payload["selected_candidate_trace_ids"] = ["bad-trace-id"]
    selected_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    status = inspect_provider_real_calibration_artifacts(
        calibration_root=tmp_path,
        provider_key="latent_diffusion",
        expected_provider_config_hash=provider_config_hash(loaded.effective_config),
    )

    assert "selected_overrides_trace_id_mismatch" in status.blocking_issues


def test_provider_calibration_selection_skips_oom_and_near_tie_smaller_batch() -> None:
    selected, _flags = select_compute_calibration_overrides(
        [
            {
                "parameter": "frame_batch_size",
                "value": 32,
                "units_per_second": 100.0,
                "peak_cuda_memory_reserved_fraction": 0.2,
            },
            {
                "parameter": "frame_batch_size",
                "value": 64,
                "units_per_second": 104.0,
                "peak_cuda_memory_reserved_fraction": 0.2,
            },
            {
                "parameter": "frame_batch_size",
                "value": 128,
                "units_per_second": 300.0,
                "peak_cuda_memory_reserved_fraction": 0.4,
                "oom": True,
            },
        ]
    )

    assert selected["frame_batch_size"] == 32


def _loaded_and_plan():
    return _loaded_and_plan_for(ModelKey.LATENT_DIFFUSION, "latent_diffusion")


def _context_for(provider_key: str, root: Path | None = None):
    loaded, plan = _loaded_and_plan_for(ModelKey(provider_key), provider_key)
    base = root or Path(tempfile.mkdtemp(prefix="t2sp-test-real-topology-"))
    topology = write_real_modeling_topology(base / "runtime")
    return build_provider_real_calibration_context(
        loaded_config=loaded,
        stage_plan=plan,
        working_dir=base,
        topology=topology,
    )


def _loaded_and_plan_for(model_key: ModelKey, provider_key: str):
    candidates = {
        "learned_pose_token": {"tokenizer_batch_size": [2]},
        "latent_diffusion": {"denoiser_batch_size": [2]},
        "articulator_aware": {"frame_batch_size": [2]},
        "base_direct": {"batch_size": [2]},
    }[provider_key]
    request = ModelRunRequest(
        model_key=model_key,
        run_name="run",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        config_path=Path(f"configs/modeling/{provider_key}.yaml"),
        compute_profile={
            "name": "test",
            "provider_overrides": {
                provider_key: {"candidates": candidates}
            },
        },
    )
    ensure_model_provider_registered(model_key)
    provider = DEFAULT_MODEL_PROVIDER_REGISTRY.require(model_key)
    result = load_model_provider_config(provider, request)
    return result.loaded_config, provider.plan_stages(request, result.loaded_config)
