from __future__ import annotations

import hashlib

import pytest

from text_to_sign_production.modeling.candidates import (
    ProviderFullDataPipelineCapability,
    VerificationEvidenceRef,
    VerifiedProviderFullDataPipelineCapability,
)
from text_to_sign_production.modeling.candidates.errors import ModelStagePlanError


def test_verified_capability_requires_evidence() -> None:
    with pytest.raises(ModelStagePlanError, match="verification evidence"):
        ProviderFullDataPipelineCapability(
            provider_key="learned_pose_token",
            full_training_data_mode="streaming_sharded",
            verified=True,
            verification_evidence=(),
            covered_stages=("fit_representation",),
        )


def test_unverified_streaming_declaration_is_not_full_safe() -> None:
    capability = ProviderFullDataPipelineCapability(
        provider_key="learned_pose_token",
        full_training_data_mode="streaming_sharded",
        verified=False,
        verification_evidence=(),
        covered_stages=("fit_representation",),
        limitations=("behavior tests have not executed",),
    )

    assert capability.is_full_safe is False


def test_verified_unsafe_mode_is_not_full_safe() -> None:
    capability = ProviderFullDataPipelineCapability(
        provider_key="base_direct",
        full_training_data_mode="unsafe_eager",
        verified=True,
        verification_evidence=("tests/behavior_contract.py::test_blocks_eager",),
        covered_stages=("train",),
    )

    assert capability.is_full_safe is False


def test_verified_capability_cannot_have_limitations() -> None:
    with pytest.raises(ModelStagePlanError, match="cannot declare limitations"):
        ProviderFullDataPipelineCapability(
            provider_key="base_direct",
            full_training_data_mode="lazy_dataloader",
            verified=True,
            verification_evidence=("behavior:lazy_dataloader:no_full_payload_load_before_first_batch",),
            covered_stages=("train",),
            limitations=("payload audit incomplete",),
        )


def test_runtime_verified_capability_requires_artifacts() -> None:
    with pytest.raises(ModelStagePlanError, match="artifact paths"):
        VerifiedProviderFullDataPipelineCapability(
            provider_key="learned_pose_token",
            model_key="learned_pose_token",
            run_name="run",
            mode="streaming_sharded",
            verified=True,
            verification_artifact_paths=(),
            calibration_artifact_path=None,
            selected_overrides_path=None,
            behavior_evidence=("behavior:fit_representation:surface_reader",),
            limitations=(),
        )


def test_runtime_capability_with_selected_provider_real_artifacts_is_full_safe(tmp_path) -> None:
    calibration_path = tmp_path / "compute_calibration.json"
    selected_path = tmp_path / "selected_overrides.json"
    _write_valid_runtime_artifacts(calibration_path, selected_path)

    capability = VerifiedProviderFullDataPipelineCapability(
        provider_key="learned_pose_token",
        model_key="learned_pose_token",
        run_name="run",
        mode="streaming_sharded",
        verified=True,
        verification_artifact_paths=(calibration_path, selected_path),
        calibration_artifact_path=calibration_path,
            selected_overrides_path=selected_path,
            behavior_evidence=("behavior:fit_representation:surface_reader",),
            limitations=(),
            verification_evidence_refs=(
                *_full_evidence_refs(calibration_path, selected_path),
                _evidence_ref("selected_overrides", selected_path),
            ),
        )

    assert capability.is_full_safe is True


def test_runtime_capability_rejects_string_only_evidence(tmp_path) -> None:
    calibration_path = tmp_path / "compute_calibration.json"
    selected_path = tmp_path / "selected_overrides.json"
    _write_valid_runtime_artifacts(calibration_path, selected_path)

    with pytest.raises(ModelStagePlanError, match="artifact refs"):
        VerifiedProviderFullDataPipelineCapability(
            provider_key="learned_pose_token",
            model_key="learned_pose_token",
            run_name="run",
            mode="streaming_sharded",
            verified=True,
            verification_artifact_paths=(calibration_path, selected_path),
            calibration_artifact_path=calibration_path,
            selected_overrides_path=selected_path,
            behavior_evidence=("behavior:fit_representation:surface_reader",),
            limitations=(),
        )


def test_runtime_capability_rejects_missing_selected_overrides(tmp_path) -> None:
    calibration_path = tmp_path / "compute_calibration.json"
    selected_path = tmp_path / "selected_overrides.json"
    _write_valid_runtime_artifacts(calibration_path, selected_path)

    with pytest.raises(ModelStagePlanError, match="selected_overrides"):
        VerifiedProviderFullDataPipelineCapability(
            provider_key="learned_pose_token",
            model_key="learned_pose_token",
            run_name="run",
            mode="streaming_sharded",
            verified=True,
            verification_artifact_paths=(calibration_path,),
            calibration_artifact_path=calibration_path,
            selected_overrides_path=None,
            behavior_evidence=("behavior:fit_representation:surface_reader",),
            limitations=(),
            verification_evidence_refs=(
                _evidence_ref("provider_real_calibration", calibration_path),
                _evidence_ref("representative_surface", calibration_path),
                _evidence_ref("progress_totals", calibration_path),
                _evidence_ref("bounded_writers", calibration_path),
                _evidence_ref("calibrated_effective_config", calibration_path),
            ),
        )


def test_runtime_capability_rejects_synthetic_calibration_artifact(tmp_path) -> None:
    calibration_path = tmp_path / "compute_calibration.json"
    selected_path = tmp_path / "selected_overrides.json"
    calibration_path.write_text(
        '{"provider_key":"learned_pose_token","model_key":"learned_pose_token","run_name":"run",'
        '"measurements":[{"benchmark_type":"synthetic_device_smoke"}]}\n',
        encoding="utf-8",
    )
    selected_path.write_text(
        '{"source":"provider_real_calibration","provider_key":"learned_pose_token",'
        '"model_key":"learned_pose_token","run_name":"run","selected_candidate_trace_ids":["trace"],'
        '"all_selected_rows_authoritative":true}\n',
        encoding="utf-8",
    )

    with pytest.raises(ModelStagePlanError, match="validated provider-real traces"):
        VerifiedProviderFullDataPipelineCapability(
            provider_key="learned_pose_token",
            model_key="learned_pose_token",
            run_name="run",
            mode="streaming_sharded",
            verified=True,
            verification_artifact_paths=(calibration_path, selected_path),
            calibration_artifact_path=calibration_path,
            selected_overrides_path=selected_path,
            behavior_evidence=("behavior:fit_representation:surface_reader",),
            limitations=(),
            verification_evidence_refs=(
                *_full_evidence_refs(calibration_path, selected_path),
                _evidence_ref("selected_overrides", selected_path),
            ),
        )


def _evidence_ref(kind: str, path) -> VerificationEvidenceRef:
    return VerificationEvidenceRef(
        kind=kind,
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        summary={"artifact": kind},
    )


def _full_evidence_refs(calibration_path, selected_path):
    return (
        _evidence_ref("provider_real_calibration", calibration_path),
        _evidence_ref("representative_surface", calibration_path),
        _evidence_ref("progress_totals", calibration_path),
        _evidence_ref("bounded_writers", calibration_path),
        _evidence_ref("calibrated_effective_config", calibration_path),
    )


def _write_valid_runtime_artifacts(calibration_path, selected_path) -> None:
    trace_id = "trace-1"
    effective_hash = "a" * 64
    build_hash = "c" * 64
    calibration_path.write_text(
        (
            '{"provider_key":"learned_pose_token","model_key":"learned_pose_token","run_name":"run",'
            '"effective_config_hash":"%s","model_build_spec_hash":"%s",'
            '"measurements":[{"benchmark_type":"provider_real",'
            '"trace_validation_passed":true,"trace_id":"%s","provider_real_trace":{},'
            '"effective_config_hash":"%s","model_build_spec_hash":"%s",'
            '"surface_provider_config_sha256":"%s"}]}\n'
        )
        % (effective_hash, build_hash, trace_id, effective_hash, build_hash, effective_hash),
        encoding="utf-8",
    )
    selected_path.write_text(
        (
            '{"source":"provider_real_calibration","provider_key":"learned_pose_token",'
            '"model_key":"learned_pose_token","run_name":"run","selected_candidate_trace_ids":["%s"],'
            '"effective_config_hash":"%s","model_build_spec_hash":"%s",'
            '"all_selected_rows_authoritative":true}\n'
        )
        % (trace_id, effective_hash, build_hash),
        encoding="utf-8",
    )
