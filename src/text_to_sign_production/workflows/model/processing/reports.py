"""Delegate model report writing to providers and index materialized files."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from text_to_sign_production.modeling.artifacts import (
    GENERATED_POSE_SAMPLE_ARCHIVE_MANIFEST_NAME,
    GENERATED_POSE_SAMPLE_ARCHIVE_NAME,
    GENERATED_POSE_SAMPLE_ARCHIVE_SHA256_NAME,
)
from text_to_sign_production.modeling.candidates import (
    ModelExecutionResult,
    ModelProvider,
    ModelStageArtifactRef,
    ModelStageExecutionContext,
    read_runtime_support_manifest,
    resolve_model_run_mode_policy,
)
from text_to_sign_production.modeling.validation import (
    ValidationChannelMetricKey,
    ValidationMetricKey,
)
from text_to_sign_production.workflows.foundation.provenance import written_file_receipt
from text_to_sign_production.workflows.foundation.review import write_json
from text_to_sign_production.workflows.model.contracts import (
    ModelReportArtifacts,
    ModelObjectiveArtifactResult,
    ModelValidationArtifactResult,
    ModelWorkflowInvariantError,
)
from text_to_sign_production.workflows.model.layout import ModelLayout
from text_to_sign_production.workflows.model.processing.performance import (
    collect_model_device_telemetry,
)
from text_to_sign_production.workflows.model.processing.calibration import (
    inspect_provider_real_calibration_artifacts,
)


def write_model_reports(
    *,
    provider: ModelProvider,
    context: ModelStageExecutionContext,
    results: ModelExecutionResult,
    layout: ModelLayout,
    validation_artifacts: ModelValidationArtifactResult,
    objective_artifacts: tuple[ModelObjectiveArtifactResult, ...] = (),
    execution_id: str,
) -> ModelReportArtifacts:
    """Delegate report materialization, then write a generic report index."""

    if not all(receipt.path.is_file() for receipt in validation_artifacts.receipts):
        raise ModelWorkflowInvariantError(
            "model reports require materialized validation artifacts."
        )
    if context.request.auxiliary_objectives and not objective_artifacts:
        raise ModelWorkflowInvariantError(
            "requested auxiliary objectives require materialized objective artifacts "
            "before model reports are written."
        )
    try:
        provider_artifacts = provider.write_model_reports(context, results)
    except Exception as exc:
        raise ModelWorkflowInvariantError("model provider report writing failed") from exc
    if not isinstance(provider_artifacts, tuple) or any(
        not isinstance(artifact, ModelStageArtifactRef) for artifact in provider_artifacts
    ):
        raise ModelWorkflowInvariantError(
            "model provider report writing must return ModelStageArtifactRef values"
        )
    runtime_root = layout.stores.runtime.repo_root.resolve(strict=False)
    report_root = layout.reports.root.resolve(strict=False)
    for artifact in provider_artifacts:
        if not artifact.path.is_file():
            continue
        materialized_path = artifact.path.resolve(strict=False)
        if not materialized_path.is_relative_to(runtime_root):
            raise ModelWorkflowInvariantError(
                "provider materialized file is outside runtime repository root: "
                f"{materialized_path}"
            )
        if artifact.kind == "model_report" and not materialized_path.is_relative_to(report_root):
            raise ModelWorkflowInvariantError(
                f"provider model report is outside model report root: {materialized_path}"
            )
    receipts = tuple(
        written_file_receipt(
            artifact.role,
            artifact.path,
            execution_id=execution_id,
            kind="model_report",
        )
        for artifact in provider_artifacts
        if artifact.path.is_file()
    )
    performance_payload = _performance_summary_payload(results)
    readiness_payload = _a100_readiness_summary_payload(
        results=results,
        effective_config=context.loaded_config.effective_config,
        compute_profile=context.request.compute_profile,
        run_mode=context.request.run_mode.value,
        workflow_completed=results.completed,
        calibration_root=layout.reports.root,
    )
    calibration_payload = _calibration_report_payload(
        layout=layout,
        readiness_payload=readiness_payload,
    )
    index_payload = {
        "schema_version": "model_report_index.v1",
        "execution_id": execution_id,
        "model_key": results.model_key.value,
        "run_name": results.run_name,
        **context.request.manifest_family.to_dict(),
        "train_split": context.request.train_split.value,
        "validation_split": context.request.validation_split.value,
        "test_split": "test",
        "prediction_splits": [split.value for split in context.request.prediction_splits],
        "auxiliary_objectives": [
            objective.value for objective in context.request.auxiliary_objectives
        ],
        "run_mode": context.request.run_mode.value,
        "run_mode_quality_claim": resolve_model_run_mode_policy(
            context.request.run_mode
        ).quality_claim,
        "model_metadata": str(layout.outputs.run_metadata_path),
        "runtime_support": _runtime_support_summary(
            layout.outputs.runtime_support_manifest_path
        ),
        "validation_outputs": {
            split_output.split.value: _generated_pose_output_summary(split_output)
            for split_output in layout.outputs.generated_pose_split_outputs
        },
        "validation_artifacts": {
            "pairing_manifest": str(validation_artifacts.pairing_manifest_path),
            "metric_results": str(validation_artifacts.metric_results_path),
            "channel_metric_results": str(validation_artifacts.channel_metric_results_path),
            "aggregate_metrics": str(validation_artifacts.aggregate_metrics_path),
            "channel_aggregate_metrics": str(validation_artifacts.channel_aggregate_metrics_path),
            "limitations": str(validation_artifacts.limitations_path),
            "summary_markdown": str(validation_artifacts.summary_markdown_path),
            "paired_count": validation_artifacts.paired_count,
            "missing_generated_count": validation_artifacts.missing_generated_count,
            "missing_reference_count": validation_artifacts.missing_reference_count,
            "failed_generated_count": validation_artifacts.failed_generated_count,
            "identity_mismatch_count": validation_artifacts.identity_mismatch_count,
        },
        "validation_metric_policy": {
            "split": "val",
            "alignment_policy": "simple_prefix_minimum_sequence_length",
            "metric_keys": [metric.value for metric in ValidationMetricKey],
            "channel_metric_keys": [metric.value for metric in ValidationChannelMetricKey],
        },
        "objective_artifacts": {
            artifact.objective_key.value: {
                "paths": {
                    key: str(path)
                    for key, path in artifact.artifact_paths.items()
                },
                "records_count": artifact.records_count,
                "skipped_count": artifact.skipped_count,
                "proxy_only": True,
                "requires_ablation": True,
                "ablation_status": "incomplete",
                "config_snapshot_path": str(artifact.config_snapshot_path),
                "config_snapshot_sha256": artifact.config_snapshot_sha256,
                "generated_manifest_path": str(artifact.generated_pose_manifest_path),
                "generated_manifest_artifact_subtype": "final_validation",
                "candidate_policy": artifact.candidate_policy,
                "ablation_readiness_path": str(artifact.ablation_readiness_path),
                "ready_for_comparison": artifact.ready_for_comparison,
                "required_baseline_missing": artifact.required_baseline_missing,
                "readiness_issues": list(artifact.readiness_issues),
            }
            for artifact in objective_artifacts
        },
        "checkpoint_paths": _checkpoint_paths(results),
        "standardization_diagnostics": _standardization_diagnostics(results),
        "performance_summary": performance_payload,
        "a100_readiness_summary": readiness_payload,
        "calibration": calibration_payload,
        "calibration_required": calibration_payload["required"],
        "calibration_executed": calibration_payload["executed"],
        "calibration_reused": calibration_payload["reused"],
        "selected_overrides": calibration_payload["selected_overrides"],
        "applied_overrides": calibration_payload["applied_overrides"],
        "calibrated_effective_config_hash": calibration_payload[
            "calibrated_effective_config_hash"
        ],
        "representative_surface_count": calibration_payload["representative_surface_count"],
        "limitations": [
            "Reports summarize workflow/provider artifacts and do not prove final model quality.",
            "Generated-pose artifacts are validation split outputs for development/checkpoint selection.",
            "Validation results are not full test split evaluation or aggregate test performance.",
            "Automatic pose/keypoint metrics are not proof of sign intelligibility.",
            "Run-mode quality claims constrain interpretation of these outputs.",
            "Standardization fallback diagnostics indicate sparse observations; smoke fallback is not a model quality claim.",
            *(
                []
                if not objective_artifacts
                else [
                    "Semantic consistency is post-generation proxy evaluation only; it is not differentiable training optimization.",
                    "An executed baseline is required before contribution claims; embedding similarity is not semantic correctness.",
                    "No completed semantic ablation exists in this stage.",
                ]
            ),
        ],
        "report_files": {
            "model_run_summary": str(layout.reports.model_run_summary_report_path),
            "stage_artifacts_detail": str(layout.reports.stage_artifacts_detail_json_path),
            "performance_summary_json": str(layout.reports.performance_summary_json_path),
            "performance_summary_markdown": str(layout.reports.performance_summary_report_path),
            "a100_readiness_summary_json": str(layout.reports.a100_readiness_summary_json_path),
            "a100_readiness_summary_markdown": str(layout.reports.a100_readiness_summary_report_path),
            "compute_calibration": str(layout.reports.root / "compute_calibration.json"),
            "compute_calibration_report": str(layout.reports.root / "compute_calibration.md"),
            "selected_overrides": str(layout.reports.root / "selected_overrides.json"),
            "calibrated_effective_config": str(
                layout.reports.root / "calibrated_effective_config.json"
            ),
            "training_artifact_summary": str(layout.reports.training_artifact_summary_json_path),
            "generated_pose_artifact_summary": str(
                layout.reports.generated_pose_artifact_summary_json_path
            ),
            "validation_summary": str(layout.reports.validation_summary_report_path),
            "checkpoint_selection": str(layout.reports.checkpoint_selection_report_path),
            "limitations": str(layout.reports.limitations_report_path),
        },
        "provider_artifacts": [
            {
                "role": artifact.role,
                "kind": artifact.kind,
                "path": str(artifact.path),
                "description": artifact.description,
                "materialized_file": artifact.path.is_file(),
            }
            for artifact in provider_artifacts
        ],
    }
    write_json(layout.reports.stage_artifacts_detail_json_path, index_payload)
    write_json(layout.reports.performance_summary_json_path, performance_payload)
    layout.reports.performance_summary_report_path.write_text(
        _performance_summary_markdown(performance_payload),
        encoding="utf-8",
    )
    write_json(layout.reports.a100_readiness_summary_json_path, readiness_payload)
    layout.reports.a100_readiness_summary_report_path.write_text(
        _a100_readiness_summary_markdown(readiness_payload),
        encoding="utf-8",
    )
    write_json(
        layout.reports.training_artifact_summary_json_path,
        _artifact_summary(provider_artifacts, kind_prefix="model_training"),
    )
    write_json(
        layout.reports.generated_pose_artifact_summary_json_path,
        _generated_pose_artifact_summary(
            provider_artifacts,
            layout=layout,
        ),
    )
    write_json(layout.reports.index_json_path, index_payload)
    detail_receipts = (
        written_file_receipt(
            "model stage artifacts detail",
            layout.reports.stage_artifacts_detail_json_path,
            execution_id=execution_id,
            kind="model_report",
        ),
        written_file_receipt(
            "model training artifact summary",
            layout.reports.training_artifact_summary_json_path,
            execution_id=execution_id,
            kind="model_report",
        ),
        written_file_receipt(
            "model generated pose artifact summary",
            layout.reports.generated_pose_artifact_summary_json_path,
            execution_id=execution_id,
            kind="model_report",
        ),
        written_file_receipt(
            "model performance summary",
            layout.reports.performance_summary_json_path,
            execution_id=execution_id,
            kind="model_report",
        ),
        written_file_receipt(
            "model performance summary markdown",
            layout.reports.performance_summary_report_path,
            execution_id=execution_id,
            kind="model_report",
        ),
        written_file_receipt(
            "model a100 readiness summary",
            layout.reports.a100_readiness_summary_json_path,
            execution_id=execution_id,
            kind="model_report",
        ),
        written_file_receipt(
            "model a100 readiness summary markdown",
            layout.reports.a100_readiness_summary_report_path,
            execution_id=execution_id,
            kind="model_report",
        ),
        *_calibration_report_receipts(layout=layout, execution_id=execution_id),
    )
    index_receipt = written_file_receipt(
        "model report index",
        layout.reports.index_json_path,
        execution_id=execution_id,
        kind="model_report",
    )
    return ModelReportArtifacts(
        execution_id=execution_id,
        artifacts=(*receipts, *detail_receipts, index_receipt),
        provider_artifacts=provider_artifacts,
    )


def _artifact_summary(
    provider_artifacts: tuple[ModelStageArtifactRef, ...],
    *,
    kind_prefix: str,
) -> dict[str, object]:
    matching = tuple(
        artifact for artifact in provider_artifacts if artifact.kind.startswith(kind_prefix)
    )
    return {
        "schema_version": f"{kind_prefix}_artifact_summary.v1",
        "artifact_count": len(matching),
        "materialized_file_count": sum(1 for artifact in matching if artifact.path.is_file()),
        "artifacts": [
            {
                "role": artifact.role,
                "kind": artifact.kind,
                "path": str(artifact.path),
                "materialized_file": artifact.path.is_file(),
            }
            for artifact in matching
        ],
        "quality_claim": "not_evaluated",
    }


def _calibration_report_payload(
    *,
    layout: ModelLayout,
    readiness_payload: Mapping[str, object],
) -> dict[str, object]:
    selected_path = layout.reports.root / "selected_overrides.json"
    calibrated_path = layout.reports.root / "calibrated_effective_config.json"
    selected = _read_json_object(selected_path)
    calibrated = _read_json_object(calibrated_path)
    selected_overrides = selected.get("overrides") if selected else {}
    applied_overrides = calibrated.get("applied_overrides") if calibrated else {}
    representative_count = _representative_surface_count(layout.reports.root / "compute_calibration.json")
    return {
        "required": bool(readiness_payload.get("provider_real_calibration_required")),
        "executed": bool(readiness_payload.get("provider_real_calibration_found"))
        and not bool(readiness_payload.get("calibration_reused")),
        "reused": bool(readiness_payload.get("calibration_reused")),
        "compute_calibration_path": str(layout.reports.root / "compute_calibration.json"),
        "compute_calibration_report_path": str(layout.reports.root / "compute_calibration.md"),
        "selected_overrides_path": str(selected_path),
        "calibrated_effective_config_path": str(calibrated_path),
        "base_effective_config_hash": calibrated.get("base_effective_config_hash") if calibrated else None,
        "selected_overrides_hash": calibrated.get("selected_overrides_hash") if calibrated else None,
        "calibrated_effective_config_hash": calibrated.get("calibrated_effective_config_hash") if calibrated else None,
        "selected_overrides": selected_overrides if isinstance(selected_overrides, Mapping) else {},
        "applied_overrides": applied_overrides if isinstance(applied_overrides, Mapping) else {},
        "representative_surface_count": representative_count,
    }


def _representative_surface_count(path: Path) -> int:
    payload = _read_json_object(path)
    measurements = payload.get("measurements") if payload else None
    if not isinstance(measurements, list):
        return 0
    seen = {
        (
            row.get("representative_source_manifest_sha256"),
            row.get("representative_surface_kind") or row.get("surface_kind"),
        )
        for row in measurements
        if isinstance(row, Mapping)
    }
    return len({item for item in seen if all(isinstance(value, str) and value for value in item)})


def _calibration_report_receipts(
    *,
    layout: ModelLayout,
    execution_id: str,
) -> tuple:
    receipts = []
    for label, path, kind in (
        ("model compute calibration", layout.reports.root / "compute_calibration.json", "compute_calibration"),
        ("model compute calibration report", layout.reports.root / "compute_calibration.md", "compute_calibration_report"),
        ("model selected overrides", layout.reports.root / "selected_overrides.json", "selected_overrides"),
        ("model calibrated effective config", layout.reports.root / "calibrated_effective_config.json", "calibrated_effective_config"),
    ):
        if path.is_file():
            receipts.append(
                written_file_receipt(
                    label,
                    path,
                    execution_id=execution_id,
                    kind=kind,
                )
            )
    return tuple(receipts)


def _read_json_object(path) -> Mapping[str, object]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _generated_pose_output_summary(split_output) -> dict[str, object]:
    split_root = split_output.manifest_path.parent
    archive_path = split_root / GENERATED_POSE_SAMPLE_ARCHIVE_NAME
    archive_manifest_path = split_root / GENERATED_POSE_SAMPLE_ARCHIVE_MANIFEST_NAME
    archive_sha256_path = split_root / GENERATED_POSE_SAMPLE_ARCHIVE_SHA256_NAME
    archive_manifest = _read_archive_manifest(archive_manifest_path)
    return {
        "manifest": str(split_output.manifest_path),
        "samples_root": str(split_output.samples_root),
        "generated_sample_count": archive_manifest.get("sample_count"),
        "archive_path": str(archive_path) if archive_path.is_file() else None,
        "archive_manifest_path": (
            str(archive_manifest_path) if archive_manifest_path.is_file() else None
        ),
        "archive_sha256_path": str(archive_sha256_path) if archive_sha256_path.is_file() else None,
        "archive_member_count": archive_manifest.get("member_count"),
        "archive_sha256": archive_manifest.get("archive_sha256"),
        "generated_manifest_sha256": archive_manifest.get("generated_manifest_sha256"),
        "individual_sample_publish": False if archive_path.is_file() else None,
    }


def _generated_pose_artifact_summary(
    provider_artifacts: tuple[ModelStageArtifactRef, ...],
    *,
    layout: ModelLayout,
) -> dict[str, object]:
    base = _artifact_summary(provider_artifacts, kind_prefix="generated_pose")
    base["validation_outputs"] = {
        split_output.split.value: _generated_pose_output_summary(split_output)
        for split_output in layout.outputs.generated_pose_split_outputs
    }
    outputs = tuple(base["validation_outputs"].values())
    base["final_validation_surface_count"] = sum(
        1 for output in outputs if output.get("archive_path") is not None
    )
    base["generated_sample_count"] = sum(
        int(output.get("generated_sample_count") or 0) for output in outputs
    )
    base["archive_count"] = sum(1 for output in outputs if output.get("archive_path") is not None)
    base["archive_member_count"] = sum(
        int(output.get("archive_member_count") or 0) for output in outputs
    )
    base["archive_verified"] = all(
        output.get("archive_path") is not None
        and output.get("archive_manifest_path") is not None
        and output.get("archive_sha256_path") is not None
        for output in outputs
    )
    base["verification_basis"] = "archive_manifest_and_sha256_files"
    if base["archive_count"]:
        base["artifact_count"] = base["archive_count"]
        base["materialized_file_count"] = sum(
            4
            for output in outputs
            if output.get("manifest") is not None and output.get("archive_path") is not None
        )
        base["quality_claim"] = (
            "archive_verified" if base["archive_verified"] else "archive_materialized"
        )
    base["individual_sample_publish"] = False
    return base


def _runtime_support_summary(path) -> dict[str, object]:
    if not path.is_file():
        return {
            "runtime_support_manifest_path": str(path),
            "manifest_exists": False,
            "support_artifact_count": 0,
            "required_for_test_model_count": 0,
        }
    try:
        manifest = read_runtime_support_manifest(path)
    except Exception as exc:
        return {
            "runtime_support_manifest_path": str(path),
            "manifest_exists": True,
            "readable": False,
            "error": str(exc),
            "support_artifact_count": 0,
            "required_for_test_model_count": 0,
        }
    return {
        "runtime_support_manifest_path": str(path),
        "manifest_exists": True,
        "readable": True,
        "provider_key": manifest.model_key,
        "support_artifact_count": len(manifest.artifacts),
        "required_for_test_model_count": sum(
            1 for artifact in manifest.artifacts if artifact.required_for_test_model
        ),
    }


def _read_archive_manifest(path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _checkpoint_paths(results: ModelExecutionResult) -> dict[str, object]:
    best = None
    last = None
    metric_name = None
    metric_value = None
    for stage in results.stages:
        metadata = dict(stage.metadata)
        if "best_checkpoint_path" in metadata:
            best = metadata.get("best_checkpoint_path")
        if "last_checkpoint_path" in metadata:
            last = metadata.get("last_checkpoint_path")
        if "best_metric_name" in metadata:
            metric_name = metadata.get("best_metric_name")
        if "best_metric_value" in metadata:
            metric_value = metadata.get("best_metric_value")
    return {
        "best_checkpoint_path": best,
        "last_checkpoint_path": last,
        "checkpoint_selection_metric": metric_name,
        "checkpoint_selection_metric_value": metric_value,
    }


def _standardization_diagnostics(results: ModelExecutionResult) -> list[dict[str, object]]:
    keys = (
        "standardization_missing_observation_policy",
        "standardization_zero_observation_coordinate_count",
        "standardization_channel_fallback_coordinate_count",
        "standardization_global_fallback_coordinate_count",
        "standardization_identity_fallback_coordinate_count",
        "standardization_sample_count",
        "standardization_frame_count",
        "standardization_fallback_summary_by_channel",
    )
    diagnostics = []
    for stage in results.stages:
        metadata = dict(stage.metadata)
        if "standardization_missing_observation_policy" not in metadata:
            continue
        diagnostics.append(
            {
                "stage_kind": stage.stage.spec.kind.value,
                "provider_stage_id": stage.stage.provider_stage_id,
                **{key: metadata.get(key) for key in keys},
                "interpretation": (
                    "Sparse-observation fallback is a smoke/debug robustness diagnostic, "
                    "not a model quality claim."
                ),
            }
        )
    return diagnostics


def _performance_summary_payload(results: ModelExecutionResult) -> dict[str, object]:
    stages = [
        dict(stage.metadata["performance"])
        for stage in results.stages
        if isinstance(stage.metadata.get("performance"), Mapping)
    ]
    total_elapsed = sum(float(stage.get("elapsed_seconds") or 0.0) for stage in stages)
    slowest = max(stages, key=lambda item: float(item.get("elapsed_seconds") or 0.0), default=None)
    peak_reserved = max(
        (
            float(stage["peak_cuda_memory_reserved_gb"])
            for stage in stages
            if isinstance(stage.get("peak_cuda_memory_reserved_gb"), int | float)
        ),
        default=None,
    )
    return {
        "schema_version": "model_performance_summary.v1",
        "device": collect_model_device_telemetry(),
        "stage_count": len(stages),
        "stages": stages,
        "total_stage_elapsed_seconds": total_elapsed,
        "slowest_stage": slowest,
        "peak_cuda_memory_reserved_gb": peak_reserved,
        "peak_cuda_memory_reserved_fraction": _max_float(
            stage.get("peak_cuda_memory_reserved_fraction") for stage in stages
        ),
        "limitations": [
            "Performance telemetry is measurement infrastructure for debugging and tuning.",
            "Smoke/debug timing and reduced sampling settings are wiring/performance signals, not model quality claims.",
        ],
    }


def _performance_summary_markdown(payload: Mapping[str, object]) -> str:
    device = payload.get("device")
    device = device if isinstance(device, Mapping) else {}
    slowest = payload.get("slowest_stage")
    slowest = slowest if isinstance(slowest, Mapping) else {}
    stages = payload.get("stages")
    stages = stages if isinstance(stages, list) else []
    precision_lines = [
        (
            f"- {stage.get('provider_stage_id')}: requested={stage.get('requested_precision_policy')}, "
            f"resolved={stage.get('resolved_precision_policy')}, "
            f"autocast_enabled={stage.get('autocast_enabled')}, "
            f"autocast_dtype={stage.get('autocast_dtype')}, "
            f"precision_applied={stage.get('precision_applied')}"
        )
        for stage in stages
        if isinstance(stage, Mapping)
    ]
    return "\n".join(
        (
            "# Model Performance Summary",
            "",
            f"- device: {device.get('device_type')}",
            f"- device_name: {device.get('device_name')}",
            f"- cuda_total_memory_gb: {device.get('cuda_total_memory_gb')}",
            f"- bf16_supported: {device.get('bf16_supported')}",
            f"- tf32_matmul_allowed: {device.get('tf32_matmul_allowed')}",
            f"- peak_cuda_memory_reserved_gb: {payload.get('peak_cuda_memory_reserved_gb')}",
            f"- peak_cuda_memory_reserved_fraction: {payload.get('peak_cuda_memory_reserved_fraction')}",
            f"- slowest_stage: {slowest.get('provider_stage_id')}",
            f"- total_stage_elapsed_seconds: {payload.get('total_stage_elapsed_seconds')}",
            "",
            "## Utilization",
            "",
            *(
                f"- {stage.get('provider_stage_id')}: fraction={stage.get('peak_cuda_memory_reserved_fraction')}, "
                f"class={stage.get('gpu_utilization_class')}, bottleneck={stage.get('bottleneck_hint')}, "
                f"{_stage_telemetry_summary(stage)}"
                for stage in stages
                if isinstance(stage, Mapping)
            ),
            "",
            "## Precision",
            "",
            *(precision_lines or ("- no stage precision telemetry",)),
            "",
            "Smoke/debug telemetry is for wiring and performance readiness, not final model quality.",
            "",
        )
    )


def _stage_telemetry_display_fields(stage: Mapping[str, object]) -> list[str]:
    preferred = (
        "batch_size",
        "tokenizer_batch_size",
        "text_to_token_batch_size",
        "reconstruction_batch_size",
        "decode_batch_size",
        "denoiser_batch_size",
        "source_batch_size",
        "frame_batch_size",
        "num_workers",
        "surface_reader_num_workers_used",
        "surface_reader_worker_mode",
        "configured_sampling_steps",
        "effective_sampling_steps",
        "candidate_count",
        "latent_autoencoder_active",
    )
    return [key for key in preferred if key in stage]


def _stage_telemetry_summary(stage: Mapping[str, object]) -> str:
    display_keys = _stage_telemetry_display_fields(stage)
    return ", ".join(f"{key}={stage.get(key)}" for key in display_keys) or "telemetry=none"


def _a100_readiness_summary_payload(
    *,
    results: ModelExecutionResult,
    effective_config: Mapping[str, object],
    compute_profile: Mapping[str, object],
    run_mode: str,
    workflow_completed: bool,
    calibration_root=None,
) -> dict[str, object]:
    device = collect_model_device_telemetry()
    stages = [
        dict(stage.metadata["performance"])
        for stage in results.stages
        if isinstance(stage.metadata.get("performance"), Mapping)
    ]
    provider_key = results.model_key.value
    application = effective_config.get("compute_profile_application")
    if not isinstance(application, Mapping):
        raise ModelWorkflowInvariantError(
            "A100 readiness requires effective_config.compute_profile_application."
        )
    required_value = application.get("telemetry_required_fields")
    if isinstance(required_value, str):
        raise ModelWorkflowInvariantError(
            "compute_profile_application.telemetry_required_fields must be a sequence."
        )
    try:
        required_batch_fields = tuple(required_value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ModelWorkflowInvariantError(
            "compute_profile_application.telemetry_required_fields must be a sequence."
        ) from exc
    if any(not isinstance(field, str) or not field.strip() for field in required_batch_fields):
        raise ModelWorkflowInvariantError(
            "compute_profile_application.telemetry_required_fields must contain text."
        )
    active_requested = _required_mapping(application, "active_requested")
    active_applied = _required_mapping(application, "active_applied")
    active_not_applicable = _required_mapping(application, "active_not_applicable")
    active_unsupported = _required_mapping(application, "active_unsupported")
    missing_active_classification = sorted(
        set(active_requested)
        - (set(active_applied) | set(active_not_applicable) | set(active_unsupported))
    )
    extra_active_classification = sorted(
        (set(active_applied) | set(active_not_applicable) | set(active_unsupported))
        - set(active_requested)
    )
    expected_runtime_values = _expected_runtime_values_from_effective_config(
        effective_config
    )
    missing_provider_batch_fields = [
        field
        for field in required_batch_fields
        if not any(stage.get(field) is not None for stage in stages)
    ]
    mismatched_provider_batch_fields = [
        {
            "field": key,
            "expected": expected,
            "observed": [
                stage.get(key)
                for stage in stages
                if stage.get(key) is not None
            ],
        }
        for key, expected in expected_runtime_values.items()
        if any(stage.get(key) is not None for stage in stages)
        and not any(stage.get(key) == expected for stage in stages)
    ]
    contract = effective_config.get("runtime_truth_contract")
    required_runtime_evidence = (
        contract.get("required_runtime_evidence")
        if isinstance(contract, Mapping)
        else {}
    )
    runtime_evidence_missing: list[str] = []
    if isinstance(required_runtime_evidence, Mapping):
        dataloader_applied = application.get("dataloader_applied")
        applied_keys = set(active_applied) | (
            set(dataloader_applied) if isinstance(dataloader_applied, Mapping) else set()
        )
        calibration_application = effective_config.get("compute_calibration_application")
        if isinstance(calibration_application, Mapping):
            selected = calibration_application.get("selected_overrides")
            if isinstance(selected, Mapping):
                applied_keys.update(str(key) for key in selected)
        for applied_key in sorted(applied_keys):
            evidence_keys = required_runtime_evidence.get(applied_key)
            if not isinstance(evidence_keys, Sequence) or isinstance(
                evidence_keys,
                str | bytes | bytearray,
            ):
                continue
            for evidence_key in evidence_keys:
                if not isinstance(evidence_key, str):
                    continue
                if not any(stage.get(evidence_key) is not None for stage in stages):
                    runtime_evidence_missing.append(evidence_key)
    batch_fields_complete = (
        not missing_provider_batch_fields
        and not mismatched_provider_batch_fields
        and not runtime_evidence_missing
    )
    telemetry_required = (
        "stage_elapsed_seconds",
        "unit_type",
        "unit_count",
        "units_per_second",
        "peak_cuda_memory_allocated_gb",
        "peak_cuda_memory_reserved_gb",
        "peak_cuda_memory_allocated_fraction",
        "peak_cuda_memory_reserved_fraction",
        "device_name",
        "cuda_total_memory_gb",
        "precision_policy",
        "autocast_enabled",
        "precision_applied",
    )
    telemetry_complete = all(
        all(key in stage for key in telemetry_required)
        for stage in stages
    ) if stages else False
    utilization = _worst_utilization_class(stages)
    warnings: list[str] = []
    if utilization == "underutilized":
        warnings.append("gpu_utilization_class is underutilized.")
    if any(stage.get("bottleneck_hint") == "cpu_or_io_bound" for stage in stages):
        warnings.append("At least one stage reports cpu_or_io_bound.")
        warnings.append("cpu_or_io_bound")
    blocking: list[str] = []
    calibration_status = inspect_provider_real_calibration_artifacts(
        calibration_root=calibration_root,
        provider_key=provider_key,
        required=run_mode == "full",
    )
    if run_mode == "full":
        blocking.extend(calibration_status.blocking_issues)
    elif (
        calibration_status.provider_real_calibration_required
        and not calibration_status.provider_real_calibration_found
    ):
        warnings.append("calibration_required_but_missing")
    elif not calibration_status.provider_real_calibration_found:
        warnings.append("calibration_optional_not_attached")
    if compute_profile.get("name") != "colab_a100_80gb":
        blocking.append("A100 compute profile is not loaded.")
    if device.get("bf16_supported") is not True:
        blocking.append("bf16 is not reported as supported by the active device.")
    if not active_requested or dict(active_requested) != dict(active_applied):
        blocking.append("A100 active provider overrides are not applied.")
    if active_unsupported:
        blocking.append("active_overrides_unsupported")
    if active_not_applicable:
        blocking.append("active_overrides_not_applicable")
    if missing_active_classification or extra_active_classification:
        blocking.append("active_overrides_unclassified")
    if not batch_fields_complete:
        blocking.append("Provider batch telemetry fields are incomplete.")
    if missing_provider_batch_fields:
        blocking.append("provider_batch_fields_missing")
    if mismatched_provider_batch_fields:
        blocking.append("provider_batch_fields_mismatched")
    if runtime_evidence_missing:
        blocking.append("runtime_evidence_missing")
    if not telemetry_complete:
        blocking.append("Stage performance telemetry is incomplete.")
    if not workflow_completed:
        blocking.append("Workflow stage execution did not complete cleanly.")
    if any(stage.get("materialization_total_unknown") is True for stage in stages):
        blocking.append("materialization_total_unknown")
    if any(stage.get("eager_full_materialization_detected") is True for stage in stages):
        blocking.append("eager_full_materialization_detected")
    if any(stage.get("provider_full_pipeline_not_streaming") is True for stage in stages):
        blocking.append("provider_full_pipeline_not_streaming")
    if any(stage.get("provider_capability_unverified") is True for stage in stages):
        blocking.append("provider_capability_unverified")
    if any(stage.get("provider_limitations_present") is True for stage in stages):
        blocking.append("provider_limitations_present")
    if any(stage.get("provider_real_calibration_missing") is True for stage in stages):
        blocking.append("provider_real_calibration_missing")
    if any(stage.get("synthetic_calibration_used_as_authoritative") is True for stage in stages):
        blocking.append("synthetic_calibration_used_as_authoritative")
    if any(stage.get("incremental_writer_missing_for_generation_or_export") is True for stage in stages):
        blocking.append("incremental_writer_missing_for_generation_or_export")
    if any(stage.get("progress_task_missing_for_long_operation") is True for stage in stages):
        blocking.append("progress_task_missing_for_long_operation")
    if any(stage.get("sharded_surface_missing") is True for stage in stages):
        blocking.append("sharded_surface_missing")
    if any(
        _cache_read_dominates_stage(stage)
        for stage in stages
    ):
        blocking.append("cache_read_dominates_runtime")
    if any(stage.get("memory_growth_unbounded") is True for stage in stages):
        blocking.append("memory_growth_unbounded")
    full_run_recommendation = "no" if blocking else ("conditional" if warnings else "yes")
    return {
        "schema_version": "model.a100_readiness_summary.v1",
        "compute_profile": compute_profile.get("name"),
        "device_name": device.get("device_name"),
        "cuda_total_memory_gb": device.get("cuda_total_memory_gb"),
        "bf16_supported": device.get("bf16_supported"),
        "active_overrides_applied": bool(active_requested) and dict(active_requested) == dict(active_applied),
        "active_requested": dict(active_requested),
        "active_applied": dict(active_applied),
        "active_not_applicable": dict(active_not_applicable),
        "active_unsupported": dict(active_unsupported),
        "provider_batch_required_fields": list(required_batch_fields),
        "expected_runtime_values": expected_runtime_values,
        "provider_batch_fields_missing": missing_provider_batch_fields,
        "provider_batch_fields_mismatched": mismatched_provider_batch_fields,
        "runtime_evidence_missing": sorted(set(runtime_evidence_missing)),
        "provider_batch_fields_complete": batch_fields_complete,
        "telemetry_complete": telemetry_complete,
        "gpu_utilization_class": utilization,
        "calibration_available": calibration_status.provider_real_calibration_found,
        "selected_overrides_available": calibration_status.selected_overrides_found,
        "provider_real_calibration_required": calibration_status.provider_real_calibration_required,
        "provider_real_calibration_found": calibration_status.provider_real_calibration_found,
        "selected_overrides_found": calibration_status.selected_overrides_found,
        "calibration_artifact_path": (
            str(calibration_status.calibration_artifact_path)
            if calibration_status.calibration_artifact_path is not None
            else None
        ),
        "selected_overrides_path": (
            str(calibration_status.selected_overrides_path)
            if calibration_status.selected_overrides_path is not None
            else None
        ),
        "calibration_authoritative": calibration_status.calibration_authoritative,
        "calibration_benchmark_type": calibration_status.calibration_benchmark_type,
        "full_run_recommendation": full_run_recommendation,
        "recommended_for_full_run": not blocking,
        "blocking_issues": blocking,
        "warnings": warnings,
    }


def _cache_read_dominates_stage(stage: Mapping[str, object]) -> bool:
    read = stage.get("cache_read_seconds")
    elapsed = stage.get("stage_elapsed_seconds")
    if not isinstance(read, int | float) or isinstance(read, bool):
        return False
    if not isinstance(elapsed, int | float) or isinstance(elapsed, bool) or float(elapsed) <= 0.0:
        return False
    return float(read) / float(elapsed) > 0.50


def _expected_runtime_values_from_effective_config(
    effective_config: Mapping[str, object],
) -> dict[str, object]:
    application = effective_config.get("compute_profile_application")
    if not isinstance(application, Mapping):
        return {}
    expected: dict[str, object] = {}
    for section_name in ("active_applied", "dataloader_applied"):
        section = application.get(section_name)
        if isinstance(section, Mapping):
            expected.update({str(key): value for key, value in section.items()})
    calibration_application = effective_config.get("compute_calibration_application")
    if isinstance(calibration_application, Mapping):
        selected = calibration_application.get("selected_overrides")
        if isinstance(selected, Mapping):
            expected.update({str(key): value for key, value in selected.items()})
    return {
        _runtime_value_alias(key): value
        for key, value in expected.items()
    }


def _runtime_value_alias(key: str) -> str:
    aliases = {
        "batch_size": "batch_size",
        "denoiser_batch_size": "denoiser_batch_size",
        "source_batch_size": "source_batch_size",
        "frame_batch_size": "frame_batch_size",
        "tokenizer_batch_size": "tokenizer_batch_size",
        "text_to_token_batch_size": "text_to_token_batch_size",
        "reconstruction_batch_size": "reconstruction_batch_size",
        "decode_batch_size": "decode_batch_size",
        "num_workers": "num_workers",
    }
    return aliases.get(key, key)


def _required_mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ModelWorkflowInvariantError(
            f"compute_profile_application.{key} must be a mapping."
        )
    return value


def _a100_readiness_summary_markdown(payload: Mapping[str, object]) -> str:
    blockers = payload.get("blocking_issues")
    blockers = blockers if isinstance(blockers, list) else []
    warnings = payload.get("warnings")
    warnings = warnings if isinstance(warnings, list) else []
    blocker_lines = tuple(f"- {issue}" for issue in blockers) or ("- none",)
    warning_lines = tuple(f"- {warning}" for warning in warnings) or ("- none",)
    return "\n".join(
        (
            "# A100 Readiness Summary",
            "",
            f"- compute_profile: {payload.get('compute_profile')}",
            f"- device_name: {payload.get('device_name')}",
            f"- bf16_supported: {payload.get('bf16_supported')}",
            f"- active_overrides_applied: {payload.get('active_overrides_applied')}",
            f"- provider_batch_fields_complete: {payload.get('provider_batch_fields_complete')}",
            f"- telemetry_complete: {payload.get('telemetry_complete')}",
            f"- gpu_utilization_class: {payload.get('gpu_utilization_class')}",
            f"- calibration_available: {payload.get('calibration_available')}",
            f"- full_run_recommendation: {payload.get('full_run_recommendation')}",
            f"- recommended_for_full_run: {payload.get('recommended_for_full_run')}",
            "",
            "## Blocking Issues",
            "",
            *blocker_lines,
            "",
            "## Warnings",
            "",
            *warning_lines,
            "",
        )
    )


def _max_float(values) -> float | None:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    return max(numeric, default=None)


def _worst_utilization_class(stages: list[dict[str, object]]) -> str:
    order = {"no_cuda": 0, "underutilized": 1, "moderate": 2, "high": 3, "near_limit": 4}
    observed = [str(stage.get("gpu_utilization_class")) for stage in stages]
    if not observed:
        return "unknown"
    return max(observed, key=lambda value: order.get(value, -1))


__all__ = ["write_model_reports"]
