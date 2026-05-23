"""Generic metadata artifact writing for a model run."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from text_to_sign_production.modeling.candidates import (
    MODEL_RUNTIME_SUPPORT_SCHEMA_VERSION,
    ModelProvider,
    ModelProviderLoadedConfig,
    ModelRuntimeSupportManifest,
    ModelStageKind,
    ModelStagePlan,
    model_run_request_to_dict,
    resolve_model_run_mode_policy,
    write_runtime_support_manifest,
)
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.data import resolve_modeling_manifest_path
from text_to_sign_production.modeling.validation import (
    ValidationChannelMetricKey,
    ValidationMetricKey,
)
from text_to_sign_production.workflows.foundation.provenance import written_file_receipt
from text_to_sign_production.workflows.foundation.review import write_json
from text_to_sign_production.workflows.model.contracts import (
    ModelResearchResolution,
    ModelObjectiveArtifactResult,
    ModelRunMetadataArtifacts,
    ModelStageArtifactReceiptResult,
    ModelStageArtifactSkip,
    ModelStageExecutionWorkflowResult,
    ModelWorkflowInvariantError,
)
from text_to_sign_production.workflows.model.layout import ModelLayout


def write_model_run_metadata_artifacts(
    *,
    layout: ModelLayout,
    research: ModelResearchResolution,
    provider: ModelProvider,
    loaded_config: ModelProviderLoadedConfig,
    stage_plan: ModelStagePlan,
    stage_execution: ModelStageExecutionWorkflowResult | None = None,
    objective_artifacts: tuple[ModelObjectiveArtifactResult, ...] = (),
    execution_id: str,
) -> ModelRunMetadataArtifacts:
    """Write workflow-owned effective config, research, and run metadata JSON."""

    if loaded_config.model_key is not research.request.model_key:
        raise ModelWorkflowInvariantError("loaded config does not match research request")
    if provider.spec.key is not research.request.model_key:
        raise ModelWorkflowInvariantError("provider does not match research request")
    if stage_plan.request != research.request:
        raise ModelWorkflowInvariantError("stage plan does not match research request")
    if stage_plan.model.key is not research.model_spec.key:
        raise ModelWorkflowInvariantError("stage plan model does not match research model")
    if stage_execution is not None and stage_execution.stage_plan != stage_plan:
        raise ModelWorkflowInvariantError("stage execution does not match metadata stage plan")
    if research.request.auxiliary_objectives and not objective_artifacts:
        raise ModelWorkflowInvariantError(
            "requested auxiliary objectives require materialized objective artifacts "
            "before model metadata is written."
        )
    _require_execution_id(execution_id)
    training_metadata = _checkpoint_stage_metadata(stage_execution)
    export_metadata = _stage_metadata(stage_execution, ModelStageKind.EXPORT_GENERATED_POSE)
    best_checkpoint_path = _existing_path_or_none(training_metadata.get("best_checkpoint_path"))
    last_checkpoint_path = _existing_path_or_none(training_metadata.get("last_checkpoint_path"))
    validation_manifest_path = _validation_manifest_path(layout)
    validation_samples_root = _validation_samples_root(layout)
    manifest_family = research.request.manifest_family
    manifest_paths: dict[str, Path | None] = {
        "drive_train_manifest_path": resolve_modeling_manifest_path(
            layout.stores.drive,
            manifest_family,
            research.request.train_split,
        ),
        "drive_validation_manifest_path": resolve_modeling_manifest_path(
            layout.stores.drive,
            manifest_family,
            research.request.validation_split,
        ),
        "drive_test_manifest_path": resolve_modeling_manifest_path(
            layout.stores.drive,
            manifest_family,
            SampleSplit.TEST,
        ),
        "runtime_train_manifest_path": resolve_modeling_manifest_path(
            layout.stores.runtime,
            manifest_family,
            research.request.train_split,
        ),
        "runtime_validation_manifest_path": resolve_modeling_manifest_path(
            layout.stores.runtime,
            manifest_family,
            research.request.validation_split,
        ),
        "runtime_test_manifest_path": None,
    }
    run_mode_policy = resolve_model_run_mode_policy(research.request.run_mode)
    semantic_metadata = _semantic_objective_metadata(objective_artifacts)
    support_manifest = _runtime_support_manifest(
        provider=provider,
        layout=layout,
        research=research,
        loaded_config=loaded_config,
        stage_execution=stage_execution,
    )
    write_runtime_support_manifest(
        layout.outputs.runtime_support_manifest_path,
        support_manifest,
    )

    effective_config = {
        "schema_version": "model_effective_config.v1",
        "request": model_run_request_to_dict(research.request),
        "source_path": (
            None if loaded_config.source_path is None else str(loaded_config.source_path)
        ),
        "effective_config": _json_value(loaded_config.effective_config),
    }
    research_spec = {
        "schema_version": "model_research_spec.v1",
        "model": research.model_spec.to_dict(),
        "objectives": [objective.to_dict() for objective in research.objective_specs],
        "trace_issues": [issue.to_dict() for issue in research.trace_issues],
    }
    run_metadata = {
        "schema_version": "model_run_metadata.v2",
        "model_key": research.request.model_key.value,
        "model_run_name": research.request.run_name,
        "run_name": research.request.run_name,
        **manifest_family.to_dict(),
        **{
            key: None if value is None else str(value)
            for key, value in manifest_paths.items()
        },
        "train_split": research.request.train_split.value,
        "validation_split": research.request.validation_split.value,
        "test_split": "test",
        "prediction_splits": [split.value for split in research.request.prediction_splits],
        "auxiliary_objectives": [
            objective.value for objective in research.request.auxiliary_objectives
        ],
        **(
            {}
            if semantic_metadata is None
            else {"semantic_objective": semantic_metadata}
        ),
        "run_mode": research.request.run_mode.value,
        "run_mode_policy": run_mode_policy.to_dict(),
        "run_mode_quality_claim": run_mode_policy.quality_claim,
        "seed": research.request.seed,
        "model_config_snapshot_path": (
            None
            if layout.runtime.model_config_path is None
            else str(layout.runtime.model_config_path)
        ),
        "effective_config_path": str(layout.outputs.effective_config_path),
        "research_spec_path": str(layout.outputs.research_spec_path),
        "runtime_support_manifest_path": str(layout.outputs.runtime_support_manifest_path),
        "runtime_support": {
            "schema_version": support_manifest.schema_version,
            "manifest_path": str(layout.outputs.runtime_support_manifest_path),
            "artifact_count": len(support_manifest.artifacts),
            "required_for_test_model_count": sum(
                1 for artifact in support_manifest.artifacts if artifact.required_for_test_model
            ),
        },
        "calibration": _calibration_metadata(stage_execution),
        "best_checkpoint_path": (
            None if best_checkpoint_path is None else str(best_checkpoint_path)
        ),
        "last_checkpoint_path": (
            None if last_checkpoint_path is None else str(last_checkpoint_path)
        ),
        "checkpoint_selection_metric": training_metadata.get("best_metric_name"),
        "checkpoint_selection_metric_value": training_metadata.get("best_metric_value"),
        "validation_generated_pose_manifest": (
            None if validation_manifest_path is None else str(validation_manifest_path)
        ),
        "validation_generated_pose_samples_root": (
            None if validation_samples_root is None else str(validation_samples_root)
        ),
        "validation_report_index": str(layout.reports.index_json_path),
        "validation_report_index_written": layout.reports.index_json_path.is_file(),
        "validation_artifact_paths": {
            "pairing_manifest": str(layout.reports.validation_pairing_manifest_path),
            "metric_results": str(layout.reports.validation_metric_results_path),
            "channel_metric_results": str(layout.reports.validation_channel_metric_results_path),
            "aggregate_metrics": str(layout.reports.validation_aggregate_metrics_path),
            "channel_aggregate_metrics": str(
                layout.reports.validation_channel_aggregate_metrics_path
            ),
            "limitations": str(layout.reports.validation_limitations_path),
            "summary_markdown": str(layout.reports.validation_summary_report_path),
        },
        "validation_metric_policy": {
            "split": SampleSplit.VAL.value,
            "alignment_policy": "simple_prefix_minimum_sequence_length",
            "metric_keys": [metric.value for metric in ValidationMetricKey],
            "channel_metric_keys": [metric.value for metric in ValidationChannelMetricKey],
        },
        "status": (
            "completed"
            if stage_execution is not None and stage_execution.execution.completed
            else "metadata_written_before_stage_completion"
        ),
        "stage_plan": [
            {
                "index": stage.index,
                "kind": stage.spec.kind.value,
                "provider_stage_id": stage.provider_stage_id,
                "required": stage.spec.required,
                "produces_generated_pose": stage.spec.produces_generated_pose,
            }
            for stage in stage_plan.stages
        ],
        "stage_results": [] if stage_execution is None else [
            {
                "index": result.stage.index,
                "kind": result.stage.spec.kind.value,
                "status": result.status.value,
                "artifact_roles": [artifact.role for artifact in result.artifacts],
                "metrics": dict(result.metrics),
                "metadata": _json_value(dict(result.metadata)),
            }
            for result in stage_execution.execution.stages
        ],
        "execution_id": execution_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "limitations": [
            "Model workflow predictions are validation split outputs for development and checkpoint selection.",
            "The test split is recorded for the test_model workflow and is not processed by model.ipynb.",
            "Validation outputs are not full test split evaluation and are not aggregate model performance.",
            f"Run-mode quality claim is {run_mode_policy.quality_claim!r}.",
            *(
                []
                if semantic_metadata is None
                else list(semantic_metadata["limitations"])
            ),
        ],
    }
    write_json(layout.outputs.effective_config_path, effective_config)
    write_json(layout.outputs.research_spec_path, research_spec)
    write_json(layout.outputs.run_metadata_path, run_metadata)
    return ModelRunMetadataArtifacts(
        execution_id=execution_id,
        effective_config_path=layout.outputs.effective_config_path,
        research_spec_path=layout.outputs.research_spec_path,
        run_metadata_path=layout.outputs.run_metadata_path,
        runtime_support_manifest_path=layout.outputs.runtime_support_manifest_path,
        effective_config=written_file_receipt(
            "model effective config",
            layout.outputs.effective_config_path,
            execution_id=execution_id,
            kind="model_effective_config",
        ),
        research_spec=written_file_receipt(
            "model research spec",
            layout.outputs.research_spec_path,
            execution_id=execution_id,
            kind="model_research_spec",
        ),
        run_metadata=written_file_receipt(
            "model run metadata",
            layout.outputs.run_metadata_path,
            execution_id=execution_id,
            kind="model_run_metadata",
        ),
        runtime_support_manifest=written_file_receipt(
            "model runtime support manifest",
            layout.outputs.runtime_support_manifest_path,
            execution_id=execution_id,
            kind="model_runtime_support_manifest",
        ),
    )


def _runtime_support_manifest(
    *,
    provider: ModelProvider,
    layout: ModelLayout,
    research: ModelResearchResolution,
    loaded_config: ModelProviderLoadedConfig,
    stage_execution: ModelStageExecutionWorkflowResult | None,
) -> ModelRuntimeSupportManifest:
    artifacts = ()
    if stage_execution is not None:
        try:
            artifacts = provider.runtime_support_artifacts(
                request=research.request,
                loaded_config=loaded_config,
                execution=stage_execution.execution,
                topology=layout.stores.runtime,
            )
        except FileNotFoundError as exc:
            raise ModelWorkflowInvariantError(str(exc)) from exc
        except Exception as exc:
            raise ModelWorkflowInvariantError(
                "provider runtime support artifact declaration failed"
            ) from exc
    config_kind = loaded_config.effective_config.get("schema_version")
    return ModelRuntimeSupportManifest(
        schema_version=MODEL_RUNTIME_SUPPORT_SCHEMA_VERSION,
        model_key=research.request.model_key.value,
        model_run_name=research.request.run_name,
        manifest_family=research.request.manifest_family.family_id,
        provider_config_kind=(
            config_kind if isinstance(config_kind, str) and config_kind.strip()
            else f"{research.request.model_key.value}.effective_config"
        ),
        artifacts=tuple(artifacts),
    )


def materialize_model_stage_artifact_receipts(
    *,
    layout: ModelLayout,
    stage_execution: ModelStageExecutionWorkflowResult,
    execution_id: str,
) -> ModelStageArtifactReceiptResult:
    """Receipt provider-produced stage files and index skipped stage refs."""

    _require_execution_id(execution_id)
    runtime_root = layout.stores.runtime.repo_root.resolve(strict=False)
    receipts = []
    skipped: list[ModelStageArtifactSkip] = []
    for artifact in stage_execution.execution.artifact_refs:
        path = artifact.path.resolve(strict=False)
        try:
            path.relative_to(runtime_root)
        except ValueError as exc:
            raise ModelWorkflowInvariantError(
                "provider stage artifact is outside runtime repository root: "
                f"role={artifact.role}, path={path}. "
                "Provider outputs must be materialized under the runtime artifact store."
            ) from exc
        if not path.exists():
            skipped.append(
                ModelStageArtifactSkip(
                    role=artifact.role,
                    kind=artifact.kind,
                    path=path,
                    reason="artifact path does not exist",
                )
            )
            continue
        if path.is_dir():
            skipped.append(
                ModelStageArtifactSkip(
                    role=artifact.role,
                    kind=artifact.kind,
                    path=path,
                    reason="directory artifact refs are indexed as skipped; publish files",
                )
            )
            continue
        if not path.is_file():
            skipped.append(
                ModelStageArtifactSkip(
                    role=artifact.role,
                    kind=artifact.kind,
                    path=path,
                    reason="artifact path is not a regular file",
                )
            )
            continue
        receipts.append(
            written_file_receipt(
                f"model stage artifact {artifact.role}",
                path,
                execution_id=execution_id,
                kind=artifact.kind,
            )
        )
    index_path = layout.outputs.stage_artifacts_index_path
    write_json(
        index_path,
        {
            "schema_version": "model_stage_artifacts_index.v1",
            "execution_id": execution_id,
            "model_key": stage_execution.execution.model_key.value,
            "run_name": stage_execution.execution.run_name,
            "publish_ownership_note": (
                "Stage artifact refs are provenance/index entries; persistent publish "
                "ownership is handled by metadata, checkpoints, runtime_support_manifest, "
                "generated_pose_archive, reports, validation, and objectives."
            ),
            "receipts": [
                {
                    "label": receipt.label,
                    "kind": receipt.kind,
                    "path": str(receipt.path),
                    "sha256": receipt.sha256,
                    "execution_id": receipt.execution_id,
                }
                for receipt in receipts
            ],
            "skipped_artifacts": [
                {
                    "role": skip.role,
                    "kind": skip.kind,
                    "path": None if skip.path is None else str(skip.path),
                    "reason": skip.reason,
                }
                for skip in skipped
            ],
        },
    )
    index_receipt = written_file_receipt(
        "model stage artifacts index",
        index_path,
        execution_id=execution_id,
        kind="model_stage_artifacts_index",
    )
    return ModelStageArtifactReceiptResult(
        execution_id=execution_id,
        receipts=tuple(receipts),
        skipped_artifacts=tuple(skipped),
        index_path=index_path,
        index_receipt=index_receipt,
    )


def _json_value(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ModelWorkflowInvariantError("provider config mapping keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise ModelWorkflowInvariantError(
        f"provider effective config is not JSON serializable: {type(value).__name__}"
    )


def _stage_metadata(
    stage_execution: ModelStageExecutionWorkflowResult | None,
    kind: ModelStageKind,
) -> dict[str, object]:
    if stage_execution is None:
        return {}
    for result in stage_execution.execution.stages:
        if result.stage.spec.kind is kind:
            return dict(result.metadata)
    return {}


def _checkpoint_stage_metadata(
    stage_execution: ModelStageExecutionWorkflowResult | None,
) -> dict[str, object]:
    if stage_execution is None:
        return {}
    selected: dict[str, object] = {}
    for result in stage_execution.execution.stages:
        metadata = dict(result.metadata)
        if "best_checkpoint_path" in metadata or "last_checkpoint_path" in metadata:
            selected.update(metadata)
    return selected


def _calibration_metadata(stage_execution: ModelStageExecutionWorkflowResult | None) -> dict[str, object]:
    pre = None if stage_execution is None else getattr(stage_execution, "pre_calibration", None)
    if pre is None:
        return {
            "required": False,
            "executed": False,
            "reused": False,
            "compute_calibration_path": None,
            "selected_overrides_path": None,
            "calibrated_effective_config_path": None,
            "base_effective_config_hash": None,
            "selected_overrides_hash": None,
            "calibrated_effective_config_hash": None,
            "representative_surfaces": [],
            "selected_overrides": {},
            "applied_overrides": {},
        }
    return {
        "required": bool(getattr(pre, "required")),
        "executed": bool(getattr(pre, "executed")),
        "reused": bool(getattr(pre, "reused")),
        "compute_calibration_path": _path_str(getattr(pre, "compute_calibration_path")),
        "selected_overrides_path": _path_str(getattr(pre, "selected_overrides_path")),
        "calibrated_effective_config_path": _path_str(
            getattr(pre, "calibrated_effective_config_path")
        ),
        "base_effective_config_hash": getattr(pre, "base_effective_config_hash"),
        "selected_overrides_hash": getattr(pre, "selected_overrides_hash"),
        "calibrated_effective_config_hash": getattr(pre, "calibrated_effective_config_hash"),
        "representative_surfaces": [
            {
                "provider_key": item.provider_key,
                "candidate_key": item.candidate_key,
                "surface_kind": item.surface_kind,
                "split": item.split,
                "source_manifest_path": str(item.source_manifest_path),
                "source_manifest_sha256": item.source_manifest_sha256,
                "provider_config_sha256": item.provider_config_sha256,
                "surface_schema_hash": item.surface_schema_hash,
                "sample_count": item.sample_count,
                "unit_count": item.unit_count,
                "surface_root": None if item.surface_root is None else str(item.surface_root),
                "surface_metadata_path": None
                if item.surface_metadata_path is None
                else str(item.surface_metadata_path),
                "dataloader_kind": item.dataloader_kind,
            }
            for item in getattr(pre, "representative_surfaces", ())
        ],
        "selected_overrides": dict(getattr(pre, "selected_overrides")),
        "applied_overrides": dict(getattr(pre, "applied_overrides")),
    }


def _path_str(value: object) -> str | None:
    return None if value is None else str(value)


def _existing_path_or_none(value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path if path.is_file() else None


def _validation_manifest_path(layout: ModelLayout) -> Path | None:
    for split_output in layout.outputs.generated_pose_split_outputs:
        if split_output.split.value == "val":
            return split_output.manifest_path if split_output.manifest_path.is_file() else None
    return None


def _validation_samples_root(layout: ModelLayout) -> Path | None:
    for split_output in layout.outputs.generated_pose_split_outputs:
        if split_output.split.value == "val":
            return split_output.samples_root if split_output.samples_root.is_dir() else None
    return None


def _require_execution_id(execution_id: str) -> None:
    if not isinstance(execution_id, str) or not execution_id.strip():
        raise ModelWorkflowInvariantError("execution_id must be non-empty")


def _semantic_objective_metadata(
    objective_artifacts: tuple[ModelObjectiveArtifactResult, ...],
) -> dict[str, object] | None:
    semantic = tuple(
        result
        for result in objective_artifacts
        if result.objective_key.value == "semantic_consistency"
    )
    if not semantic:
        return None
    if len(semantic) != 1:
        raise ModelWorkflowInvariantError(
            "metadata requires at most one semantic_consistency objective artifact result"
        )
    result = semantic[0]
    return {
        "attached": True,
        "requires_ablation": True,
        "ablation_status": "incomplete",
        "proxy_only": True,
        "config_snapshot_path": str(result.config_snapshot_path),
        "config_snapshot_sha256": result.config_snapshot_sha256,
        "generated_manifest_path": str(result.generated_pose_manifest_path),
        "generated_manifest_artifact_subtype": "final_validation",
        "candidate_policy": result.candidate_policy,
        "ablation_readiness_path": str(result.ablation_readiness_path),
        "ready_for_comparison": result.ready_for_comparison,
        "required_baseline_missing": result.required_baseline_missing,
        "readiness_issues": list(result.readiness_issues),
        "alignment_aggregate_path": str(result.artifact_paths["alignment_aggregate"]),
        "records_count": result.records_count,
        "skipped_count": result.skipped_count,
        "limitations": [
            "Semantic consistency artifacts are post-generation proxy evaluation only.",
            "Semantic consistency is not differentiable training optimization in this stage.",
            "An executed with/without baseline is required before contribution claims.",
            "No completed semantic ablation exists in this stage.",
            "Embedding similarity is not proof of sign intelligibility or linguistic semantic correctness.",
            "No retrieval, gloss, dictionary, or avatar/rendering mechanism is active.",
        ],
    }


__all__ = [
    "materialize_model_stage_artifact_receipts",
    "write_model_run_metadata_artifacts",
]
