"""Thin workflow coordination for attached model auxiliary objectives."""

from __future__ import annotations

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.candidates import (
    ModelStageArtifactRef,
    is_generated_pose_manifest_role,
)
from text_to_sign_production.modeling.data import load_manifest_samples
from text_to_sign_production.modeling.objectives.semantic_consistency import (
    load_semantic_consistency_config,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.integration import (
    write_semantic_objective_artifacts_for_validation_outputs,
)
from text_to_sign_production.modeling.registry import require_model_spec, require_objective_spec
from text_to_sign_production.modeling.research import ObjectiveKey
from text_to_sign_production.workflows.foundation.provenance import written_file_receipt
from text_to_sign_production.workflows.model.contracts import (
    ModelObjectiveArtifactResult,
    ModelStageExecutionWorkflowResult,
    ModelWorkflowInvariantError,
)
from text_to_sign_production.workflows.model.layout import ModelLayout


def write_model_objective_artifacts(
    *,
    layout: ModelLayout,
    stage_execution: ModelStageExecutionWorkflowResult,
    execution_id: str,
) -> tuple[ModelObjectiveArtifactResult, ...]:
    """Materialize requested post-generation objective outputs after provider export."""

    requested = stage_execution.stage_plan.request.auxiliary_objectives
    if not requested:
        return ()
    if any(objective is not ObjectiveKey.SEMANTIC_CONSISTENCY for objective in requested):
        raise ModelWorkflowInvariantError(
            "unsupported auxiliary objective requested for model workflow processing; "
            "only semantic_consistency is integrated in this stage."
    )
    manifest_ref = find_current_final_validation_generated_pose_manifest_ref(
        layout=layout,
        stage_execution=stage_execution,
    )
    manifest_path = manifest_ref.path.resolve(strict=False)
    generated_payload_root = manifest_path.parent
    config_snapshot_path = layout.runtime.semantic_objective_config_path
    if config_snapshot_path is None or not config_snapshot_path.is_file():
        raise ModelWorkflowInvariantError(
            "semantic_consistency requires its runtime config snapshot before processing; "
            "restore and verify runtime so configs/objectives/semantic_consistency.yaml is materialized."
        )
    request = stage_execution.stage_plan.request
    try:
        config = load_semantic_consistency_config(config_snapshot_path)
        sources = load_manifest_samples(
            layout.stores.runtime,
            request.manifest_family,
            SampleSplit.VAL,
        )
        bundle = write_semantic_objective_artifacts_for_validation_outputs(
            model_key=request.model_key,
            run_name=request.run_name,
            split=SampleSplit.VAL,
            objective_config=config,
            model_spec=require_model_spec(request.model_key),
            objective_spec=require_objective_spec(ObjectiveKey.SEMANTIC_CONSISTENCY),
            manifest_family=request.manifest_family,
            config_snapshot_path=config_snapshot_path,
            generated_pose_manifest_path=manifest_path,
            generated_payload_root=generated_payload_root,
            source_manifest_samples=sources,
            output_root=layout.reports.semantic_objective_root,
            run_metadata_path=layout.outputs.run_metadata_path,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ModelWorkflowInvariantError(
            f"semantic_consistency objective artifact processing failed: {exc}"
        ) from exc
    receipts = tuple(
        written_file_receipt(
            f"semantic objective artifact {artifact.role}",
            artifact.path,
            execution_id=execution_id,
            kind=artifact.kind,
        )
        for artifact in bundle.artifacts
    )
    return (
        ModelObjectiveArtifactResult(
            objective_key=ObjectiveKey.SEMANTIC_CONSISTENCY,
            artifact_paths={
                "config": bundle.config_path,
                "attachment_decision": bundle.attachment_decision_path,
                "ablation_plan": bundle.ablation_plan_path,
                "ablation_readiness": bundle.ablation_readiness_path,
                "text_embeddings": bundle.text_embeddings_path,
                "pose_embeddings": bundle.pose_embeddings_path,
                "alignment_results": bundle.alignment_results_path,
                "alignment_aggregate": bundle.alignment_aggregate_path,
                **{
                    f"report_{path.stem}": path
                    for path in bundle.report_paths
                },
            },
            artifact_refs=bundle.artifacts,
            records_count=bundle.records_count,
            skipped_count=bundle.skipped_count,
            receipts=receipts,
            ablation_readiness_path=bundle.ablation_readiness_path,
            candidate_policy=bundle.candidate_policy.policy,
            generated_pose_manifest_path=bundle.generated_pose_manifest_path,
            config_snapshot_path=bundle.config_snapshot_path,
            config_snapshot_sha256=bundle.config_snapshot_sha256,
            ready_for_comparison=bundle.ready_for_comparison,
            required_baseline_missing=bundle.required_baseline_missing,
            readiness_issues=bundle.readiness_issues,
        ),
    )


def find_current_final_validation_generated_pose_manifest_ref(
    *,
    layout: ModelLayout,
    stage_execution: ModelStageExecutionWorkflowResult,
) -> ModelStageArtifactRef:
    """Return the current completed run's strict final validation manifest reference."""

    if not stage_execution.execution.completed:
        raise ModelWorkflowInvariantError(
            "semantic_consistency requires completed current provider stage execution "
            "before reading final validation generated poses."
        )
    request = stage_execution.stage_plan.request
    expected_metadata = {
        "split": SampleSplit.VAL.value,
        "artifact_subtype": "final_validation",
        "model_key": request.model_key.value,
        "producer_stage": "export_generated_pose",
    }
    for artifact in stage_execution.execution.artifact_refs:
        if not is_generated_pose_manifest_role(artifact.role):
            continue
        if all(artifact.metadata.get(key) == value for key, value in expected_metadata.items()):
            resolved = artifact.path.resolve(strict=False)
            runtime_root = layout.stores.runtime.repo_root.resolve(strict=False)
            if not resolved.is_relative_to(runtime_root):
                raise ModelWorkflowInvariantError(
                    "semantic_consistency final_validation manifest must be under the current "
                    f"runtime output root {runtime_root}: {resolved}"
                )
            if not resolved.is_file():
                raise ModelWorkflowInvariantError(
                    "semantic_consistency final_validation manifest artifact ref is not a "
                    f"materialized file: {resolved}"
                )
            return artifact
    raise ModelWorkflowInvariantError(
        "semantic_consistency requires the current completed provider artifact "
        "role='generated_pose_manifest', split='val', artifact_subtype='final_validation', "
        f"model_key='{request.model_key.value}', producer_stage='export_generated_pose'. "
        "Decoded, reconstruction, test_model, missing-subtype, and stale layout outputs "
        "cannot be used."
    )


__all__ = [
    "find_current_final_validation_generated_pose_manifest_ref",
    "write_model_objective_artifacts",
]
