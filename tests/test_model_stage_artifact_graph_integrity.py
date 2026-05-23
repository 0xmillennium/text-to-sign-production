from __future__ import annotations

from pathlib import Path

import pytest

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.candidates import (
    ModelExecutionResult,
    ModelStageArtifactRef,
    ModelStageKind,
    ModelStageResult,
    ModelStageStatus,
    default_stage_plan_for_request,
)
from text_to_sign_production.modeling.candidates.artifacts import (
    GeneratedPoseManifestArtifactSubtype,
    generated_pose_manifest_artifact_ref,
)
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.model.contracts import ModelWorkflowInvariantError
from text_to_sign_production.workflows.model.contracts.config import ModelWorkflowConfig
from text_to_sign_production.workflows.model.layout import ModelLayout, build_model_layout
from text_to_sign_production.workflows.model.processing.stages import (
    ensure_no_exact_duplicate_model_artifact_refs,
)


@pytest.mark.unit
def test_exact_duplicate_artifact_refs_are_rejected(tmp_path: Path) -> None:
    layout = _layout(tmp_path, ModelKey.BASE_DIRECT)
    path = layout.outputs.generated_pose_split_outputs[0].manifest_path
    duplicate = generated_pose_manifest_artifact_ref(
        path=path,
        split=SampleSplit.VAL,
        producer_stage="export_generated_pose",
        generation_mode="deterministic",
        model_key="base_direct",
        artifact_subtype=GeneratedPoseManifestArtifactSubtype.FINAL_VALIDATION,
    )
    execution = _execution(
        layout,
        ModelKey.BASE_DIRECT,
        (
            (
                ModelStageKind.GENERATE,
                (duplicate,),
            ),
            (
                ModelStageKind.EXPORT_GENERATED_POSE,
                (duplicate,),
            ),
        ),
    )

    with pytest.raises(ModelWorkflowInvariantError, match="duplicate model stage artifact ref"):
        ensure_no_exact_duplicate_model_artifact_refs(execution)


@pytest.mark.unit
def test_same_path_with_different_role_is_not_globally_forbidden(tmp_path: Path) -> None:
    layout = _layout(tmp_path, ModelKey.BASE_DIRECT)
    path = layout.outputs.model_run_root / "shared.json"
    execution = _execution(
        layout,
        ModelKey.BASE_DIRECT,
        (
            (
                ModelStageKind.GENERATE,
                (
                    ModelStageArtifactRef(path=path, kind="model_intermediate", role="role_a"),
                    ModelStageArtifactRef(path=path, kind="model_intermediate", role="role_b"),
                ),
            ),
        ),
    )

    ensure_no_exact_duplicate_model_artifact_refs(execution)


@pytest.mark.unit
def test_learned_pose_token_final_generated_manifest_ref_is_not_duplicated(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path, ModelKey.LEARNED_POSE_TOKEN)
    decoded_manifest = (
        layout.outputs.model_run_root
        / "intermediates/decode_to_pose/decoded_pose_intermediates/val/generated_pose/manifest.jsonl"
    )
    final_manifest = layout.outputs.generated_pose_split_outputs[0].manifest_path
    execution = _execution(
        layout,
        ModelKey.LEARNED_POSE_TOKEN,
        (
            (
                ModelStageKind.DECODE_TO_POSE,
                (
                    generated_pose_manifest_artifact_ref(
                        path=decoded_manifest,
                        kind="generated_pose_manifest",
                        split=SampleSplit.VAL,
                        producer_stage="decode_to_pose",
                        generation_mode="predicted_length",
                        model_key="learned_pose_token",
                        artifact_subtype=GeneratedPoseManifestArtifactSubtype.DECODED_INTERMEDIATE,
                    ),
                ),
            ),
            (
                ModelStageKind.EXPORT_GENERATED_POSE,
                (
                    generated_pose_manifest_artifact_ref(
                        path=final_manifest,
                        kind="generated_pose_manifest",
                        split=SampleSplit.VAL,
                        producer_stage="export_generated_pose",
                        generation_mode="predicted_length",
                        model_key="learned_pose_token",
                        artifact_subtype=GeneratedPoseManifestArtifactSubtype.FINAL_VALIDATION,
                    ),
                ),
            ),
        ),
    )

    ensure_no_exact_duplicate_model_artifact_refs(execution)
    final_refs = [
        artifact
        for artifact in execution.artifact_refs
        if artifact.path.resolve(strict=False) == final_manifest.resolve(strict=False)
        and artifact.kind == "generated_pose_manifest"
        and artifact.role == "generated_pose_manifest"
    ]
    assert len(final_refs) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_key", "generation_mode"),
    (
        (ModelKey.BASE_DIRECT, "deterministic"),
        (ModelKey.LEARNED_POSE_TOKEN, "predicted_length"),
        (ModelKey.LATENT_DIFFUSION, "stochastic"),
        (ModelKey.ARTICULATOR_AWARE, "predicted_length"),
    ),
)
def test_provider_final_generated_pose_manifest_refs_are_duplicate_free(
    tmp_path: Path,
    model_key: ModelKey,
    generation_mode: str,
) -> None:
    layout = _layout(tmp_path, model_key)
    final_manifest = layout.outputs.generated_pose_split_outputs[0].manifest_path
    execution = _execution(
        layout,
        model_key,
        (
            (
                ModelStageKind.EXPORT_GENERATED_POSE,
                (
                    generated_pose_manifest_artifact_ref(
                        path=final_manifest,
                        kind="generated_pose_manifest",
                        split=SampleSplit.VAL,
                        producer_stage="export_generated_pose",
                        generation_mode=generation_mode,
                        model_key=model_key.value,
                        artifact_subtype=GeneratedPoseManifestArtifactSubtype.FINAL_VALIDATION,
                    ),
                ),
            ),
        ),
    )

    ensure_no_exact_duplicate_model_artifact_refs(execution)


def _layout(tmp_path: Path, model_key: ModelKey) -> ModelLayout:
    config = ModelWorkflowConfig(
        project_root=tmp_path,
        drive_project_root=tmp_path / "drive",
        model_key=model_key,
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_name="run001",
    )
    return build_model_layout(config)


def _execution(
    layout: ModelLayout,
    model_key: ModelKey,
    stages: tuple[tuple[ModelStageKind, tuple[ModelStageArtifactRef, ...]], ...],
) -> ModelExecutionResult:
    request = layout.config.to_model_run_request()
    stage_plan = default_stage_plan_for_request(request)
    planned_by_kind = {stage.spec.kind: stage for stage in stage_plan.stages}
    return ModelExecutionResult(
        model_key=model_key,
        run_name=layout.config.run_name,
        stages=tuple(
            ModelStageResult(
                stage=planned_by_kind[kind],
                status=ModelStageStatus.COMPLETED,
                artifacts=artifacts,
            )
            for kind, artifacts in stages
        ),
    )
