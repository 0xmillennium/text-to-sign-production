from __future__ import annotations

import json
from pathlib import Path

import pytest

from text_to_sign_production.modeling.candidates import (
    MODEL_RUNTIME_SUPPORT_SCHEMA_VERSION,
    ModelExecutionResult,
    ModelProviderLoadedConfig,
    ModelRuntimeSupportManifest,
    ModelStageArtifactRef,
    ModelStageResult,
    ModelStageStatus,
    default_stage_plan_for_request,
    require_model_provider,
    support_artifact_from_model_run_file,
    write_runtime_support_manifest,
)
from text_to_sign_production.modeling.candidates.bootstrap import ensure_model_provider_registered
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.foundation.provenance import written_file_receipt
from text_to_sign_production.workflows.model.contracts import (
    ModelPublishSourceBundle,
    ModelRunMetadataArtifacts,
    ModelStageExecutionWorkflowResult,
    ModelWorkflowInvariantError,
)
from text_to_sign_production.workflows.model.contracts.config import ModelWorkflowConfig
from text_to_sign_production.workflows.model.layout import ModelLayout, build_model_layout
from text_to_sign_production.workflows.model.publish.plan import (
    _runtime_support_source_paths,
    build_model_publish_plan,
)


@pytest.mark.unit
def test_learned_pose_token_tokenizer_publish_owner_is_runtime_support(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path, ModelKey.LEARNED_POSE_TOKEN)
    source_path = layout.outputs.model_run_root / "intermediates/representation/checkpoints/tokenizer.pt"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"tokenizer")
    metadata = _metadata(layout, runtime_support_paths=((
        source_path,
        "learned_pose_token_tokenizer_checkpoint",
        "learned_pose_token",
    ),))
    stage_execution = _stage_execution(
        layout,
        ModelKey.LEARNED_POSE_TOKEN,
        (
            ModelStageArtifactRef(
                path=source_path,
                kind="model_representation_tokenizer_checkpoint",
                role="representation_artifact",
            ),
        ),
    )

    plan = build_model_publish_plan(
        layout=layout,
        sources=ModelPublishSourceBundle(
            metadata_artifacts=metadata,
            stage_execution=stage_execution,
        ),
    )

    targets = _targets_named(plan, "tokenizer.pt")
    assert len(targets) == 1
    assert targets[0].source_path == source_path.resolve(strict=False)
    assert targets[0].label == "publish runtime support learned_pose_token_tokenizer_checkpoint"
    assert targets[0].kind == "model_provider_support"
    assert _runtime_support_source_paths(ModelPublishSourceBundle(metadata_artifacts=metadata)) == {
        source_path.resolve(strict=False)
    }


@pytest.mark.unit
def test_learned_pose_token_standardization_stats_publish_owner_is_runtime_support(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path, ModelKey.LEARNED_POSE_TOKEN)
    source_path = layout.outputs.model_run_root / "intermediates/representation/standardization_stats.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("{}\n", encoding="utf-8")
    metadata = _metadata(layout, runtime_support_paths=((
        source_path,
        "learned_pose_token_standardization_stats",
        "learned_pose_token",
    ),))
    stage_execution = _stage_execution(
        layout,
        ModelKey.LEARNED_POSE_TOKEN,
        (
            ModelStageArtifactRef(
                path=source_path,
                kind="model_representation_standardization_stats",
                role="representation_artifact",
            ),
        ),
    )

    plan = build_model_publish_plan(
        layout=layout,
        sources=ModelPublishSourceBundle(
            metadata_artifacts=metadata,
            stage_execution=stage_execution,
        ),
    )

    targets = _targets_named(plan, "standardization_stats.json")
    assert len(targets) == 1
    assert targets[0].label == "publish runtime support learned_pose_token_standardization_stats"
    assert targets[0].kind == "model_provider_support"


@pytest.mark.unit
def test_latent_temporal_autoencoder_publish_owner_is_runtime_support(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path, ModelKey.LATENT_DIFFUSION)
    source_path = layout.outputs.model_run_root / "intermediates/latent/autoencoder/window_autoencoder.pt"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"autoencoder")
    metadata = _metadata(layout, runtime_support_paths=((
        source_path,
        "latent_diffusion_autoencoder_checkpoint",
        "latent_diffusion",
    ),))
    stage_execution = _stage_execution(
        layout,
        ModelKey.LATENT_DIFFUSION,
        (
            ModelStageArtifactRef(
                path=source_path,
                kind="model_latent_autoencoder_checkpoint",
                role="latent_autoencoder_checkpoint",
            ),
        ),
    )

    plan = build_model_publish_plan(
        layout=layout,
        sources=ModelPublishSourceBundle(
            metadata_artifacts=metadata,
            stage_execution=stage_execution,
        ),
    )

    targets = _targets_named(plan, "window_autoencoder.pt")
    assert len(targets) == 1
    assert targets[0].label == "publish runtime support latent_diffusion_autoencoder_checkpoint"
    assert targets[0].kind == "model_provider_support"


@pytest.mark.unit
def test_canonical_best_and_last_checkpoints_are_explicit_checkpoint_sources(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path, ModelKey.LEARNED_POSE_TOKEN)
    best = layout.outputs.checkpoints_root / "best.pt"
    last = layout.outputs.checkpoints_root / "last.pt"
    best.parent.mkdir(parents=True, exist_ok=True)
    best.write_bytes(b"best")
    last.write_bytes(b"last")
    metadata = _metadata(
        layout,
        run_metadata={
            "best_checkpoint_path": str(best),
            "last_checkpoint_path": str(last),
        },
    )

    plan = build_model_publish_plan(
        layout=layout,
        sources=ModelPublishSourceBundle(metadata_artifacts=metadata),
    )

    best_targets = _targets_named(plan, "best.pt")
    last_targets = _targets_named(plan, "last.pt")
    assert len(best_targets) == 1
    assert best_targets[0].label == "publish best checkpoint"
    assert best_targets[0].kind == "model_checkpoint"
    assert len(last_targets) == 1
    assert last_targets[0].label == "publish last checkpoint"
    assert last_targets[0].kind == "model_checkpoint"


@pytest.mark.unit
def test_true_duplicate_publish_target_still_errors_with_source_details(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path, ModelKey.BASE_DIRECT)
    metadata = _metadata(layout, runtime_support_paths=((
        layout.outputs.run_metadata_path,
        "base_direct_compatibility_config",
        "base_direct",
    ),))

    with pytest.raises(ModelWorkflowInvariantError) as excinfo:
        build_model_publish_plan(
            layout=layout,
            sources=ModelPublishSourceBundle(metadata_artifacts=metadata),
        )

    message = str(excinfo.value)
    assert "duplicate model publish target" in message
    assert "target_path=" in message
    assert "previous_label=" in message
    assert "previous_kind=" in message
    assert "previous_source_path=" in message
    assert "current_label=" in message
    assert "current_kind=" in message
    assert "current_source_path=" in message


@pytest.mark.unit
def test_stage_artifact_refs_are_not_raw_publish_sources(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path, ModelKey.LEARNED_POSE_TOKEN)
    stage_paths = (
        layout.outputs.model_run_root / "intermediates/representation/checkpoints/tokenizer.pt",
        layout.outputs.model_run_root / "intermediates/latent/autoencoder/window_autoencoder.pt",
        layout.outputs.model_run_root / "intermediates/representation/intermediate.json",
        layout.outputs.model_run_root / "reports/provider_report.md",
    )
    for path in stage_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")
    stage_execution = _stage_execution(
        layout,
        ModelKey.LEARNED_POSE_TOKEN,
        (
            ModelStageArtifactRef(
                path=stage_paths[0],
                kind="model_representation_tokenizer_checkpoint",
                role="representation_artifact",
            ),
            ModelStageArtifactRef(
                path=stage_paths[1],
                kind="model_latent_autoencoder_checkpoint",
                role="latent_autoencoder_checkpoint",
            ),
            ModelStageArtifactRef(
                path=stage_paths[2],
                kind="model_intermediate",
                role="provider_intermediate",
            ),
            ModelStageArtifactRef(
                path=stage_paths[3],
                kind="model_report",
                role="provider_report",
            ),
        ),
    )

    plan = build_model_publish_plan(
        layout=layout,
        sources=ModelPublishSourceBundle(stage_execution=stage_execution),
    )

    planned_sources = {target.source_path for target in plan.targets}
    assert all(path.resolve(strict=False) not in planned_sources for path in stage_paths)


def _layout(tmp_path: Path, model_key: ModelKey) -> ModelLayout:
    config = ModelWorkflowConfig(
        project_root=tmp_path,
        drive_project_root=tmp_path / "drive",
        model_key=model_key,
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_name="run001",
    )
    return build_model_layout(config)


def _metadata(
    layout: ModelLayout,
    *,
    run_metadata: dict[str, object] | None = None,
    runtime_support_paths: tuple[tuple[Path, str, str], ...] = (),
) -> ModelRunMetadataArtifacts:
    for path in (
        layout.outputs.effective_config_path,
        layout.outputs.research_spec_path,
        layout.outputs.run_metadata_path,
        layout.outputs.runtime_support_manifest_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    layout.outputs.effective_config_path.write_text("{}\n", encoding="utf-8")
    layout.outputs.research_spec_path.write_text("{}\n", encoding="utf-8")
    metadata_payload = {
        "schema_version": "model_run_metadata.v2",
        "model_key": layout.config.model_key.value,
        "run_name": layout.config.run_name,
        "best_checkpoint_path": None,
        "last_checkpoint_path": None,
    }
    if run_metadata is not None:
        metadata_payload.update(run_metadata)
    layout.outputs.run_metadata_path.write_text(
        json.dumps(metadata_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts = tuple(
        support_artifact_from_model_run_file(
            model_run_root=layout.outputs.model_run_root,
            path=path,
            role=role,
            provider_key=provider_key,
        )
        for path, role, provider_key in runtime_support_paths
    )
    write_runtime_support_manifest(
        layout.outputs.runtime_support_manifest_path,
        ModelRuntimeSupportManifest(
            schema_version=MODEL_RUNTIME_SUPPORT_SCHEMA_VERSION,
            model_key=layout.config.model_key.value,
            model_run_name=layout.config.run_name,
            manifest_family=layout.config.manifest_family.family_id,
            provider_config_kind=f"{layout.config.model_key.value}.effective_config",
            artifacts=artifacts,
        ),
    )
    return ModelRunMetadataArtifacts(
        execution_id="exec001",
        effective_config_path=layout.outputs.effective_config_path,
        research_spec_path=layout.outputs.research_spec_path,
        run_metadata_path=layout.outputs.run_metadata_path,
        runtime_support_manifest_path=layout.outputs.runtime_support_manifest_path,
        effective_config=written_file_receipt(
            "model effective config",
            layout.outputs.effective_config_path,
            execution_id="exec001",
            kind="model_effective_config",
        ),
        research_spec=written_file_receipt(
            "model research spec",
            layout.outputs.research_spec_path,
            execution_id="exec001",
            kind="model_research_spec",
        ),
        run_metadata=written_file_receipt(
            "model run metadata",
            layout.outputs.run_metadata_path,
            execution_id="exec001",
            kind="model_run_metadata",
        ),
        runtime_support_manifest=written_file_receipt(
            "model runtime support manifest",
            layout.outputs.runtime_support_manifest_path,
            execution_id="exec001",
            kind="model_runtime_support_manifest",
        ),
    )


def _stage_execution(
    layout: ModelLayout,
    model_key: ModelKey,
    artifacts: tuple[ModelStageArtifactRef, ...],
) -> ModelStageExecutionWorkflowResult:
    ensure_model_provider_registered(model_key)
    provider = require_model_provider(model_key)
    request = layout.config.to_model_run_request()
    stage_plan = default_stage_plan_for_request(request)
    loaded_config = ModelProviderLoadedConfig(
        model_key=model_key,
        source_path=None,
        raw_config={},
        effective_config={},
    )
    execution = ModelExecutionResult(
        model_key=model_key,
        run_name=layout.config.run_name,
        stages=(
            ModelStageResult(
                stage=stage_plan.stages[0],
                status=ModelStageStatus.COMPLETED,
                artifacts=artifacts,
            ),
        ),
    )
    return ModelStageExecutionWorkflowResult(
        provider=provider,
        loaded_config=loaded_config,
        stage_plan=stage_plan,
        execution=execution,
    )


def _targets_named(plan, name: str):
    return [target for target in plan.targets if target.source_path.name == name]
