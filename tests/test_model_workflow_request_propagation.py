from __future__ import annotations

from pathlib import Path
from shutil import copy2

import pytest

from text_to_sign_production.modeling.candidates import (
    ModelRunMode,
    ObjectiveAttachmentError,
)
from text_to_sign_production.modeling.candidates.bootstrap import (
    ensure_model_provider_registered,
)
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey, ObjectiveKey
from text_to_sign_production.workflows.model.contracts import ModelWorkflowConfig
from text_to_sign_production.workflows.model.workflow import ModelWorkflow


ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_MODELS = (
    ModelKey.LEARNED_POSE_TOKEN,
    ModelKey.LATENT_DIFFUSION,
    ModelKey.ARTICULATOR_AWARE,
)
CONFIG_RELPATHS = {
    ModelKey.BASE_DIRECT: Path("configs/modeling/base_direct.yaml"),
    ModelKey.LEARNED_POSE_TOKEN: Path("configs/modeling/learned_pose_token.yaml"),
    ModelKey.LATENT_DIFFUSION: Path("configs/modeling/latent_diffusion.yaml"),
    ModelKey.ARTICULATOR_AWARE: Path("configs/modeling/articulator_aware.yaml"),
}


def _workflow(
    tmp_path: Path,
    model_key: ModelKey,
    *,
    semantic: bool = True,
) -> ModelWorkflow:
    ensure_model_provider_registered(model_key)
    _copy_project_config(tmp_path, CONFIG_RELPATHS[model_key])
    if semantic:
        _copy_project_config(
            tmp_path,
            Path("configs/modeling/objectives/semantic_consistency.yaml"),
        )
    config = ModelWorkflowConfig(
        project_root=tmp_path,
        drive_project_root=tmp_path / "drive",
        model_key=model_key,
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        model_config_relpath=CONFIG_RELPATHS[model_key],
        run_mode=ModelRunMode.SMOKE,
        run_name=f"test-{model_key.value}",
        auxiliary_objectives=(
            (ObjectiveKey.SEMANTIC_CONSISTENCY,) if semantic else ()
        ),
    )
    return ModelWorkflow(config)


def _copy_project_config(project_root: Path, relpath: Path) -> None:
    target = project_root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    copy2(ROOT / relpath, target)


def _materialize_runtime_snapshots(workflow: ModelWorkflow) -> None:
    layout = workflow.layout
    assert layout is not None
    if layout.runtime.model_config_path is not None:
        layout.runtime.model_config_path.parent.mkdir(parents=True, exist_ok=True)
        assert layout.runtime.model_config_original_path is not None
        copy2(layout.runtime.model_config_original_path, layout.runtime.model_config_path)
    if layout.runtime.semantic_objective_config_path is not None:
        layout.runtime.semantic_objective_config_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        assert layout.runtime.semantic_objective_config_original_path is not None
        copy2(
            layout.runtime.semantic_objective_config_original_path,
            layout.runtime.semantic_objective_config_path,
        )


@pytest.mark.parametrize("model_key", SEMANTIC_MODELS)
def test_runtime_plan_preserves_semantic_objective_config_path(
    tmp_path: Path,
    model_key: ModelKey,
) -> None:
    workflow = _workflow(tmp_path, model_key)

    plan = workflow.plan_runtime()
    request = plan.execution_inputs.request
    assert workflow.layout is not None

    assert ObjectiveKey.SEMANTIC_CONSISTENCY in request.auxiliary_objectives
    assert ObjectiveKey.SEMANTIC_CONSISTENCY in request.objective_config_paths
    assert request.objective_config_paths[ObjectiveKey.SEMANTIC_CONSISTENCY] == (
        workflow.layout.runtime.semantic_objective_config_path
    )


@pytest.mark.parametrize("model_key", SEMANTIC_MODELS)
def test_runtime_validation_accepts_semantic_request_with_snapshot_path(
    tmp_path: Path,
    model_key: ModelKey,
) -> None:
    workflow = _workflow(tmp_path, model_key)

    plan = workflow.plan_runtime()
    workflow.validate_runtime_plan(plan)


@pytest.mark.parametrize("model_key", SEMANTIC_MODELS)
def test_research_resolution_preserves_execution_request_identity(
    tmp_path: Path,
    model_key: ModelKey,
) -> None:
    workflow = _workflow(tmp_path, model_key)
    plan = workflow.plan_runtime()

    resolution = workflow.resolve_research(plan.execution_inputs)

    assert resolution.request == plan.execution_inputs.request
    assert (
        resolution.request.objective_config_paths
        == plan.execution_inputs.request.objective_config_paths
    )


@pytest.mark.parametrize("model_key", SEMANTIC_MODELS)
def test_provider_config_result_carries_request(
    tmp_path: Path,
    model_key: ModelKey,
) -> None:
    workflow = _workflow(tmp_path, model_key)
    _materialize_runtime_snapshots(workflow)
    plan = workflow.plan_runtime()
    research = workflow.resolve_research(plan.execution_inputs)
    provider_resolution = workflow.resolve_provider(research.request)

    provider_config = workflow.load_provider_config(
        provider_resolution,
        research.request,
    )

    assert provider_config.request == research.request


@pytest.mark.parametrize("model_key", SEMANTIC_MODELS)
def test_stage_plan_uses_provider_config_request(
    tmp_path: Path,
    model_key: ModelKey,
) -> None:
    workflow = _workflow(tmp_path, model_key)
    _materialize_runtime_snapshots(workflow)
    plan = workflow.plan_runtime()
    research = workflow.resolve_research(plan.execution_inputs)
    provider_resolution = workflow.resolve_provider(research.request)
    provider_config = workflow.load_provider_config(
        provider_resolution,
        research.request,
    )

    stage_plan = workflow.plan_model_stages(provider_config)

    assert stage_plan.stage_plan.request == provider_config.request


def test_base_direct_rejects_semantic_objective(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path, ModelKey.BASE_DIRECT)
    plan = workflow.plan_runtime()

    with pytest.raises(ObjectiveAttachmentError):
        workflow.resolve_research(plan.execution_inputs)
