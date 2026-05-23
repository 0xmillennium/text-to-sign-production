from __future__ import annotations

from pathlib import Path
from shutil import copy2

import pytest

from text_to_sign_production.modeling.candidates import ModelRunMode
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
    ModelKey.LEARNED_POSE_TOKEN: Path("configs/modeling/learned_pose_token.yaml"),
    ModelKey.LATENT_DIFFUSION: Path("configs/modeling/latent_diffusion.yaml"),
    ModelKey.ARTICULATOR_AWARE: Path("configs/modeling/articulator_aware.yaml"),
}


def _workflow(tmp_path: Path, model_key: ModelKey) -> ModelWorkflow:
    ensure_model_provider_registered(model_key)
    _copy_project_file(tmp_path, CONFIG_RELPATHS[model_key])
    _copy_project_file(
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
        run_name=f"test-{model_key.value}-semantic",
        auxiliary_objectives=(ObjectiveKey.SEMANTIC_CONSISTENCY,),
    )
    return ModelWorkflow(config)


def _copy_project_file(project_root: Path, relpath: Path) -> None:
    target = project_root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    copy2(ROOT / relpath, target)


def _materialize_runtime_snapshots(workflow: ModelWorkflow) -> None:
    assert workflow.layout is not None
    for source, target in (
        (
            workflow.layout.runtime.model_config_original_path,
            workflow.layout.runtime.model_config_path,
        ),
        (
            workflow.layout.runtime.semantic_objective_config_original_path,
            workflow.layout.runtime.semantic_objective_config_path,
        ),
    ):
        assert source is not None
        assert target is not None
        target.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, target)


@pytest.mark.parametrize("model_key", SEMANTIC_MODELS)
def test_semantic_objective_snapshot_path_reaches_provider_request_and_effective_config(
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

    assert (
        ObjectiveKey.SEMANTIC_CONSISTENCY
        in provider_config.request.objective_config_paths
    )
    assert provider_config.loaded_config.effective_config["objective_config_paths"][
        "semantic_consistency"
    ].endswith("provenance/config/objectives/semantic_consistency.yaml")
