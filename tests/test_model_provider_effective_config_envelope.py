from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from shutil import copy2

import pytest

from text_to_sign_production.modeling.candidates import ModelRunMode
from text_to_sign_production.modeling.candidates.bootstrap import (
    ensure_model_provider_registered,
)
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.model.contracts import ModelWorkflowConfig
from text_to_sign_production.workflows.model.workflow import ModelWorkflow


ROOT = Path(__file__).resolve().parents[1]
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
    compute_profile: str,
) -> ModelWorkflow:
    ensure_model_provider_registered(model_key)
    _copy_project_file(tmp_path, CONFIG_RELPATHS[model_key])
    if compute_profile != "portable":
        _copy_project_file(
            tmp_path,
            Path(f"configs/modeling/compute_profiles/{compute_profile}.yaml"),
        )
    config = ModelWorkflowConfig(
        project_root=tmp_path,
        drive_project_root=tmp_path / "drive",
        model_key=model_key,
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        model_config_relpath=CONFIG_RELPATHS[model_key],
        run_mode=ModelRunMode.SMOKE,
        compute_profile=compute_profile,
        run_name=f"test-{model_key.value}-{compute_profile}",
    )
    return ModelWorkflow(config)


def _copy_project_file(project_root: Path, relpath: Path) -> None:
    target = project_root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    copy2(ROOT / relpath, target)


def _materialize_model_config_snapshot(workflow: ModelWorkflow) -> None:
    assert workflow.layout is not None
    source = workflow.layout.runtime.model_config_original_path
    target = workflow.layout.runtime.model_config_path
    assert source is not None
    assert target is not None
    target.parent.mkdir(parents=True, exist_ok=True)
    copy2(source, target)


def _provider_config(workflow: ModelWorkflow):
    _materialize_model_config_snapshot(workflow)
    plan = workflow.plan_runtime()
    research = workflow.resolve_research(plan.execution_inputs)
    provider_resolution = workflow.resolve_provider(research.request)
    return workflow.load_provider_config(provider_resolution, research.request)


@pytest.mark.parametrize("model_key", tuple(CONFIG_RELPATHS))
def test_provider_effective_config_exposes_audit_envelope(
    tmp_path: Path,
    model_key: ModelKey,
) -> None:
    workflow = _workflow(tmp_path, model_key, compute_profile="portable")

    provider_config = _provider_config(workflow)
    effective = provider_config.loaded_config.effective_config

    assert effective["model_key"] == model_key.value
    assert effective["run_name"] == workflow.config.run_name
    assert effective["manifest_family"] == workflow.config.manifest_family.family_id
    assert effective["run_mode"] == workflow.config.run_mode.value
    assert isinstance(effective["compute_profile"], Mapping)
    assert isinstance(effective["auxiliary_objectives"], list)
    assert isinstance(effective["objective_config_paths"], Mapping)
    assert isinstance(effective["compute_profile_active_overrides"], Mapping)


@pytest.mark.parametrize("model_key", tuple(CONFIG_RELPATHS))
def test_provider_effective_config_exposes_a100_profile_name(
    tmp_path: Path,
    model_key: ModelKey,
) -> None:
    workflow = _workflow(tmp_path, model_key, compute_profile="colab_a100_80gb")

    provider_config = _provider_config(workflow)
    effective = provider_config.loaded_config.effective_config

    assert effective["compute_profile"]["name"] == "colab_a100_80gb"
