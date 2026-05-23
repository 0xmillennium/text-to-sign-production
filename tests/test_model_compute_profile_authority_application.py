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


def _copy_project_file(project_root: Path, relpath: Path) -> None:
    target = project_root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    copy2(ROOT / relpath, target)


def _provider_config(tmp_path: Path, model_key: ModelKey):
    ensure_model_provider_registered(model_key)
    _copy_project_file(tmp_path, CONFIG_RELPATHS[model_key])
    _copy_project_file(
        tmp_path,
        Path("configs/modeling/compute_profiles/colab_a100_80gb.yaml"),
    )
    workflow = ModelWorkflow(
        ModelWorkflowConfig(
            project_root=tmp_path,
            drive_project_root=tmp_path / "drive",
            model_key=model_key,
            manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
            model_config_relpath=CONFIG_RELPATHS[model_key],
            run_mode=ModelRunMode.FULL,
            compute_profile="colab_a100_80gb",
            run_name=f"test-{model_key.value}-a100-full",
        )
    )
    assert workflow.layout is not None
    source = workflow.layout.runtime.model_config_original_path
    target = workflow.layout.runtime.model_config_path
    assert source is not None
    assert target is not None
    target.parent.mkdir(parents=True, exist_ok=True)
    copy2(source, target)
    plan = workflow.plan_runtime()
    research = workflow.resolve_research(plan.execution_inputs)
    provider_resolution = workflow.resolve_provider(research.request)
    return workflow.load_provider_config(provider_resolution, research.request)


@pytest.mark.parametrize("model_key", tuple(CONFIG_RELPATHS))
def test_full_a100_compute_profile_requested_keys_are_classified(
    tmp_path: Path,
    model_key: ModelKey,
) -> None:
    provider_config = _provider_config(tmp_path, model_key)
    effective = provider_config.loaded_config.effective_config
    application = effective["compute_profile_application"]
    assert isinstance(application, Mapping)

    requested_active = set(application["active_requested"])
    classified_active = (
        set(application["active_applied"])
        | set(application["active_not_applicable"])
        | set(application["active_unsupported"])
    )
    assert requested_active == classified_active

    requested_dataloader = set(application["dataloader_requested"])
    classified_dataloader = (
        set(application["dataloader_applied"])
        | set(application["dataloader_not_applicable"])
        | set(application["dataloader_unsupported"])
    )
    assert requested_dataloader == classified_dataloader

    requested_candidates = set(application["candidates_requested"])
    classified_candidates = (
        set(application["candidates_applicable"])
        | set(application["candidates_not_applicable"])
        | set(application["candidates_unsupported"])
    )
    assert requested_candidates == classified_candidates

    assert application["active_unsupported"] == {}
    assert application["dataloader_unsupported"] == {}
    assert application["candidates_unsupported"] == {}
