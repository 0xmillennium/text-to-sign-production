from __future__ import annotations

from pathlib import Path

import pytest

from text_to_sign_production.modeling.candidates import ModelRunMode
from text_to_sign_production.modeling.candidates.configs import ModelRunRequest
from text_to_sign_production.modeling.candidates.bootstrap import (
    ensure_model_provider_registered,
)
from text_to_sign_production.modeling.candidates.registry import require_model_provider
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.model.contracts.compute_profiles import (
    load_model_compute_profile,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_RELPATHS = {
    ModelKey.BASE_DIRECT: Path("configs/modeling/base_direct.yaml"),
    ModelKey.LEARNED_POSE_TOKEN: Path("configs/modeling/learned_pose_token.yaml"),
    ModelKey.LATENT_DIFFUSION: Path("configs/modeling/latent_diffusion.yaml"),
    ModelKey.ARTICULATOR_AWARE: Path("configs/modeling/articulator_aware.yaml"),
}
EXPECTED_KEYS = {
    ModelKey.BASE_DIRECT: ("batch_size",),
    ModelKey.LEARNED_POSE_TOKEN: (
        "tokenizer_batch_size",
        "text_to_token_batch_size",
        "reconstruction_batch_size",
        "decode_batch_size",
    ),
    ModelKey.LATENT_DIFFUSION: ("denoiser_batch_size",),
    ModelKey.ARTICULATOR_AWARE: ("source_batch_size", "frame_batch_size"),
}


def _request(model_key: ModelKey) -> ModelRunRequest:
    return ModelRunRequest(
        model_key=model_key,
        run_name=f"test-{model_key.value}-full-a100",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_mode=ModelRunMode.FULL,
        config_path=ROOT / CONFIG_RELPATHS[model_key],
        compute_profile=load_model_compute_profile(ROOT, "colab_a100_80gb").to_dict(),
    )


@pytest.mark.parametrize("model_key", tuple(CONFIG_RELPATHS))
def test_calibration_policy_uses_compute_profile_application(model_key: ModelKey) -> None:
    ensure_model_provider_registered(model_key)
    provider = require_model_provider(model_key)
    loaded = provider.load_config(_request(model_key))
    application = loaded.effective_config["compute_profile_application"]

    policy = provider.calibration_policy(loaded)

    assert set(policy.candidate_keys) == set(application["calibration_candidate_keys"])
    assert set(policy.override_targets) == set(policy.candidate_keys)
    assert tuple(policy.candidate_keys) == EXPECTED_KEYS[model_key]
    forbidden = {
        "materialization_workers",
        "autoencoder_batch_size",
        "generation_batch_size",
    }
    assert forbidden.isdisjoint(policy.candidate_keys)
    assert forbidden.isdisjoint(policy.override_targets)
