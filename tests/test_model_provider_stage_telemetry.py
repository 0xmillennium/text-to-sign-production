from __future__ import annotations

from pathlib import Path

from text_to_sign_production.modeling.candidates import ModelRunMode
from text_to_sign_production.modeling.candidates.configs import ModelRunRequest
from text_to_sign_production.modeling.candidates.articulator_aware.config import (
    articulator_aware_config_from_mapping,
)
from text_to_sign_production.modeling.candidates.articulator_aware.provider import (
    ArticulatorAwareProvider,
    _articulator_aware_performance_metadata,
)
from text_to_sign_production.modeling.candidates.base_direct.config import (
    load_base_direct_config,
)
from text_to_sign_production.modeling.candidates.base_direct.provider import (
    _base_direct_performance_metadata,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.config import (
    latent_diffusion_config_from_mapping,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.provider import (
    LatentDiffusionProvider,
    _latent_diffusion_performance_metadata,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.config import (
    load_learned_pose_token_config,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.provider import (
    _learned_pose_token_performance_metadata,
)
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.model.contracts.compute_profiles import (
    load_model_compute_profile,
)

ROOT = Path(__file__).resolve().parents[1]


def _request(model_key: ModelKey, config_relpath: str) -> ModelRunRequest:
    return ModelRunRequest(
        model_key=model_key,
        run_name=f"test-{model_key.value}-full-a100",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_mode=ModelRunMode.FULL,
        config_path=ROOT / config_relpath,
        compute_profile=load_model_compute_profile(ROOT, "colab_a100_80gb").to_dict(),
    )


def test_base_direct_stage_metadata_exposes_a100_fields() -> None:
    request = _request(ModelKey.BASE_DIRECT, "configs/modeling/base_direct.yaml")
    config = load_base_direct_config(request.config_path, request=request)

    metadata = _base_direct_performance_metadata(config)

    assert metadata["batch_size"] == 8
    assert metadata["num_workers"] == 4
    assert metadata["prefetch_factor"] == 4


def test_learned_pose_token_stage_metadata_exposes_a100_fields() -> None:
    request = _request(
        ModelKey.LEARNED_POSE_TOKEN,
        "configs/modeling/learned_pose_token.yaml",
    )
    config = load_learned_pose_token_config(request.config_path, request=request)

    metadata = _learned_pose_token_performance_metadata(config)

    assert metadata["tokenizer_batch_size"] == 512
    assert metadata["decode_batch_size"] == 256
    assert metadata["num_workers"] == 4
    assert "materialization_workers" not in metadata


def test_latent_diffusion_stage_metadata_exposes_a100_fields() -> None:
    request = _request(
        ModelKey.LATENT_DIFFUSION,
        "configs/modeling/latent_diffusion.yaml",
    )
    loaded = LatentDiffusionProvider().load_config(request)
    provider_config = loaded.effective_config["provider_config"]
    assert isinstance(provider_config, dict)
    config = latent_diffusion_config_from_mapping(provider_config)

    metadata = _latent_diffusion_performance_metadata(config)

    assert metadata["denoiser_batch_size"] == 128
    assert metadata["num_workers"] == 4
    assert metadata["latent_autoencoder_active"] is False
    assert "autoencoder_batch_size" not in metadata


def test_articulator_aware_stage_metadata_exposes_a100_fields() -> None:
    request = _request(
        ModelKey.ARTICULATOR_AWARE,
        "configs/modeling/articulator_aware.yaml",
    )
    loaded = ArticulatorAwareProvider().load_config(request)
    provider_config = loaded.effective_config["provider_config"]
    assert isinstance(provider_config, dict)
    config = articulator_aware_config_from_mapping(provider_config)

    metadata = _articulator_aware_performance_metadata(config)

    assert metadata["source_batch_size"] == 64
    assert metadata["frame_batch_size"] == 2048
    assert metadata["num_workers"] == 4
