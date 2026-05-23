from pathlib import Path

from text_to_sign_production.modeling.candidates import ModelRunMode
from text_to_sign_production.modeling.candidates.configs import ModelRunRequest
from text_to_sign_production.modeling.candidates.articulator_aware.provider import (
    ArticulatorAwareProvider,
)
from text_to_sign_production.modeling.candidates.base_direct.config import (
    load_base_direct_config,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.provider import (
    LatentDiffusionProvider,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.config import (
    load_learned_pose_token_config,
)
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.model.contracts.compute_profiles import (
    load_model_compute_profile,
    provider_candidate_overrides,
)


ROOT = Path(__file__).resolve().parents[1]


def _request(mode: str, profile: str) -> ModelRunRequest:
    return _provider_request(
        ModelKey.LEARNED_POSE_TOKEN,
        mode,
        profile,
        ROOT / "configs/modeling/learned_pose_token.yaml",
    )


def _provider_request(
    model_key: ModelKey,
    mode: str,
    profile: str,
    config_path: Path,
) -> ModelRunRequest:
    return ModelRunRequest(
        model_key=model_key,
        run_name=f"test-{mode}",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_mode=ModelRunMode(mode),
        config_path=config_path,
        compute_profile=load_model_compute_profile(ROOT, profile).to_dict(),
    )


def _training(mode: str, profile: str = "colab_a100_80gb"):
    return load_learned_pose_token_config(
        ROOT / "configs/modeling/learned_pose_token.yaml",
        request=_request(mode, profile),
    ).training


def test_a100_smoke_active_overrides_apply() -> None:
    training = _training("smoke")
    assert training.tokenizer_batch_size == 128
    assert training.text_to_token_batch_size == 64
    assert training.reconstruction_batch_size == 64
    assert training.decode_batch_size == 64
    assert training.materialization_workers == 0
    assert training.cache_materialized_sources is True
    assert training.num_workers == 4


def test_a100_debug_active_overrides_apply() -> None:
    training = _training("debug")
    assert training.tokenizer_batch_size == 256
    assert training.text_to_token_batch_size == 128
    assert training.reconstruction_batch_size == 128
    assert training.decode_batch_size == 128


def test_a100_full_active_overrides_apply() -> None:
    config = load_learned_pose_token_config(
        ROOT / "configs/modeling/learned_pose_token.yaml",
        request=_request("full", "colab_a100_80gb"),
    )
    training = config.training
    assert training.tokenizer_batch_size == 512
    assert training.text_to_token_batch_size == 256
    assert training.reconstruction_batch_size == 256
    assert training.decode_batch_size == 256
    assert training.cache_materialized_sources is True
    assert training.num_workers == 4
    assert "materialization_workers" not in config.compute_profile_active_overrides


def test_candidates_remain_visible_but_not_applied_unless_active() -> None:
    profile = load_model_compute_profile(ROOT, "colab_a100_80gb").to_dict()
    candidates = provider_candidate_overrides(
        profile,
        provider_key="learned_pose_token",
    )
    assert candidates["tokenizer_batch_size"] == [128, 256, 512, 1024]
    assert _training("debug").tokenizer_batch_size == 256


def test_portable_profile_does_not_apply_a100_active_overrides() -> None:
    training = _training("debug", "portable")
    assert training.tokenizer_batch_size is None
    assert training.effective_tokenizer_batch_size == training.batch_size


def test_base_direct_a100_smoke_active_overrides_apply() -> None:
    request = _provider_request(
        ModelKey.BASE_DIRECT,
        "smoke",
        "colab_a100_80gb",
        ROOT / "configs/modeling/base_direct.yaml",
    )

    config = load_base_direct_config(request.config_path, request=request)

    assert config.training.batch_size == 2
    assert config.compute_profile_active_overrides["batch_size"] == 2


def test_base_direct_a100_full_dataloader_overrides_apply() -> None:
    request = _provider_request(
        ModelKey.BASE_DIRECT,
        "full",
        "colab_a100_80gb",
        ROOT / "configs/modeling/base_direct.yaml",
    )

    config = load_base_direct_config(request.config_path, request=request)

    assert config.training.batch_size == 8
    assert config.training.num_workers == 4
    assert config.training.pin_memory is True
    assert config.training.persistent_workers is True
    assert config.training.prefetch_factor == 4


def test_articulator_aware_a100_smoke_active_overrides_apply() -> None:
    request = _provider_request(
        ModelKey.ARTICULATOR_AWARE,
        "smoke",
        "colab_a100_80gb",
        ROOT / "configs/modeling/articulator_aware.yaml",
    )

    loaded = ArticulatorAwareProvider().load_config(request)
    effective = loaded.effective_config
    provider_config = effective["provider_config"]
    assert isinstance(provider_config, dict)

    assert effective["compute_profile_active_overrides"] == {
        "source_batch_size": 16,
        "frame_batch_size": 512,
    }
    assert provider_config["training"]["batch_size"] == 16
    assert provider_config["training"]["frame_batch_size"] == 512
    assert effective["training"]["source_batch_size"] == 16
    assert effective["training"]["frame_batch_size"] == 512


def test_articulator_aware_a100_full_dataloader_overrides_apply() -> None:
    request = _provider_request(
        ModelKey.ARTICULATOR_AWARE,
        "full",
        "colab_a100_80gb",
        ROOT / "configs/modeling/articulator_aware.yaml",
    )

    loaded = ArticulatorAwareProvider().load_config(request)
    effective = loaded.effective_config
    provider_config = effective["provider_config"]
    assert isinstance(provider_config, dict)

    assert provider_config["training"]["batch_size"] == 64
    assert provider_config["training"]["frame_batch_size"] == 2048
    assert provider_config["training"]["num_workers"] == 4


def test_latent_diffusion_a100_smoke_active_overrides_apply() -> None:
    request = _provider_request(
        ModelKey.LATENT_DIFFUSION,
        "smoke",
        "colab_a100_80gb",
        ROOT / "configs/modeling/latent_diffusion.yaml",
    )

    loaded = LatentDiffusionProvider().load_config(request)
    effective = loaded.effective_config
    provider_config = effective["provider_config"]
    assert isinstance(provider_config, dict)

    assert effective["compute_profile_active_overrides"] == {
        "denoiser_batch_size": 32,
    }
    assert provider_config["training"]["batch_size"] == 32
    assert provider_config["training"]["num_workers"] == 4
    assert effective["latent_autoencoder"]["active"] is False
    assert "autoencoder_batch_size" not in effective["compute_profile_active_overrides"]
    assert "autoencoder_batch_size" not in effective["calibration_override_bindings"]
    assert "generation_batch_size" not in effective["compute_profile_active_overrides"]


def test_latent_diffusion_a100_full_denoiser_overrides_apply() -> None:
    request = _provider_request(
        ModelKey.LATENT_DIFFUSION,
        "full",
        "colab_a100_80gb",
        ROOT / "configs/modeling/latent_diffusion.yaml",
    )

    loaded = LatentDiffusionProvider().load_config(request)
    effective = loaded.effective_config
    provider_config = effective["provider_config"]
    assert isinstance(provider_config, dict)

    assert provider_config["training"]["batch_size"] == 128
    assert provider_config["training"]["num_workers"] == 4
    assert "autoencoder_batch_size" not in effective["compute_profile_active_overrides"]
    assert "autoencoder_batch_size" not in effective["calibration_override_bindings"]
    assert effective["latent_autoencoder"]["active"] is False


def test_latent_diffusion_profile_candidates_exclude_generation_batch_size() -> None:
    request = _provider_request(
        ModelKey.LATENT_DIFFUSION,
        "full",
        "colab_a100_80gb",
        ROOT / "configs/modeling/latent_diffusion.yaml",
    )
    loaded = LatentDiffusionProvider().load_config(request)

    candidates = provider_candidate_overrides(
        request.compute_profile,
        provider_key="latent_diffusion",
    )
    policy = LatentDiffusionProvider().calibration_policy(loaded)

    assert "generation_batch_size" not in candidates
    assert "generation_batch_size" not in policy.candidate_keys
    assert "autoencoder_batch_size" not in candidates
    assert "autoencoder_batch_size" not in policy.candidate_keys
