from pathlib import Path

from text_to_sign_production.modeling.candidates import ModelRunMode
from text_to_sign_production.modeling.candidates.configs import ModelRunRequest
from text_to_sign_production.modeling.candidates.learned_pose_token.config import (
    learned_pose_token_config_from_effective_dict,
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


def _request(profile: str = "colab_a100_80gb") -> ModelRunRequest:
    return ModelRunRequest(
        model_key=ModelKey.LEARNED_POSE_TOKEN,
        run_name="test-debug",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_mode=ModelRunMode.DEBUG,
        config_path=ROOT / "configs/modeling/learned_pose_token.yaml",
        compute_profile=load_model_compute_profile(ROOT, profile).to_dict(),
    )


def test_explicit_batch_fields_parse_correctly() -> None:
    config = load_learned_pose_token_config(
        ROOT / "configs/modeling/learned_pose_token.yaml",
        request=_request(),
    )
    assert config.training.tokenizer_batch_size == 256
    assert config.training.text_to_token_batch_size == 128
    assert config.training.reconstruction_batch_size == 128
    assert config.training.decode_batch_size == 128


def test_null_explicit_fields_fall_back_to_batch_size() -> None:
    config = load_learned_pose_token_config(
        ROOT / "configs/modeling/learned_pose_token.yaml",
        request=_request("portable"),
    )
    assert config.training.tokenizer_batch_size is None
    assert config.training.effective_tokenizer_batch_size == config.training.batch_size
    assert config.training.effective_text_to_token_batch_size == config.training.batch_size


def test_a100_active_overrides_flow_into_provider_effective_config() -> None:
    config = load_learned_pose_token_config(
        ROOT / "configs/modeling/learned_pose_token.yaml",
        request=_request(),
    )
    effective = config.to_dict()
    rehydrated = learned_pose_token_config_from_effective_dict(
        effective,
        source_path=config.source_path,
    )
    assert rehydrated.training.tokenizer_batch_size == 256
    assert rehydrated.training.cache_materialized_sources is True


def test_stage_metadata_includes_non_null_provider_batch_fields() -> None:
    config = load_learned_pose_token_config(
        ROOT / "configs/modeling/learned_pose_token.yaml",
        request=_request(),
    )
    metadata = _learned_pose_token_performance_metadata(config)
    assert metadata["tokenizer_batch_size"] == 256
    assert metadata["text_to_token_batch_size"] == 128
    assert metadata["reconstruction_batch_size"] == 128
    assert metadata["decode_batch_size"] == 128
    assert "materialization_workers" not in metadata
    assert metadata["cache_materialized_sources"] is True
    assert metadata["num_workers"] == 4
