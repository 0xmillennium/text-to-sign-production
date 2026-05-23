from __future__ import annotations

from text_to_sign_production.modeling.candidates.bootstrap import ensure_model_provider_registered
from text_to_sign_production.modeling.candidates.registry import require_model_provider
from text_to_sign_production.modeling.research import ModelKey


def test_all_providers_fail_closed_until_behaviorally_verified() -> None:
    expected = {
        ModelKey.BASE_DIRECT: "lazy_dataloader",
        ModelKey.LEARNED_POSE_TOKEN: "streaming_sharded",
        ModelKey.LATENT_DIFFUSION: "streaming_sharded",
        ModelKey.ARTICULATOR_AWARE: "streaming_sharded",
    }
    for model_key, expected_mode in expected.items():
        ensure_model_provider_registered(model_key)
        capability = require_model_provider(model_key).full_data_pipeline_capability

        assert capability.provider_key == model_key.value
        assert capability.full_training_data_mode == expected_mode
        if model_key in {
            ModelKey.BASE_DIRECT,
            ModelKey.LEARNED_POSE_TOKEN,
            ModelKey.LATENT_DIFFUSION,
            ModelKey.ARTICULATOR_AWARE,
        }:
            assert capability.verified is True
            assert capability.verification_evidence
            assert capability.is_full_safe
            assert capability.limitations == ()
        else:
            assert capability.verified is False
            assert capability.verification_evidence == ()
            assert capability.verification == ()
            assert not capability.is_full_safe
            assert capability.limitations
        assert capability.covered_stages
