from __future__ import annotations

import inspect

from text_to_sign_production.modeling.candidates.latent_diffusion.provider import (
    LatentDiffusionProvider,
    _generate_validation_latents,
    _train_latent_denoiser,
)


def test_latent_diffusion_full_capability_is_verified_after_streaming_migration() -> None:
    capability = LatentDiffusionProvider().full_data_pipeline_capability

    assert capability.full_training_data_mode == "streaming_sharded"
    assert capability.verified is True
    assert capability.is_full_safe
    assert capability.verification_evidence


def test_latent_diffusion_full_stages_do_not_call_eager_source_builders() -> None:
    provider = LatentDiffusionProvider()
    sources = "\n".join(
        (
            inspect.getsource(provider._cache_latents),
            inspect.getsource(_train_latent_denoiser),
            inspect.getsource(_generate_validation_latents),
        )
    )

    assert "build_latent_source_samples(" not in sources
    assert "cache_latent_sequences(" not in sources
    assert "tuple(source_samples)" not in sources
    assert "train_source_surface_path" in sources
    assert "validation_source_surface_path" in sources
