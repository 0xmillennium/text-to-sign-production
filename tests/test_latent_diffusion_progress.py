from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_latent_diffusion_provider_declares_expected_progress_tasks() -> None:
    provider = Path(
        "src/text_to_sign_production/modeling/candidates/latent_diffusion/provider.py"
    ).read_text(encoding="utf-8")
    autoencoder = Path(
        "src/text_to_sign_production/modeling/candidates/latent_diffusion/autoencoder.py"
    ).read_text(encoding="utf-8")
    source = provider + autoencoder

    for operation in (
        "materialize_train",
        "materialize_val",
        "cache_train_latents",
        "cache_validation_latents",
        "train_denoiser_epoch_",
        "validate_denoiser",
        "generate_validation_candidates",
        "predicted_latent_manifest_val",
        "write_final_generated_pose",
        "autoencoder_train_epoch_",
        "autoencoder_validation",
        "build_train_windows",
        "build_validation_windows",
    ):
        assert operation in source
