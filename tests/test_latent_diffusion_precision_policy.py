from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_latent_diffusion_training_paths_use_shared_autocast_context() -> None:
    provider_source = (
        ROOT
        / "src/text_to_sign_production/modeling/candidates/latent_diffusion/provider.py"
    ).read_text(encoding="utf-8")
    autoencoder_source = (
        ROOT
        / "src/text_to_sign_production/modeling/candidates/latent_diffusion/autoencoder.py"
    ).read_text(encoding="utf-8")

    assert "with autocast_context(resolved_precision):" in provider_source
    assert "precision_policy=precision_policy" in provider_source
    assert "generation_precision_applied" in provider_source
    assert "with autocast_context(resolved_precision):" in autoencoder_source


def test_latent_diffusion_precision_metadata_reports_actual_application() -> None:
    provider_source = (
        ROOT
        / "src/text_to_sign_production/modeling/candidates/latent_diffusion/provider.py"
    ).read_text(encoding="utf-8")

    assert "resolved.to_metadata(precision_applied=applied)" in provider_source
    assert "bool(precision_applied) and resolved.autocast_enabled" in provider_source
