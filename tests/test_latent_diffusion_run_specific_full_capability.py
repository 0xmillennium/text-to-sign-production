from __future__ import annotations

from pathlib import Path

from text_to_sign_production.modeling.candidates import ModelRunMode, ModelRunRequest
from text_to_sign_production.modeling.candidates.latent_diffusion.provider import (
    LatentDiffusionProvider,
)
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.model.contracts.compute_profiles import (
    load_model_compute_profile,
)
from text_to_sign_production.workflows.model.processing.stages import (
    _missing_required_full_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def _loaded():
    request = ModelRunRequest(
        model_key=ModelKey.LATENT_DIFFUSION,
        run_name="latent-run-specific",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_mode=ModelRunMode.FULL,
        config_path=ROOT / "configs/modeling/latent_diffusion.yaml",
        compute_profile=load_model_compute_profile(ROOT, "colab_a100_80gb").to_dict(),
    )
    return LatentDiffusionProvider().load_config(request)


def test_standardized_bfh_frame_capability_does_not_claim_autoencoder_training() -> None:
    loaded = _loaded()
    capability = LatentDiffusionProvider().full_data_pipeline_capability_for_config(loaded)

    assert loaded.effective_config["latent_autoencoder"]["active"] is False
    assert "behavior:autoencoder_training:surface_reader" not in capability.verification_evidence
    assert "behavior:latent_windows:surface_reader" not in capability.verification_evidence


def test_standardized_bfh_frame_full_evidence_does_not_require_autoencoder_training() -> None:
    loaded = _loaded()
    capability = LatentDiffusionProvider().full_data_pipeline_capability_for_config(loaded)

    missing = _missing_required_full_evidence(
        provider_key="latent_diffusion",
        observed=capability.verification_evidence,
        effective_config=loaded.effective_config,
    )

    assert "behavior:autoencoder_training:surface_reader" not in missing
    assert missing == ()
