from pathlib import Path

from text_to_sign_production.modeling.candidates import ModelRunMode
from text_to_sign_production.modeling.candidates.latent_diffusion.provider import (
    _effective_sampling_steps,
)
from text_to_sign_production.workflows.model.contracts.compute_profiles import (
    load_model_compute_profile,
)


ROOT = Path(__file__).resolve().parents[1]


def _effective(profile_name: str, run_mode: str):
    return {
        "run_mode": run_mode,
        "compute_profile": load_model_compute_profile(ROOT, profile_name).to_dict(),
    }


def test_portable_full_uses_config_sampling_steps() -> None:
    assert _effective_sampling_steps(
        _effective("portable", ModelRunMode.FULL.value),
        configured_steps=100,
    ) == (100, "config")


def test_colab_a100_smoke_uses_profile_override() -> None:
    assert _effective_sampling_steps(
        _effective("colab_a100_80gb", ModelRunMode.SMOKE.value),
        configured_steps=100,
    ) == (10, "compute_profile_smoke_override")


def test_colab_a100_debug_uses_profile_override() -> None:
    assert _effective_sampling_steps(
        _effective("colab_a100_80gb", ModelRunMode.DEBUG.value),
        configured_steps=100,
    ) == (25, "compute_profile_debug_override")
