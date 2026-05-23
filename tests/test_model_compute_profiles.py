import json
from pathlib import Path

import pytest

from text_to_sign_production.workflows.model.contracts.compute_profiles import (
    apply_torch_runtime_settings,
    load_model_compute_profile,
)


ROOT = Path(__file__).resolve().parents[1]


def test_portable_profile_loads() -> None:
    profile = load_model_compute_profile(ROOT, "portable")
    assert profile.name == "portable"
    assert profile.precision.policy == "auto"


def test_colab_a100_profile_loads() -> None:
    profile = load_model_compute_profile(ROOT, "colab_a100_80gb")
    assert profile.name == "colab_a100_80gb"
    assert profile.provider_overrides["latent_diffusion"]["smoke_sampling_steps"] == 10


def test_unknown_profile_fails_clearly() -> None:
    with pytest.raises(ValueError, match="unknown model compute profile"):
        load_model_compute_profile(ROOT, "missing_profile")


def test_torch_settings_application_is_cpu_guarded() -> None:
    profile = load_model_compute_profile(ROOT, "portable")
    apply_torch_runtime_settings(profile)


def test_model_notebook_defaults_to_portable_compute_profile() -> None:
    notebook = json.loads((ROOT / "notebooks/model.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
        if "MODEL_COMPUTE_PROFILE" in line or "Colab A100" in line
    )

    assert 'MODEL_COMPUTE_PROFILE = "portable"' in source
    assert "# For Colab A100 80GB runs:" in source
    assert '# MODEL_COMPUTE_PROFILE = "colab_a100_80gb"' in source
