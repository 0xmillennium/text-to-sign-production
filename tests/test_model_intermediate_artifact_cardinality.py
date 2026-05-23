from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_learned_pose_token_decode_does_not_publish_per_sample_token_artifacts() -> None:
    source = Path(
        "src/text_to_sign_production/modeling/candidates/learned_pose_token/provider.py"
    ).read_text(encoding="utf-8")

    assert "predicted_token_sample:val:" not in source
    assert "model_text_to_token_predicted_token_sample" not in source
    assert "predicted_token_sample_count" in source


@pytest.mark.unit
def test_latent_diffusion_generate_does_not_publish_per_sample_latent_artifacts() -> None:
    source = Path(
        "src/text_to_sign_production/modeling/candidates/latent_diffusion/provider.py"
    ).read_text(encoding="utf-8")

    assert 'role="predicted_latent_sample"' not in source
    assert "model_latent_generation_sample" not in source
    assert "predicted_latent_sample_count" in source
    assert "decoded_intermediate_sample_count" in source


@pytest.mark.unit
def test_model_publish_plan_does_not_plan_individual_generated_pose_npz_samples() -> None:
    source = Path("src/text_to_sign_production/workflows/model/publish/plan.py").read_text(
        encoding="utf-8"
    )

    assert "generated_pose/samples/*.npz" not in source
    assert "generated_pose/samples/" not in source
