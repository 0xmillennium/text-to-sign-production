from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_full_standardization_builders_use_streaming_accumulator() -> None:
    learned = (
        ROOT
        / "src/text_to_sign_production/modeling/candidates/learned_pose_token/dataset.py"
    ).read_text(encoding="utf-8")
    latent = (
        ROOT
        / "src/text_to_sign_production/modeling/candidates/latent_diffusion/dataset.py"
    ).read_text(encoding="utf-8")

    assert "BfhStandardizationAccumulator" in learned
    assert "BfhStandardizationAccumulator" in latent
    assert "build_latent_source_surface" in latent
