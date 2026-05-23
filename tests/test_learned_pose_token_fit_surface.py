from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fit_representation_uses_pose_token_surfaces() -> None:
    source = (
        ROOT
        / "src/text_to_sign_production/modeling/candidates/learned_pose_token/provider.py"
    ).read_text(encoding="utf-8")

    assert "_materialize_pose_token_surface" in source
    assert "train_surface=train_surface" in source
    assert "validation_surface=validation_surface" in source
