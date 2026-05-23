from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_full_guard_is_capability_based_not_provider_allowlist() -> None:
    source = (
        ROOT / "src/text_to_sign_production/workflows/model/processing/stages.py"
    ).read_text(encoding="utf-8")

    assert "sharded_providers" not in source
    assert "full_data_pipeline_capability" in source


def test_provider_modules_expose_streaming_surface_contract_names() -> None:
    required = {
        "learned_pose_token": (
            "ModelDataSurface",
            "build_pose_token_training_surface",
            "ModelDataSurfaceReader",
        ),
        "latent_diffusion": (
            "build_latent_source_surface",
            "build_latent_sequence_surface_from_source_surface",
        ),
        "articulator_aware": (
            "build_articulator_source_surface",
            "build_articulator_frame_surface_from_source_surface",
        ),
        "base_direct": ("lazy_dataloader",),
    }
    for provider, needles in required.items():
        if provider == "learned_pose_token":
            source = "\n".join((
                (
                    ROOT
                    / "src/text_to_sign_production/modeling/candidates/learned_pose_token/provider.py"
                ).read_text(encoding="utf-8"),
                (
                    ROOT
                    / "src/text_to_sign_production/modeling/candidates/learned_pose_token/trainer.py"
                ).read_text(encoding="utf-8"),
            ))
        else:
            path = (
                ROOT
                / "src/text_to_sign_production/modeling/candidates"
                / provider
                / ("dataset.py" if provider in {"latent_diffusion", "articulator_aware"} else "provider.py")
            )
            source = path.read_text(encoding="utf-8")
        for needle in needles:
            assert needle in source
