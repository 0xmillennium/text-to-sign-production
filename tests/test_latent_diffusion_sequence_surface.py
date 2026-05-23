from __future__ import annotations

from text_to_sign_production.modeling.candidates.latent_diffusion.dataset import (
    build_latent_sequence_surface_from_source_surface,
)


def test_latent_sequence_surface_builder_is_available() -> None:
    assert (
        build_latent_sequence_surface_from_source_surface.__name__
        == "build_latent_sequence_surface_from_source_surface"
    )
