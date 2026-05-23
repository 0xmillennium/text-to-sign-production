from __future__ import annotations

from text_to_sign_production.modeling.candidates.articulator_aware.dataset import (
    build_articulator_source_surface,
)


def test_articulator_source_surface_builder_is_available() -> None:
    assert build_articulator_source_surface.__name__ == "build_articulator_source_surface"
