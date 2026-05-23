from __future__ import annotations

import inspect

from text_to_sign_production.modeling.candidates.articulator_aware.provider import (
    ArticulatorAwareProvider,
)


def test_articulator_aware_full_capability_is_verified_after_surface_trainer_migration() -> None:
    capability = ArticulatorAwareProvider().full_data_pipeline_capability

    assert capability.full_training_data_mode == "streaming_sharded"
    assert capability.verified is True
    assert capability.is_full_safe
    assert capability.verification_evidence


def test_articulator_aware_provider_full_stages_use_surface_entrypoints() -> None:
    provider = ArticulatorAwareProvider()
    sources = "\n".join(
        (
            inspect.getsource(provider._train),
            inspect.getsource(provider._export),
        )
    )

    assert "build_articulator_source_samples(" not in sources
    assert "train_articulator_model(" not in sources
    assert "train_articulator_model_from_surfaces(" in sources
    assert "build_articulator_frame_surface_from_source_surface" in inspect.getsource(
        __import__(
            "text_to_sign_production.modeling.candidates.articulator_aware.provider",
            fromlist=["_materialize_articulator_frame_surface"],
        )._materialize_articulator_frame_surface
    )
