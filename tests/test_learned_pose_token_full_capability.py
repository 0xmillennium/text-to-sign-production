from __future__ import annotations

import inspect

from text_to_sign_production.modeling.candidates.learned_pose_token.provider import (
    LearnedPoseTokenProvider,
    _decode_validation_text,
)


def test_learned_pose_token_full_capability_is_verified_after_decode_export_streaming() -> None:
    capability = LearnedPoseTokenProvider().full_data_pipeline_capability

    assert capability.full_training_data_mode == "streaming_sharded"
    assert capability.verified is True
    assert capability.is_full_safe
    assert capability.verification_evidence


def test_learned_pose_token_full_stages_do_not_route_through_eager_materializer() -> None:
    provider = LearnedPoseTokenProvider()
    sources = "\n".join(
        (
            inspect.getsource(provider._fit_representation),
            inspect.getsource(provider._evaluate_reconstruction),
            inspect.getsource(provider._train_text_to_token),
            inspect.getsource(provider._decode_to_pose),
            inspect.getsource(_decode_validation_text),
        )
    )

    assert "_materialize_pose_token_samples(" not in sources
    assert "build_pose_token_training_samples(" not in sources
    assert " _sources_by_id(" not in sources
    assert "_read_sequences(" not in sources
    assert "_sequence_batches(" not in sources
    assert "ModelDataSurfaceReader" in sources
