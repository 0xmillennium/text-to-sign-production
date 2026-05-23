from __future__ import annotations

import numpy as np

from text_to_sign_production.modeling.backbones.bfh_standardization import (
    BfhStandardizationAccumulator,
    compute_bfh_standardization_stats,
)
from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    BfhVectorizedPose,
    default_bfh_tensor_layout,
)


def test_streaming_accumulator_matches_batch_stats() -> None:
    poses = _poses()
    batch = compute_bfh_standardization_stats(
        poses,
        missing_observation_policy="identity_fallback",
    )
    accumulator = BfhStandardizationAccumulator()
    for pose in poses:
        accumulator.update(pose)
    streaming = accumulator.finalize(
        epsilon=1e-6,
        missing_observation_policy="identity_fallback",
    )

    assert np.allclose(streaming.mean, batch.mean)
    assert np.allclose(streaming.std, batch.std)
    assert np.array_equal(streaming.observation_count, batch.observation_count)
    assert streaming.fallback_summary_by_channel == batch.fallback_summary_by_channel


def _poses() -> tuple[BfhVectorizedPose, ...]:
    layout = default_bfh_tensor_layout()
    shape = (2, layout.total_joint_count, layout.coordinate_dimensions)
    mask_shape = (2, layout.total_joint_count)
    first = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    second = first + 3.0
    mask = np.ones(mask_shape, dtype=np.bool_)
    mask[:, -1] = False
    confidence = mask.astype(np.float32)
    return (
        BfhVectorizedPose(
            layout=layout,
            values=first,
            validity_mask=mask,
            frame_validity_mask=np.ones((2,), dtype=np.bool_),
            confidence_values=confidence,
            frame_count=2,
            source_sample_id="s1",
        ),
        BfhVectorizedPose(
            layout=layout,
            values=second,
            validity_mask=mask,
            frame_validity_mask=np.ones((2,), dtype=np.bool_),
            confidence_values=confidence,
            frame_count=2,
            source_sample_id="s2",
        ),
    )
