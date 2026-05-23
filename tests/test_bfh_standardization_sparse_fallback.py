import numpy as np
import pytest

from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.backbones.bfh_standardization import (
    BFH_STANDARDIZATION_SCHEMA_VERSION,
    compute_bfh_standardization_stats,
    read_bfh_standardization_stats_json,
    write_bfh_standardization_stats_json,
)
from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    BfhVectorizedPose,
    default_bfh_tensor_layout,
)
from text_to_sign_production.modeling.data.errors import ModelingDataError


def _pose(observed_channels: tuple[PoseChannel, ...]) -> BfhVectorizedPose:
    layout = default_bfh_tensor_layout()
    values = np.zeros((2, layout.total_joint_count, layout.coordinate_dimensions), dtype=np.float32)
    confidence = np.zeros((2, layout.total_joint_count), dtype=np.float32)
    for frame in range(2):
        values[frame, :, 0] = np.arange(layout.total_joint_count, dtype=np.float32) + frame
        values[frame, :, 1] = np.arange(layout.total_joint_count, dtype=np.float32) * 2.0 + frame
    for channel in observed_channels:
        confidence[:, layout.channel_slices[channel]] = 1.0
    frame_mask = np.ones((2,), dtype=np.bool_)
    validity = frame_mask[:, None] & (confidence > 0.0)
    return BfhVectorizedPose(
        layout=layout,
        values=values,
        validity_mask=validity,
        frame_validity_mask=frame_mask,
        confidence_values=confidence,
        frame_count=2,
        source_sample_id="sample",
    )


def test_raise_policy_preserves_strict_failure() -> None:
    with pytest.raises(ModelingDataError, match="zero valid observations"):
        compute_bfh_standardization_stats(
            [_pose((PoseChannel.BODY,))],
            missing_observation_policy="raise",
        )


def test_channel_fallback_handles_missing_coordinates_when_channel_observed() -> None:
    pose = _pose(tuple(PoseChannel))
    layout = pose.layout
    confidence = np.array(pose.confidence_values, copy=True)
    missing_joint = layout.channel_slices[PoseChannel.LEFT_HAND].start
    confidence[:, missing_joint] = 0.0
    sparse = BfhVectorizedPose(
        layout=layout,
        values=pose.values,
        validity_mask=pose.frame_validity_mask[:, None] & (confidence > 0.0),
        frame_validity_mask=pose.frame_validity_mask,
        confidence_values=confidence,
        frame_count=pose.frame_count,
        source_sample_id="sparse",
    )
    stats = compute_bfh_standardization_stats(
        [sparse],
        missing_observation_policy="channel_fallback",
    )
    assert stats.zero_observation_coordinate_count == 2
    assert stats.channel_fallback_coordinate_count == 2
    assert stats.global_fallback_coordinate_count == 0
    assert stats.fallback_summary_by_channel["left_hand"]["channel_fallback"] == 2


def test_global_fallback_handles_missing_channel_when_global_observed() -> None:
    stats = compute_bfh_standardization_stats(
        [_pose((PoseChannel.BODY,))],
        missing_observation_policy="global_fallback",
    )
    assert stats.zero_observation_coordinate_count > 0
    assert stats.global_fallback_coordinate_count == stats.zero_observation_coordinate_count
    assert stats.identity_fallback_coordinate_count == 0


def test_completely_unobserved_input_requires_explicit_identity_fallback() -> None:
    pose = _pose(())
    with pytest.raises(ModelingDataError, match="no valid coordinate observations"):
        compute_bfh_standardization_stats(
            [pose],
            missing_observation_policy="global_fallback",
        )
    stats = compute_bfh_standardization_stats(
        [pose],
        missing_observation_policy="identity_fallback",
    )
    assert np.all(stats.mean == 0.0)
    assert np.all(stats.std == 1.0)
    assert stats.identity_fallback_coordinate_count == stats.zero_observation_coordinate_count


def test_json_round_trips_v2_diagnostics(tmp_path) -> None:
    stats = compute_bfh_standardization_stats(
        [_pose((PoseChannel.BODY,))],
        missing_observation_policy="global_fallback",
    )
    path = tmp_path / "stats.json"
    write_bfh_standardization_stats_json(path, stats)
    restored = read_bfh_standardization_stats_json(path)
    assert restored.schema_version == BFH_STANDARDIZATION_SCHEMA_VERSION
    assert np.array_equal(restored.observation_count, stats.observation_count)
    assert np.array_equal(restored.mean, stats.mean)
    assert np.array_equal(restored.std, stats.std)
    assert restored.fallback_summary_by_channel == stats.fallback_summary_by_channel
