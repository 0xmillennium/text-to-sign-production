"""Canonical channel partitions and masks for vectorized full-BFH poses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    BfhTensorLayout,
    BfhVectorizedPose,
)
from text_to_sign_production.modeling.data.errors import ModelingDataError


@dataclass(frozen=True, slots=True)
class BfhChannelPartition:
    """Joint and flattened-coordinate range assigned to one BFH channel."""

    channel: PoseChannel
    joint_start: int
    joint_stop: int
    joint_count: int
    feature_start: int
    feature_stop: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel", PoseChannel(self.channel))
        for field_name in (
            "joint_start",
            "joint_stop",
            "joint_count",
            "feature_start",
            "feature_stop",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ModelingDataError(f"{field_name} must be a non-negative integer.")
        if self.joint_count <= 0 or self.joint_stop - self.joint_start != self.joint_count:
            raise ModelingDataError("BFH channel partition joint range is inconsistent.")
        if self.feature_stop <= self.feature_start:
            raise ModelingDataError("BFH channel partition feature range is empty.")

    def to_dict(self) -> dict[str, object]:
        return {
            "channel": self.channel.value,
            "joint_start": self.joint_start,
            "joint_stop": self.joint_stop,
            "joint_count": self.joint_count,
            "feature_start": self.feature_start,
            "feature_stop": self.feature_stop,
        }


def build_bfh_channel_partitions(
    layout: BfhTensorLayout,
) -> tuple[BfhChannelPartition, ...]:
    """Build non-overlapping partitions covering every canonical BFH joint."""

    _require_layout(layout)
    partitions = tuple(
        BfhChannelPartition(
            channel=channel,
            joint_start=layout.channel_slices[channel].start,
            joint_stop=layout.channel_slices[channel].stop,
            joint_count=layout.channel_slices[channel].stop - layout.channel_slices[channel].start,
            feature_start=layout.channel_slices[channel].start * layout.coordinate_dimensions,
            feature_stop=layout.channel_slices[channel].stop * layout.coordinate_dimensions,
        )
        for channel in layout.channels
    )
    covered = [index for part in partitions for index in range(part.joint_start, part.joint_stop)]
    if covered != list(range(layout.total_joint_count)):
        raise ModelingDataError("BFH channel partitions do not cover canonical joints exactly once.")
    return partitions


def channel_joint_mask(
    layout: BfhTensorLayout,
    channel: PoseChannel,
) -> np.ndarray:
    """Return the boolean joint selector for one canonical BFH channel."""

    selected = _channel(layout, channel)
    mask = np.zeros((layout.total_joint_count,), dtype=np.bool_)
    mask[layout.channel_slices[selected]] = True
    mask.setflags(write=False)
    return mask


def channel_feature_slice(
    layout: BfhTensorLayout,
    channel: PoseChannel,
) -> slice:
    """Return the flattened coordinate range for one channel."""

    selected = _channel(layout, channel)
    joints = layout.channel_slices[selected]
    return slice(
        joints.start * layout.coordinate_dimensions,
        joints.stop * layout.coordinate_dimensions,
    )


def apply_channel_mask(
    vectorized: BfhVectorizedPose,
    channel: PoseChannel,
) -> BfhVectorizedPose:
    """Retain observations for one channel while preserving full-BFH shape."""

    if not isinstance(vectorized, BfhVectorizedPose):
        raise ModelingDataError("vectorized must be a BfhVectorizedPose.")
    mask = channel_joint_mask(vectorized.layout, channel)
    values = np.where(mask[None, :, None], vectorized.values, 0.0)
    validity = vectorized.validity_mask & mask[None, :]
    confidence = np.where(mask[None, :], vectorized.confidence_values, 0.0)
    return BfhVectorizedPose(
        layout=vectorized.layout,
        values=values,
        validity_mask=validity,
        frame_validity_mask=vectorized.frame_validity_mask,
        confidence_values=confidence,
        frame_count=vectorized.frame_count,
        source_sample_id=vectorized.source_sample_id,
    )


def _require_layout(layout: BfhTensorLayout) -> None:
    if not isinstance(layout, BfhTensorLayout):
        raise ModelingDataError("layout must be a BfhTensorLayout.")


def _channel(layout: BfhTensorLayout, channel: PoseChannel) -> PoseChannel:
    _require_layout(layout)
    try:
        selected = PoseChannel(channel)
    except ValueError as exc:
        raise ModelingDataError(f"unknown BFH channel: {channel!r}.") from exc
    if selected not in layout.channel_slices:
        raise ModelingDataError(f"BFH channel is not present in layout: {selected.value!r}.")
    return selected


__all__ = [
    "BfhChannelPartition",
    "apply_channel_mask",
    "build_bfh_channel_partitions",
    "channel_feature_slice",
    "channel_joint_mask",
]
