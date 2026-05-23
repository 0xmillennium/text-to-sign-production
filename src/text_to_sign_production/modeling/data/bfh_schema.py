"""Canonical full-BFH pose-array contract for modeling surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from text_to_sign_production.core.models import PreparedSample
from text_to_sign_production.data.gate.pose.schema import (
    CANONICAL_POSE_CHANNELS,
    POSE_CHANNEL_JOINT_COUNTS,
    POSE_COORDINATE_DIMENSIONS,
)
from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.data.errors import ModelingDataError

FULL_BFH_CHANNEL_POLICY = "full_bfh"
FULL_BFH_CHANNELS = (
    CANONICAL_POSE_CHANNELS
    if CANONICAL_POSE_CHANNELS
    == (
        PoseChannel.BODY,
        PoseChannel.LEFT_HAND,
        PoseChannel.RIGHT_HAND,
        PoseChannel.FACE,
    )
    else (
        PoseChannel.BODY,
        PoseChannel.LEFT_HAND,
        PoseChannel.RIGHT_HAND,
        PoseChannel.FACE,
    )
)


@dataclass(frozen=True, slots=True)
class BfhChannelSpec:
    """Shape contract for one full-BFH pose channel."""

    channel: PoseChannel
    joint_count: int
    coordinate_dimensions: int = 2
    xyc_dimensions: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel", coerce_pose_channel(self.channel))
        if self.joint_count <= 0:
            raise ModelingDataError("joint_count must be positive.")
        if self.coordinate_dimensions != POSE_COORDINATE_DIMENSIONS:
            raise ModelingDataError("coordinate_dimensions must be 2.")
        if self.xyc_dimensions != 3:
            raise ModelingDataError("xyc_dimensions must be 3.")


def coerce_pose_channel(channel: PoseChannel | str) -> PoseChannel:
    """Coerce a pose channel string or enum value."""

    try:
        return PoseChannel(channel)
    except ValueError as exc:
        raise ModelingDataError(f"unknown pose channel: {channel!r}") from exc


BFH_CHANNEL_SPECS: Mapping[PoseChannel, BfhChannelSpec] = MappingProxyType(
    {
        channel: BfhChannelSpec(
            channel=channel,
            joint_count=POSE_CHANNEL_JOINT_COUNTS[channel],
        )
        for channel in FULL_BFH_CHANNELS
    }
)


@dataclass(frozen=True, slots=True)
class BfhPoseArrays:
    """Validated full-BFH x/y/confidence arrays for modeling."""

    body_xyc: np.ndarray
    left_hand_xyc: np.ndarray
    right_hand_xyc: np.ndarray
    face_xyc: np.ndarray
    valid_frame_mask: np.ndarray

    def __post_init__(self) -> None:
        mask = np.asarray(self.valid_frame_mask, dtype=np.bool_).copy()
        if mask.ndim != 1:
            raise ModelingDataError("valid_frame_mask must be 1D.")
        frame_count = int(mask.shape[0])
        if frame_count <= 0:
            raise ModelingDataError("frame_count must be positive.")
        object.__setattr__(self, "valid_frame_mask", mask)

        for field_name, channel in _FIELD_CHANNELS:
            value = _coerce_channel_array(
                getattr(self, field_name),
                channel=channel,
                frame_count=frame_count,
                field_name=field_name,
            )
            object.__setattr__(self, field_name, value)

    @property
    def frame_count(self) -> int:
        """Number of frames in the pose arrays."""

        return int(self.valid_frame_mask.shape[0])

    @property
    def channels(self) -> Mapping[PoseChannel, np.ndarray]:
        """Return channel arrays keyed by canonical pose channel."""

        return MappingProxyType(
            {
                PoseChannel.BODY: self.body_xyc,
                PoseChannel.LEFT_HAND: self.left_hand_xyc,
                PoseChannel.RIGHT_HAND: self.right_hand_xyc,
                PoseChannel.FACE: self.face_xyc,
            }
        )

    def channel_xyc(self, channel: PoseChannel | str) -> np.ndarray:
        """Return the x/y/confidence array for a channel."""

        resolved = coerce_pose_channel(channel)
        return self.channels[resolved]

    def coordinates(self, channel: PoseChannel | str) -> np.ndarray:
        """Return x/y coordinates for a channel."""

        return channel_coordinates(self.channel_xyc(channel))

    def confidence(self, channel: PoseChannel | str) -> np.ndarray:
        """Return confidence values for a channel."""

        return channel_confidence(self.channel_xyc(channel))


def channel_coordinates(xyc: np.ndarray) -> np.ndarray:
    """Return x/y coordinate slice from an x/y/confidence tensor."""

    return np.asarray(xyc)[..., :2]


def channel_confidence(xyc: np.ndarray) -> np.ndarray:
    """Return confidence slice from an x/y/confidence tensor."""

    return np.asarray(xyc)[..., 2]


def pose_arrays_from_prepared_sample(sample: PreparedSample) -> BfhPoseArrays:
    """Adapt a core PreparedSample pose into the modeling BFH contract."""

    return BfhPoseArrays(
        body_xyc=sample.pose.body_xyc,
        left_hand_xyc=sample.pose.left_hand_xyc,
        right_hand_xyc=sample.pose.right_hand_xyc,
        face_xyc=sample.pose.face_xyc,
        valid_frame_mask=sample.pose.valid_frame_mask,
    )


def validate_bfh_pose_arrays(arrays: BfhPoseArrays) -> tuple[str, ...]:
    """Return BFH pose-array validation issues without raising."""

    try:
        BfhPoseArrays(
            body_xyc=arrays.body_xyc,
            left_hand_xyc=arrays.left_hand_xyc,
            right_hand_xyc=arrays.right_hand_xyc,
            face_xyc=arrays.face_xyc,
            valid_frame_mask=arrays.valid_frame_mask,
        )
    except (AttributeError, IndexError, ModelingDataError, TypeError, ValueError) as exc:
        return (str(exc),)
    return ()


def _coerce_channel_array(
    value: np.ndarray,
    *,
    channel: PoseChannel,
    frame_count: int,
    field_name: str,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"f", "i", "u", "b"}:
        raise ModelingDataError(f"{field_name} must be numeric.")
    array = np.asarray(raw, dtype=np.float32).copy()
    expected = (frame_count, BFH_CHANNEL_SPECS[channel].joint_count, 3)
    if array.shape != expected:
        raise ModelingDataError(f"{field_name} must have shape {expected}.")
    if not np.all(np.isfinite(array)):
        raise ModelingDataError(f"{field_name} must contain only finite values.")
    confidence = channel_confidence(array)
    bad_mask = (confidence < 0.0) | (confidence > 1.0)
    if np.any(bad_mask):
        bad_count = int(np.count_nonzero(bad_mask))
        raise ModelingDataError(
            f"{field_name} confidence must be within [0.0, 1.0]; "
            f"bad_count={bad_count}, "
            f"min={float(np.nanmin(confidence)):.6g}, "
            f"max={float(np.nanmax(confidence)):.6g}."
        )
    return array


_FIELD_CHANNELS = (
    ("body_xyc", PoseChannel.BODY),
    ("left_hand_xyc", PoseChannel.LEFT_HAND),
    ("right_hand_xyc", PoseChannel.RIGHT_HAND),
    ("face_xyc", PoseChannel.FACE),
)


__all__ = [
    "BFH_CHANNEL_SPECS",
    "FULL_BFH_CHANNEL_POLICY",
    "FULL_BFH_CHANNELS",
    "BfhChannelSpec",
    "BfhPoseArrays",
    "channel_confidence",
    "channel_coordinates",
    "coerce_pose_channel",
    "pose_arrays_from_prepared_sample",
    "validate_bfh_pose_arrays",
]
