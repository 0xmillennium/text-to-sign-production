"""Canonical full-BFH vector layout and reversible array conversion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from text_to_sign_production.data.gate.pose.schema import POSE_COORDINATE_DIMENSIONS
from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.data.bfh_schema import (
    BFH_CHANNEL_SPECS,
    FULL_BFH_CHANNELS,
    BfhPoseArrays,
)
from text_to_sign_production.modeling.data.errors import ModelingDataError

BFH_VECTORIZATION_SCHEMA_VERSION = "t2sp-bfh-vectorization-v1"


@dataclass(frozen=True, slots=True)
class BfhTensorLayout:
    """Canonical ordered full-BFH joint and flattened-feature layout."""

    schema_version: str
    channels: tuple[PoseChannel, ...]
    coordinate_dimensions: int
    channel_slices: Mapping[PoseChannel, slice]
    total_joint_count: int
    total_feature_dim: int

    def __post_init__(self) -> None:
        if self.schema_version != BFH_VECTORIZATION_SCHEMA_VERSION:
            raise ModelingDataError("BFH tensor layout schema_version is unsupported.")
        channels = tuple(PoseChannel(channel) for channel in self.channels)
        if channels != tuple(FULL_BFH_CHANNELS):
            raise ModelingDataError(
                "BFH tensor layout channels must match canonical full-BFH channel order."
            )
        if self.coordinate_dimensions != POSE_COORDINATE_DIMENSIONS:
            raise ModelingDataError(
                "BFH tensor layout coordinate_dimensions must match the BFH schema."
            )
        expected_slices: dict[PoseChannel, slice] = {}
        joint_start = 0
        for channel in channels:
            joint_stop = joint_start + BFH_CHANNEL_SPECS[channel].joint_count
            expected_slices[channel] = slice(joint_start, joint_stop)
            joint_start = joint_stop
        if set(self.channel_slices) != set(channels):
            raise ModelingDataError("BFH tensor layout must include one slice for each channel.")
        for channel, expected_slice in expected_slices.items():
            observed = self.channel_slices[channel]
            if not isinstance(observed, slice) or observed != expected_slice:
                raise ModelingDataError(
                    f"BFH tensor layout slice for {channel.value!r} is not canonical."
                )
        if self.total_joint_count != joint_start:
            raise ModelingDataError("BFH tensor layout total_joint_count is inconsistent.")
        if self.total_feature_dim != joint_start * self.coordinate_dimensions:
            raise ModelingDataError("BFH tensor layout total_feature_dim is inconsistent.")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "channel_slices", MappingProxyType(expected_slices))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable layout record."""

        return {
            "schema_version": self.schema_version,
            "channels": [channel.value for channel in self.channels],
            "coordinate_dimensions": self.coordinate_dimensions,
            "channel_slices": {
                channel.value: {
                    "start": self.channel_slices[channel].start,
                    "stop": self.channel_slices[channel].stop,
                }
                for channel in self.channels
            },
            "total_joint_count": self.total_joint_count,
            "total_feature_dim": self.total_feature_dim,
        }


@dataclass(frozen=True, slots=True)
class BfhVectorizedPose:
    """Coordinates and observation truth in the canonical joint layout."""

    layout: BfhTensorLayout
    values: np.ndarray
    validity_mask: np.ndarray
    frame_validity_mask: np.ndarray
    confidence_values: np.ndarray
    frame_count: int
    source_sample_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.layout, BfhTensorLayout):
            raise ModelingDataError("layout must be a BfhTensorLayout.")
        if not isinstance(self.frame_count, int) or isinstance(self.frame_count, bool):
            raise ModelingDataError("frame_count must be an integer.")
        if self.frame_count < 1:
            raise ModelingDataError("frame_count must be at least 1.")
        expected_values = (
            self.frame_count,
            self.layout.total_joint_count,
            self.layout.coordinate_dimensions,
        )
        values = _numeric_array(self.values, "values", expected_values)
        validity_mask = _bool_array(
            self.validity_mask,
            "validity_mask",
            (self.frame_count, self.layout.total_joint_count),
        )
        frame_validity_mask = _bool_array(
            self.frame_validity_mask,
            "frame_validity_mask",
            (self.frame_count,),
        )
        confidence_values = _numeric_array(
            self.confidence_values,
            "confidence_values",
            (self.frame_count, self.layout.total_joint_count),
        )
        if np.any((confidence_values < 0.0) | (confidence_values > 1.0)):
            raise ModelingDataError("confidence_values must be within [0.0, 1.0].")
        expected_validity = frame_validity_mask[:, None] & (confidence_values > 0.0)
        if not np.array_equal(validity_mask, expected_validity):
            raise ModelingDataError(
                "validity_mask must exactly match valid frames with positive confidence_values."
            )
        if not np.all(np.isfinite(values[validity_mask])):
            raise ModelingDataError("values must be finite at valid BFH observations.")
        if self.source_sample_id is not None:
            _require_text(self.source_sample_id, "source_sample_id")
        object.__setattr__(self, "values", _read_only(values))
        object.__setattr__(self, "validity_mask", _read_only(validity_mask))
        object.__setattr__(self, "frame_validity_mask", _read_only(frame_validity_mask))
        object.__setattr__(self, "confidence_values", _read_only(confidence_values))


def default_bfh_tensor_layout() -> BfhTensorLayout:
    """Build the canonical full-BFH concatenated-joint layout."""

    channel_slices: dict[PoseChannel, slice] = {}
    start = 0
    for channel in FULL_BFH_CHANNELS:
        stop = start + BFH_CHANNEL_SPECS[channel].joint_count
        channel_slices[channel] = slice(start, stop)
        start = stop
    return BfhTensorLayout(
        schema_version=BFH_VECTORIZATION_SCHEMA_VERSION,
        channels=tuple(FULL_BFH_CHANNELS),
        coordinate_dimensions=POSE_COORDINATE_DIMENSIONS,
        channel_slices=channel_slices,
        total_joint_count=start,
        total_feature_dim=start * POSE_COORDINATE_DIMENSIONS,
    )


def bfh_tensor_layout_from_dict(record: Mapping[str, object]) -> BfhTensorLayout:
    """Parse a strictly shaped JSON layout record."""

    if not isinstance(record, Mapping):
        raise ModelingDataError("BFH tensor layout record must be a JSON object.")
    expected_keys = {
        "schema_version",
        "channels",
        "coordinate_dimensions",
        "channel_slices",
        "total_joint_count",
        "total_feature_dim",
    }
    if set(record) != expected_keys:
        raise ModelingDataError("BFH tensor layout record keys do not match the schema.")
    raw_channels = record["channels"]
    raw_slices = record["channel_slices"]
    if not isinstance(raw_channels, list) or not isinstance(raw_slices, Mapping):
        raise ModelingDataError("BFH tensor layout channels and channel_slices are invalid.")
    channels = tuple(PoseChannel(_text(value, "channel")) for value in raw_channels)
    channel_slices: dict[PoseChannel, slice] = {}
    for channel in channels:
        raw_slice = raw_slices.get(channel.value)
        if not isinstance(raw_slice, Mapping) or set(raw_slice) != {"start", "stop"}:
            raise ModelingDataError(f"BFH tensor layout slice is invalid for {channel.value!r}.")
        channel_slices[channel] = slice(
            _integer(raw_slice["start"], "slice start"),
            _integer(raw_slice["stop"], "slice stop"),
        )
    return BfhTensorLayout(
        schema_version=_text(record["schema_version"], "schema_version"),
        channels=channels,
        coordinate_dimensions=_integer(record["coordinate_dimensions"], "coordinate_dimensions"),
        channel_slices=channel_slices,
        total_joint_count=_integer(record["total_joint_count"], "total_joint_count"),
        total_feature_dim=_integer(record["total_feature_dim"], "total_feature_dim"),
    )


def vectorize_bfh_pose_arrays(
    arrays: BfhPoseArrays,
    *,
    sample_id: str | None = None,
) -> BfhVectorizedPose:
    """Concatenate already-loaded BFH channel coordinates without losing confidence."""

    if not isinstance(arrays, BfhPoseArrays):
        raise ModelingDataError("arrays must be a validated BfhPoseArrays instance.")
    layout = default_bfh_tensor_layout()
    values = np.concatenate(
        tuple(arrays.coordinates(channel) for channel in layout.channels),
        axis=1,
    )
    confidence = np.concatenate(
        tuple(arrays.confidence(channel) for channel in layout.channels),
        axis=1,
    )
    validity = np.asarray(arrays.valid_frame_mask, dtype=np.bool_)[:, None] & (confidence > 0.0)
    return BfhVectorizedPose(
        layout=layout,
        values=values,
        validity_mask=validity,
        frame_validity_mask=arrays.valid_frame_mask,
        confidence_values=confidence,
        frame_count=arrays.frame_count,
        source_sample_id=sample_id,
    )


def bfh_pose_arrays_from_vectorized(vectorized: BfhVectorizedPose) -> BfhPoseArrays:
    """Restore channel XYC arrays from a canonical vectorized pose."""

    _require_vectorized(vectorized)
    arrays: dict[PoseChannel, np.ndarray] = {}
    for channel in vectorized.layout.channels:
        joint_slice = vectorized.layout.channel_slices[channel]
        arrays[channel] = np.concatenate(
            (
                np.asarray(vectorized.values[:, joint_slice], dtype=np.float32),
                np.asarray(vectorized.confidence_values[:, joint_slice, None], dtype=np.float32),
            ),
            axis=-1,
        )
    return BfhPoseArrays(
        body_xyc=arrays[PoseChannel.BODY],
        left_hand_xyc=arrays[PoseChannel.LEFT_HAND],
        right_hand_xyc=arrays[PoseChannel.RIGHT_HAND],
        face_xyc=arrays[PoseChannel.FACE],
        valid_frame_mask=vectorized.frame_validity_mask,
    )


def flatten_bfh_vectorized_pose(vectorized: BfhVectorizedPose) -> np.ndarray:
    """Return canonical per-frame flattened BFH coordinates."""

    _require_vectorized(vectorized)
    return np.asarray(vectorized.values, dtype=np.float32).reshape(
        vectorized.frame_count,
        vectorized.layout.total_feature_dim,
    )


def unflatten_bfh_pose_values(
    values: np.ndarray,
    *,
    layout: BfhTensorLayout,
    frame_count: int,
) -> np.ndarray:
    """Restore `(frames, joints, coordinates)` values from flattened coordinates."""

    if not isinstance(layout, BfhTensorLayout):
        raise ModelingDataError("layout must be a BfhTensorLayout.")
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count < 1:
        raise ModelingDataError("frame_count must be a positive integer.")
    flat = _numeric_array(values, "values", (frame_count, layout.total_feature_dim))
    return flat.reshape(frame_count, layout.total_joint_count, layout.coordinate_dimensions)


def _require_vectorized(value: BfhVectorizedPose) -> None:
    if not isinstance(value, BfhVectorizedPose):
        raise ModelingDataError("pose must be a BfhVectorizedPose.")


def _numeric_array(value: np.ndarray, field_name: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape:
        raise ModelingDataError(f"{field_name} must have shape {shape}; got {raw.shape}.")
    if raw.dtype.kind not in {"f", "i", "u", "b"}:
        raise ModelingDataError(f"{field_name} must be numeric.")
    return np.asarray(raw, dtype=np.float32).copy()


def _bool_array(value: np.ndarray, field_name: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind != "b":
        raise ModelingDataError(f"{field_name} must be boolean with shape {shape}.")
    return np.asarray(raw, dtype=np.bool_).copy()


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _require_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ModelingDataError(f"{field_name} must be non-empty.")


def _text(value: object, field_name: str) -> str:
    _require_text(value, field_name)
    return str(value)


def _integer(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ModelingDataError(f"{field_name} must be an integer.")
    return value


__all__ = [
    "BFH_VECTORIZATION_SCHEMA_VERSION",
    "BfhTensorLayout",
    "BfhVectorizedPose",
    "bfh_pose_arrays_from_vectorized",
    "bfh_tensor_layout_from_dict",
    "default_bfh_tensor_layout",
    "flatten_bfh_vectorized_pose",
    "unflatten_bfh_pose_values",
    "vectorize_bfh_pose_arrays",
]
