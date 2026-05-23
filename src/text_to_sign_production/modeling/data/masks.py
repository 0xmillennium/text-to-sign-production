"""Mask helpers for canonical full-BFH modeling pose arrays."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.data.bfh_schema import BfhPoseArrays
from text_to_sign_production.modeling.data.errors import ModelingDataError


def valid_frame_mask(arrays: BfhPoseArrays) -> np.ndarray:
    """Return the sample-level valid-frame mask."""

    return np.asarray(arrays.valid_frame_mask, dtype=np.bool_)


def channel_observation_mask(
    arrays: BfhPoseArrays,
    channel: PoseChannel | str,
    *,
    confidence_threshold: float = 0.0,
) -> np.ndarray:
    """Return valid confidence observations for one channel, shaped (T, joints)."""

    threshold = _confidence_threshold(confidence_threshold)
    confidence = arrays.confidence(channel)
    return valid_frame_mask(arrays)[:, None] & (confidence > threshold)


def channel_frame_presence_mask(
    arrays: BfhPoseArrays,
    channel: PoseChannel | str,
    *,
    confidence_threshold: float = 0.0,
) -> np.ndarray:
    """Return frame-level channel presence for one channel, shaped (T,)."""

    return np.any(
        channel_observation_mask(
            arrays,
            channel,
            confidence_threshold=confidence_threshold,
        ),
        axis=1,
    )


def _confidence_threshold(value: float) -> float:
    threshold = float(value)
    if not np.isfinite(threshold):
        raise ModelingDataError("confidence_threshold must be finite.")
    if threshold < 0.0 or threshold > 1.0:
        raise ModelingDataError("confidence_threshold must be within [0.0, 1.0].")
    return threshold


__all__ = [
    "channel_frame_presence_mask",
    "channel_observation_mask",
    "valid_frame_mask",
]
