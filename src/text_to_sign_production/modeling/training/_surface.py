"""Internal channel-surface validation shared by Sprint 3 numeric utilities."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import torch

from text_to_sign_production.modeling.data import (
    SPRINT3_TARGET_CHANNEL_SHAPES,
    SPRINT3_TARGET_CHANNELS,
)


def iter_validated_pose_channels(
    *,
    predictions: object,
    targets: object,
    mask_shape: tuple[int, int],
    mask_device: torch.device,
) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
    """Yield validated default Sprint 3 prediction/target channel pairs."""

    for channel in SPRINT3_TARGET_CHANNELS:
        prediction = _extract_channel_tensor(
            predictions,
            channel=channel,
            surface_name="predictions",
        )
        target = _extract_channel_tensor(targets, channel=channel, surface_name="targets")
        _validate_pose_tensor(
            prediction,
            channel=channel,
            surface_name="predictions",
            mask_shape=mask_shape,
            mask_device=mask_device,
        )
        _validate_pose_tensor(
            target,
            channel=channel,
            surface_name="targets",
            mask_shape=mask_shape,
            mask_device=mask_device,
        )
        if prediction.shape != target.shape:
            raise ValueError(
                f"predictions and targets channel {channel!r} must have matching shapes."
            )
        yield channel, prediction, target


def valid_frame_count(effective_frame_mask: torch.Tensor) -> torch.Tensor:
    """Return the scalar tensor count of valid frame positions."""

    return effective_frame_mask.sum()


def floating_point_accumulation_dtype(value: torch.Tensor) -> torch.dtype:
    """Return a stable accumulation dtype for Sprint 3 scalar reductions."""

    if value.dtype == torch.float64:
        return torch.float64
    return torch.float32


def _extract_channel_tensor(surface: object, *, channel: str, surface_name: str) -> torch.Tensor:
    value: Any
    if isinstance(surface, Mapping):
        if channel not in surface:
            raise ValueError(f"{surface_name} is missing required channel {channel!r}.")
        value = surface[channel]
    else:
        if not hasattr(surface, channel):
            raise ValueError(f"{surface_name} is missing required channel {channel!r}.")
        value = getattr(surface, channel)

    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{surface_name} channel {channel!r} must be a torch.Tensor.")
    return value


def _validate_pose_tensor(
    tensor: torch.Tensor,
    *,
    channel: str,
    surface_name: str,
    mask_shape: tuple[int, int],
    mask_device: torch.device,
) -> None:
    if not torch.is_floating_point(tensor):
        raise ValueError(f"{surface_name} channel {channel!r} must use a floating dtype.")
    if tensor.ndim != 4:
        raise ValueError(f"{surface_name} channel {channel!r} must have shape [B, T, K, 2].")
    if tuple(tensor.shape[:2]) != mask_shape:
        raise ValueError(
            f"{surface_name} channel {channel!r} batch/time shape must match the masks."
        )
    expected_shape = SPRINT3_TARGET_CHANNEL_SHAPES[channel]
    if tuple(tensor.shape[2:]) != expected_shape:
        raise ValueError(
            f"{surface_name} channel {channel!r} has pose shape {tuple(tensor.shape[2:])}; "
            f"expected {expected_shape}."
        )
    if tensor.device != mask_device:
        raise ValueError(f"{surface_name} channel {channel!r} device must match the masks.")
