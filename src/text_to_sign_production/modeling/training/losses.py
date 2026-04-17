"""Mask-aware regression losses for Sprint 3 baseline pose outputs."""

from __future__ import annotations

import torch

from ._surface import (
    floating_point_accumulation_dtype,
    iter_validated_pose_channels,
    valid_frame_count,
)
from .masking import build_effective_frame_mask


def masked_pose_mse_loss(
    predictions: object,
    targets: object,
    padding_mask: torch.Tensor,
    frame_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return contribution-weighted MSE over valid Sprint 3 pose coordinates.

    The computed channels are fixed to the default Sprint 3 target surface: ``body``,
    ``left_hand``, and ``right_hand``. Extra channels are ignored. Padded frames and invalid real
    frames do not contribute to either the numerator or denominator.
    """

    effective_frame_mask = build_effective_frame_mask(
        padding_mask=padding_mask,
        frame_valid_mask=frame_valid_mask,
    )
    frame_count = valid_frame_count(effective_frame_mask)
    if bool((frame_count == 0).item()):
        raise ValueError("masked pose MSE loss has zero valid contributing frames.")

    total_squared_error: torch.Tensor | None = None
    total_coordinate_count: torch.Tensor | None = None
    contribution_mask = effective_frame_mask.unsqueeze(-1).unsqueeze(-1)
    for _channel, prediction, target in iter_validated_pose_channels(
        predictions=predictions,
        targets=targets,
        mask_shape=(effective_frame_mask.shape[0], effective_frame_mask.shape[1]),
        mask_device=effective_frame_mask.device,
    ):
        squared_error = (prediction - target).square()
        masked_squared_error = squared_error * contribution_mask.to(dtype=squared_error.dtype)
        accumulation_dtype = floating_point_accumulation_dtype(masked_squared_error)
        channel_squared_error = masked_squared_error.sum(dtype=accumulation_dtype)
        total_squared_error = (
            channel_squared_error
            if total_squared_error is None
            else total_squared_error + channel_squared_error
        )
        channel_coordinate_count = frame_count * prediction.shape[2] * prediction.shape[3]
        total_coordinate_count = (
            channel_coordinate_count
            if total_coordinate_count is None
            else total_coordinate_count + channel_coordinate_count
        )

    if total_squared_error is None or total_coordinate_count is None:
        raise ValueError("masked pose MSE loss has no configured target channels.")
    result_dtype = floating_point_accumulation_dtype(total_squared_error)
    return total_squared_error / total_coordinate_count.to(dtype=result_dtype)
