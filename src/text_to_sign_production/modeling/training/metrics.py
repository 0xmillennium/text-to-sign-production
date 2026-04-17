"""Minimum validation metrics for Sprint 3 baseline pose outputs."""

from __future__ import annotations

import torch

from ._surface import (
    floating_point_accumulation_dtype,
    iter_validated_pose_channels,
    valid_frame_count,
)
from .masking import build_effective_frame_mask


def masked_average_keypoint_l2_error(
    predictions: object,
    targets: object,
    padding_mask: torch.Tensor,
    frame_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return contribution-weighted average keypoint L2 error on valid frames.

    Distances are computed per keypoint over the coordinate dimension and then averaged over valid
    timesteps across ``body``, ``left_hand``, and ``right_hand``.
    """

    effective_frame_mask = build_effective_frame_mask(
        padding_mask=padding_mask,
        frame_valid_mask=frame_valid_mask,
    )
    frame_count = valid_frame_count(effective_frame_mask)
    if bool((frame_count == 0).item()):
        raise ValueError("masked average keypoint L2 error has zero valid contributing frames.")

    total_l2_error: torch.Tensor | None = None
    total_keypoint_count: torch.Tensor | None = None
    contribution_mask = effective_frame_mask.unsqueeze(-1)
    for _channel, prediction, target in iter_validated_pose_channels(
        predictions=predictions,
        targets=targets,
        mask_shape=(effective_frame_mask.shape[0], effective_frame_mask.shape[1]),
        mask_device=effective_frame_mask.device,
    ):
        keypoint_l2_error = torch.linalg.vector_norm(prediction - target, ord=2, dim=-1)
        masked_l2_error = keypoint_l2_error * contribution_mask.to(dtype=keypoint_l2_error.dtype)
        accumulation_dtype = floating_point_accumulation_dtype(masked_l2_error)
        channel_l2_error = masked_l2_error.sum(dtype=accumulation_dtype)
        total_l2_error = (
            channel_l2_error if total_l2_error is None else total_l2_error + channel_l2_error
        )
        channel_keypoint_count = frame_count * prediction.shape[2]
        total_keypoint_count = (
            channel_keypoint_count
            if total_keypoint_count is None
            else total_keypoint_count + channel_keypoint_count
        )

    if total_l2_error is None or total_keypoint_count is None:
        raise ValueError("masked average keypoint L2 error has no configured target channels.")
    result_dtype = floating_point_accumulation_dtype(total_l2_error)
    return total_l2_error / total_keypoint_count.to(dtype=result_dtype)
