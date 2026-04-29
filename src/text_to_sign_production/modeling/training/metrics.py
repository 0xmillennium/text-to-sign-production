"""Minimum validation metrics for M0 full-BFH pose outputs."""

from __future__ import annotations

import torch

from ._surface import (
    floating_point_accumulation_dtype,
    iter_validated_pose_channels,
)
from .losses import _reference_confidence_point_mask
from .masking import build_effective_frame_mask


def masked_average_keypoint_l2_error(
    predictions: object,
    targets: object,
    padding_mask: torch.Tensor,
    frame_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return contribution-weighted average keypoint L2 error on valid frames.

    Distances are computed per keypoint over the coordinate dimension and then averaged over valid
    timesteps across the full BFH channel set.
    """

    effective_frame_mask = build_effective_frame_mask(
        padding_mask=padding_mask,
        frame_valid_mask=frame_valid_mask,
    )
    total_l2_error: torch.Tensor | None = None
    total_keypoint_count: torch.Tensor | None = None
    for _channel, prediction, target in iter_validated_pose_channels(
        predictions=predictions,
        targets=targets,
        mask_shape=(effective_frame_mask.shape[0], effective_frame_mask.shape[1]),
        mask_device=effective_frame_mask.device,
    ):
        point_mask = _reference_confidence_point_mask(
            targets,
            channel=_channel,
            effective_frame_mask=effective_frame_mask,
        )
        valid_point_count = point_mask.sum()
        if bool((valid_point_count == 0).item()):
            continue
        keypoint_l2_error = torch.linalg.vector_norm(prediction - target, ord=2, dim=-1)
        masked_l2_error = keypoint_l2_error * point_mask.to(dtype=keypoint_l2_error.dtype)
        accumulation_dtype = floating_point_accumulation_dtype(masked_l2_error)
        channel_l2_error = masked_l2_error.sum(dtype=accumulation_dtype)
        total_l2_error = (
            channel_l2_error if total_l2_error is None else total_l2_error + channel_l2_error
        )
        total_keypoint_count = (
            valid_point_count
            if total_keypoint_count is None
            else total_keypoint_count + valid_point_count
        )

    if total_l2_error is None or total_keypoint_count is None:
        raise ValueError("masked average keypoint L2 error has no supervised target points.")
    result_dtype = floating_point_accumulation_dtype(total_l2_error)
    return total_l2_error / total_keypoint_count.to(dtype=result_dtype)


def masked_pose_metric_tensors(
    predictions: object,
    targets: object,
    padding_mask: torch.Tensor,
    frame_valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return confidence-masked M0 metric tensors in canonical coordinate space."""

    effective_frame_mask = build_effective_frame_mask(
        padding_mask=padding_mask,
        frame_valid_mask=frame_valid_mask,
    )
    total_l2: torch.Tensor | None = None
    total_l1: torch.Tensor | None = None
    total_sq: torch.Tensor | None = None
    total_points: torch.Tensor | None = None
    total_valid_frames = effective_frame_mask.sum()
    metrics: dict[str, torch.Tensor] = {
        "temporal_velocity_l2_diagnostic": torch.zeros(
            (),
            dtype=torch.float32,
            device=effective_frame_mask.device,
        ),
        "temporal_smoothness_diagnostic": torch.zeros(
            (),
            dtype=torch.float32,
            device=effective_frame_mask.device,
        ),
    }
    temporal_count = 0
    for channel, prediction, target in iter_validated_pose_channels(
        predictions=predictions,
        targets=targets,
        mask_shape=(effective_frame_mask.shape[0], effective_frame_mask.shape[1]),
        mask_device=effective_frame_mask.device,
    ):
        point_mask = _reference_confidence_point_mask(
            targets,
            channel=channel,
            effective_frame_mask=effective_frame_mask,
        )
        valid_points = point_mask.sum()
        metrics[f"per_channel_{channel}_valid_point_count"] = valid_points
        metrics[f"per_channel_{channel}_valid_frame_count"] = point_mask.any(dim=-1).sum()
        if bool((valid_points == 0).item()):
            zero = prediction.sum(dtype=floating_point_accumulation_dtype(prediction)) * 0.0
            metrics[f"per_channel_{channel}_masked_l2"] = zero
            metrics[f"per_channel_{channel}_masked_mae"] = zero
            continue
        error = prediction - target
        l2 = torch.linalg.vector_norm(error, ord=2, dim=-1)
        l1 = error.abs().mean(dim=-1)
        sq = error.square().mean(dim=-1)
        mask = point_mask.to(dtype=error.dtype)
        accumulation_dtype = floating_point_accumulation_dtype(error)
        channel_l2 = (l2 * mask).sum(dtype=accumulation_dtype)
        channel_l1 = (l1 * mask).sum(dtype=accumulation_dtype)
        channel_sq = (sq * mask).sum(dtype=accumulation_dtype)
        valid_points_float = valid_points.to(dtype=accumulation_dtype)
        metrics[f"per_channel_{channel}_masked_l2"] = channel_l2 / valid_points_float
        metrics[f"per_channel_{channel}_masked_mae"] = channel_l1 / valid_points_float
        metrics[f"per_channel_{channel}_masked_l2_sum"] = channel_l2
        metrics[f"per_channel_{channel}_masked_abs_sum"] = channel_l1
        metrics[f"per_channel_{channel}_masked_squared_sum"] = channel_sq
        total_l2 = channel_l2 if total_l2 is None else total_l2 + channel_l2
        total_l1 = channel_l1 if total_l1 is None else total_l1 + channel_l1
        total_sq = channel_sq if total_sq is None else total_sq + channel_sq
        total_points = valid_points if total_points is None else total_points + valid_points
        velocity, smoothness = _temporal_diagnostics(prediction, effective_frame_mask)
        metrics["temporal_velocity_l2_diagnostic"] = (
            metrics["temporal_velocity_l2_diagnostic"] + velocity
        )
        metrics["temporal_smoothness_diagnostic"] = (
            metrics["temporal_smoothness_diagnostic"] + smoothness
        )
        temporal_count += 1
    if total_l2 is None or total_l1 is None or total_sq is None or total_points is None:
        raise ValueError("masked pose metrics have no supervised target points.")
    result_dtype = floating_point_accumulation_dtype(total_l2)
    point_count = total_points.to(dtype=result_dtype)
    metrics["validation_masked_l2_mean"] = total_l2 / point_count
    metrics["validation_masked_mae"] = total_l1 / point_count
    metrics["validation_masked_rmse"] = torch.sqrt(total_sq / point_count)
    metrics["validation_metric"] = metrics["validation_masked_l2_mean"]
    # Raw numerator sums for correct epoch-level aggregation.
    # Epoch aggregation must sum these, not re-weight per-batch means.
    metrics["masked_l2_sum"] = total_l2
    metrics["masked_abs_sum"] = total_l1
    metrics["masked_squared_sum"] = total_sq
    metrics["valid_point_count"] = total_points
    metrics["per_channel_valid_point_count"] = total_points
    metrics["per_channel_valid_frame_count"] = total_valid_frames
    if temporal_count > 0:
        metrics["temporal_velocity_l2_diagnostic"] = (
            metrics["temporal_velocity_l2_diagnostic"] / temporal_count
        )
        metrics["temporal_smoothness_diagnostic"] = (
            metrics["temporal_smoothness_diagnostic"] / temporal_count
        )
    return metrics


def metric_tensor_to_float(value: torch.Tensor) -> float:
    """Detach a scalar tensor as a Python float."""

    return float(value.detach().cpu().item())


def _temporal_diagnostics(
    prediction: torch.Tensor,
    effective_frame_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape[1] < 2:
        zero = prediction.sum(dtype=floating_point_accumulation_dtype(prediction)) * 0.0
        return zero, zero
    pair_mask = effective_frame_mask[:, 1:] & effective_frame_mask[:, :-1]
    if not bool(pair_mask.any().item()):
        zero = prediction.sum(dtype=floating_point_accumulation_dtype(prediction)) * 0.0
        return zero, zero
    velocity = prediction[:, 1:] - prediction[:, :-1]
    velocity_l2 = torch.linalg.vector_norm(velocity, ord=2, dim=-1).mean(dim=-1)
    masked_velocity = velocity_l2 * pair_mask.to(dtype=velocity_l2.dtype)
    velocity_mean = masked_velocity.sum() / pair_mask.sum().to(dtype=masked_velocity.dtype)
    if prediction.shape[1] < 3:
        return velocity_mean, velocity_mean * 0.0
    smooth_mask = pair_mask[:, 1:] & pair_mask[:, :-1]
    if not bool(smooth_mask.any().item()):
        return velocity_mean, velocity_mean * 0.0
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    smooth_l2 = torch.linalg.vector_norm(acceleration, ord=2, dim=-1).mean(dim=-1)
    masked_smooth = smooth_l2 * smooth_mask.to(dtype=smooth_l2.dtype)
    smooth_mean = masked_smooth.sum() / smooth_mask.sum().to(dtype=masked_smooth.dtype)
    return velocity_mean, smooth_mean
