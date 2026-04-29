"""Mask-aware channel-balanced regression losses for M0 full-BFH pose outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from text_to_sign_production.modeling.contracts import (
    DEFAULT_CHANNEL_WEIGHTS,
    REFERENCE_CONFIDENCE_MASK_POLICY,
    validate_channel_weights,
)

from ._surface import (
    floating_point_accumulation_dtype,
    iter_validated_pose_channels,
    valid_frame_count,
)
from .masking import build_effective_frame_mask


@dataclass(frozen=True, slots=True)
class ChannelBalancedPoseLoss:
    """Tensor loss plus per-channel scalar surfaces for M0 metric writers."""

    total_loss: torch.Tensor
    channel_losses: dict[str, torch.Tensor]
    channel_valid_point_counts: dict[str, torch.Tensor]
    channel_weights: dict[str, float]

    def detached_metrics(self) -> dict[str, float]:
        """Return CPU float metrics using the Phase 4A per-channel names."""

        metrics = {
            f"{channel}_loss": float(loss.detach().cpu().item())
            for channel, loss in self.channel_losses.items()
        }
        metrics["total_loss"] = float(self.total_loss.detach().cpu().item())
        return metrics

    def detached_valid_point_counts(self) -> dict[str, int]:
        """Return CPU integer valid point counts for each supervised M0 channel."""

        return {
            channel: int(count.detach().cpu().item())
            for channel, count in self.channel_valid_point_counts.items()
        }


def masked_pose_mse_loss(
    predictions: object,
    targets: object,
    padding_mask: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    *,
    channel_weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Return channel-balanced MSE over valid M0 full-BFH pose coordinates."""

    return channel_balanced_masked_pose_mse_loss(
        predictions=predictions,
        targets=targets,
        padding_mask=padding_mask,
        frame_valid_mask=frame_valid_mask,
        channel_weights=channel_weights,
    ).total_loss


def channel_balanced_masked_pose_mse_loss(
    predictions: object,
    targets: object,
    padding_mask: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    *,
    channel_weights: Mapping[str, float] | None = None,
) -> ChannelBalancedPoseLoss:
    """Return per-channel and total M0 masked MSE.

    Each BFH channel is reduced independently over valid frames and supervised target points, then
    combined by explicit channel weights. Reference confidence arrays are used only to determine
    target supervision availability under ``REFERENCE_CONFIDENCE_MASK_POLICY``; generated
    prediction confidence is not used by this loss and is not treated as model uncertainty.
    """

    resolved_channel_weights = validate_channel_weights(
        DEFAULT_CHANNEL_WEIGHTS if channel_weights is None else channel_weights
    )
    effective_frame_mask = build_effective_frame_mask(
        padding_mask=padding_mask,
        frame_valid_mask=frame_valid_mask,
    )
    frame_count = valid_frame_count(effective_frame_mask)
    if bool((frame_count == 0).item()):
        raise ValueError("masked pose MSE loss has zero valid contributing frames.")

    channel_losses: dict[str, torch.Tensor] = {}
    channel_valid_point_counts: dict[str, torch.Tensor] = {}
    weighted_total: torch.Tensor | None = None
    active_weight_sum = 0.0
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
        valid_point_count = point_mask.sum()
        channel_valid_point_counts[channel] = valid_point_count
        if bool((valid_point_count == 0).item()):
            channel_loss = prediction.sum(dtype=floating_point_accumulation_dtype(prediction)) * 0.0
            channel_losses[channel] = channel_loss
            continue

        squared_error = (prediction - target).square().mean(dim=-1)
        masked_squared_error = squared_error * point_mask.to(dtype=squared_error.dtype)
        accumulation_dtype = floating_point_accumulation_dtype(masked_squared_error)
        channel_squared_error = masked_squared_error.sum(dtype=accumulation_dtype)
        channel_loss = channel_squared_error / valid_point_count.to(dtype=accumulation_dtype)
        channel_losses[channel] = channel_loss

        channel_weight = resolved_channel_weights[channel]
        if channel_weight <= 0.0:
            continue
        weighted_channel_loss = channel_loss * channel_weight
        weighted_total = (
            weighted_channel_loss
            if weighted_total is None
            else weighted_total + weighted_channel_loss
        )
        active_weight_sum += channel_weight

    if weighted_total is None or active_weight_sum <= 0.0:
        raise ValueError(
            "masked pose MSE loss has no supervised target points for any positive-weight "
            f"channel under {REFERENCE_CONFIDENCE_MASK_POLICY!r}."
        )
    return ChannelBalancedPoseLoss(
        total_loss=weighted_total / active_weight_sum,
        channel_losses=channel_losses,
        channel_valid_point_counts=channel_valid_point_counts,
        channel_weights=resolved_channel_weights,
    )


def _reference_confidence_point_mask(
    targets: object,
    *,
    channel: str,
    effective_frame_mask: torch.Tensor,
) -> torch.Tensor:
    confidence = _extract_reference_confidence_tensor(
        targets,
        channel=channel,
        mask_shape=(effective_frame_mask.shape[0], effective_frame_mask.shape[1]),
        mask_device=effective_frame_mask.device,
    )
    reference_point_available = torch.isfinite(confidence) & (confidence > 0.0)
    return effective_frame_mask.unsqueeze(-1) & reference_point_available


def _extract_reference_confidence_tensor(
    surface: object,
    *,
    channel: str,
    mask_shape: tuple[int, int],
    mask_device: torch.device,
) -> torch.Tensor:
    field_name = f"{channel}_confidence"
    value: Any
    if isinstance(surface, Mapping):
        if field_name not in surface:
            raise ValueError(f"targets are missing required confidence field {field_name!r}.")
        value = surface[field_name]
    else:
        if not hasattr(surface, field_name):
            raise ValueError(f"targets are missing required confidence field {field_name!r}.")
        value = getattr(surface, field_name)

    if not isinstance(value, torch.Tensor):
        raise ValueError(f"targets confidence field {field_name!r} must be a torch.Tensor.")
    if not torch.is_floating_point(value):
        raise ValueError(f"targets confidence field {field_name!r} must use a floating dtype.")
    if value.ndim != 3:
        raise ValueError(f"targets confidence field {field_name!r} must have shape [B, T, K].")
    if tuple(value.shape[:2]) != mask_shape:
        raise ValueError(
            f"targets confidence field {field_name!r} batch/time shape must match the masks."
        )
    if value.device != mask_device:
        raise ValueError(f"targets confidence field {field_name!r} device must match the masks.")
    return value
