"""Stable direct-regression loss entry point for the M0 provider."""

from __future__ import annotations

from collections.abc import Mapping


def compute_base_direct_loss(
    predictions: object,
    targets: object,
    padding_mask: object,
    frame_valid_mask: object,
    *,
    channel_weights: Mapping[str, float] | None = None,
):
    """Compute the existing channel-balanced masked supervised regression loss."""

    from text_to_sign_production.modeling.training.losses import (
        channel_balanced_masked_pose_mse_loss,
    )

    return channel_balanced_masked_pose_mse_loss(
        predictions=predictions,
        targets=targets,
        padding_mask=padding_mask,
        frame_valid_mask=frame_valid_mask,
        channel_weights=channel_weights,
    )


__all__ = ["compute_base_direct_loss"]
