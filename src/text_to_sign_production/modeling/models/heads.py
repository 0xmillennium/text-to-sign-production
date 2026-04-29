"""Pose output heads for channel-separated M0 full-BFH predictions."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class PoseChannelHead(nn.Module):
    """Project decoder features into one continuous pose channel."""

    def __init__(
        self,
        *,
        input_dim: int,
        keypoints: int,
        coordinate_dim: int = 2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if keypoints <= 0:
            raise ValueError("keypoints must be positive.")
        if coordinate_dim <= 0:
            raise ValueError("coordinate_dim must be positive.")

        self.keypoints = keypoints
        self.coordinate_dim = coordinate_dim
        self.projection = nn.Linear(input_dim, keypoints * coordinate_dim)

    def forward(self, decoder_features: torch.Tensor) -> torch.Tensor:
        """Return channel predictions with shape [B, T, keypoints, coordinate_dim]."""

        if decoder_features.ndim != 3:
            raise ValueError("decoder_features must have shape [B, T, H].")
        batch_size, target_frames, _ = decoder_features.shape
        projected = self.projection(decoder_features)
        return cast(
            torch.Tensor,
            projected.view(
                batch_size,
                target_frames,
                self.keypoints,
                self.coordinate_dim,
            ),
        )
