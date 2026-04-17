"""Pose output heads for channel-separated Sprint 3 baseline predictions."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class PoseChannelHead(nn.Module):
    """Project temporal latents into one continuous pose channel."""

    def __init__(
        self,
        *,
        latent_dim: int,
        keypoints: int,
        coordinate_dim: int = 2,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if keypoints <= 0:
            raise ValueError("keypoints must be positive.")
        if coordinate_dim <= 0:
            raise ValueError("coordinate_dim must be positive.")

        self.keypoints = keypoints
        self.coordinate_dim = coordinate_dim
        self.projection = nn.Linear(latent_dim, keypoints * coordinate_dim)

    def forward(self, temporal_latents: torch.Tensor) -> torch.Tensor:
        """Return channel predictions with shape [B, T, keypoints, coordinate_dim]."""

        if temporal_latents.ndim != 3:
            raise ValueError("temporal_latents must have shape [B, T, H].")
        batch_size, target_frames, _ = temporal_latents.shape
        projected = self.projection(temporal_latents)
        return cast(
            torch.Tensor,
            projected.view(
                batch_size,
                target_frames,
                self.keypoints,
                self.coordinate_dim,
            ),
        )
