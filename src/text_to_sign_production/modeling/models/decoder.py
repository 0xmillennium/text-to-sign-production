"""Simple temporal decoder for Sprint 3 baseline pose regression."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class SimplePoseDecoder(nn.Module):
    """Expand pooled text embeddings into per-frame latent features."""

    def __init__(
        self,
        *,
        text_embedding_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        if text_embedding_dim <= 0:
            raise ValueError("text_embedding_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")

        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Linear(text_embedding_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        pooled_embedding: torch.Tensor,
        *,
        lengths: torch.Tensor,
        target_frames: int,
    ) -> torch.Tensor:
        """Create frame-level latents using text features and normalized positions."""

        if pooled_embedding.ndim != 2:
            raise ValueError("pooled_embedding must have shape [B, D].")
        if lengths.ndim != 1:
            raise ValueError("lengths must have shape [B].")
        if lengths.shape[0] != pooled_embedding.shape[0]:
            raise ValueError("lengths batch dimension must match pooled_embedding.")
        if target_frames < 0:
            raise ValueError("target_frames must not be negative.")

        batch_size = pooled_embedding.shape[0]
        device = pooled_embedding.device
        dtype = pooled_embedding.dtype
        positions = torch.arange(target_frames, device=device, dtype=dtype).view(1, target_frames)
        denominators = (lengths.to(device=device, dtype=dtype) - 1).clamp_min(1).view(batch_size, 1)
        normalized_positions = (positions / denominators).unsqueeze(-1)
        text_features = pooled_embedding.unsqueeze(1).expand(batch_size, target_frames, -1)

        decoder_input = torch.cat((text_features, normalized_positions), dim=-1)
        return cast(torch.Tensor, self.network(decoder_input))
