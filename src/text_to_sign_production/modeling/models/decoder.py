"""Simple temporal decoder for M0 direct pose regression."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn


class SimplePoseDecoder(nn.Module):
    """Expand pooled text embeddings into per-frame residual decoder features."""

    def __init__(
        self,
        *,
        text_embedding_dim: int,
        hidden_dim: int = 256,
        feature_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        frame_position_encoding_dim: int = 0,
    ) -> None:
        super().__init__()
        resolved_feature_dim = hidden_dim if feature_dim is None else feature_dim
        if text_embedding_dim <= 0:
            raise ValueError("text_embedding_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if resolved_feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError("dropout must be in [0, 1].")
        if frame_position_encoding_dim < 0:
            raise ValueError("frame_position_encoding_dim must not be negative.")

        self.feature_dim = resolved_feature_dim
        self.frame_position_encoding_dim = frame_position_encoding_dim
        input_dim = text_embedding_dim + 1 + frame_position_encoding_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            _ResidualMlpBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, resolved_feature_dim),
            nn.GELU(),
        )

    def forward(
        self,
        pooled_embedding: torch.Tensor,
        *,
        lengths: torch.Tensor,
        target_frames: int,
    ) -> torch.Tensor:
        """Create frame-level decoder features using text features and normalized positions."""

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

        inputs = [text_features, normalized_positions]
        if self.frame_position_encoding_dim > 0:
            inputs.append(
                _sinusoidal_position_encoding(
                    normalized_positions, self.frame_position_encoding_dim
                )
            )
        decoder_input = torch.cat(inputs, dim=-1)
        features = self.input_projection(decoder_input)
        for block in self.blocks:
            features = block(features)
        return cast(torch.Tensor, self.output_projection(features))


class _ResidualMlpBlock(nn.Module):
    def __init__(self, *, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, features + self.network(self.norm(features)))


def _sinusoidal_position_encoding(
    normalized_positions: torch.Tensor,
    encoding_dim: int,
) -> torch.Tensor:
    if encoding_dim <= 0:
        raise ValueError("encoding_dim must be positive.")
    device = normalized_positions.device
    dtype = normalized_positions.dtype
    half_dim = max(1, (encoding_dim + 1) // 2)
    frequencies = torch.exp(
        torch.arange(half_dim, device=device, dtype=dtype)
        * (-math.log(10000.0) / max(1, half_dim - 1))
    )
    angles = normalized_positions * frequencies.view(1, 1, half_dim)
    encoded = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    return encoded[..., :encoding_dim]
