"""Text-conditioned temporal MLP denoiser for latent_diffusion foundation."""

from __future__ import annotations

import math

import torch
from torch import nn

from text_to_sign_production.modeling.candidates.latent_diffusion.config import (
    LatentDiffusionConfig,
    LatentLengthConfig,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.errors import (
    LatentDiffusionError,
)


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding with a projection head."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise LatentDiffusionError("timestep embedding_dim must be positive.")
        self.embedding_dim = int(embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim != 1:
            raise LatentDiffusionError("timestep must have shape (batch,).")
        half = self.embedding_dim // 2
        if half == 0:
            raw = timestep.to(dtype=torch.float32).unsqueeze(1)
        else:
            scale = math.log(10000.0) / max(half - 1, 1)
            frequencies = torch.exp(
                torch.arange(half, device=timestep.device, dtype=torch.float32) * -scale
            )
            angles = timestep.to(dtype=torch.float32).unsqueeze(1) * frequencies.unsqueeze(0)
            raw = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
            if raw.shape[1] < self.embedding_dim:
                raw = torch.nn.functional.pad(raw, (0, self.embedding_dim - raw.shape[1]))
        return self.projection(raw)


class TemporalPositionEmbedding(nn.Module):
    """Learned frame-position embedding for frame-level latent denoising."""

    def __init__(self, embedding_dim: int, *, max_positions: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise LatentDiffusionError("position embedding_dim must be positive.")
        if max_positions <= 0:
            raise LatentDiffusionError("max_positions must be positive.")
        self.embedding = nn.Embedding(max_positions, embedding_dim)
        self.max_positions = int(max_positions)

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        if position.ndim != 1:
            raise LatentDiffusionError("position must have shape (batch,).")
        if torch.any(position < 0) or torch.any(position >= self.embedding.num_embeddings):
            raise LatentDiffusionError(
                "position contains an index outside the temporal embedding range."
            )
        return self.embedding(position.to(dtype=torch.long))


class LatentDenoiser(nn.Module):
    """Temporal MLP epsilon predictor conditioned on text and frame position."""

    def __init__(
        self,
        *,
        latent_dim: int,
        text_embedding_dim: int,
        hidden_dim: int,
        timestep_embedding_dim: int,
        position_embedding_dim: int,
        dropout: float,
        max_positions: int,
    ) -> None:
        super().__init__()
        for value, name in (
            (latent_dim, "latent_dim"),
            (text_embedding_dim, "text_embedding_dim"),
            (hidden_dim, "hidden_dim"),
        ):
            if value <= 0:
                raise LatentDiffusionError(f"{name} must be positive.")
        if not 0.0 <= float(dropout) < 1.0:
            raise LatentDiffusionError("dropout must be in [0, 1).")
        self.latent_dim = int(latent_dim)
        self.text_embedding_dim = int(text_embedding_dim)
        self.timestep_embedding = TimestepEmbedding(timestep_embedding_dim)
        self.position_embedding = TemporalPositionEmbedding(
            position_embedding_dim,
            max_positions=max_positions,
        )
        self.max_positions = int(max_positions)
        input_dim = (
            self.latent_dim
            + self.text_embedding_dim
            + timestep_embedding_dim
            + position_embedding_dim
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.latent_dim),
        )

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        text_embedding: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_latent.ndim != 2 or noisy_latent.shape[1] != self.latent_dim:
            raise LatentDiffusionError(
                f"noisy_latent must have shape (batch, {self.latent_dim})."
            )
        batch = noisy_latent.shape[0]
        if text_embedding.shape != (batch, self.text_embedding_dim):
            raise LatentDiffusionError(
                f"text_embedding must have shape (batch, {self.text_embedding_dim})."
            )
        if timestep.shape != (batch,):
            raise LatentDiffusionError("timestep must have shape (batch,).")
        if position.shape != (batch,):
            raise LatentDiffusionError("position must have shape (batch,).")
        if batch > self.max_positions:
            raise LatentDiffusionError(
                f"latent denoiser sequence length {batch} exceeds denoiser.max_positions "
                f"{self.max_positions}."
            )
        joined = torch.cat(
            (
                noisy_latent,
                text_embedding,
                self.timestep_embedding(timestep),
                self.position_embedding(position),
            ),
            dim=1,
        )
        return self.net(joined)


def build_latent_denoiser(
    *,
    latent_dim: int,
    text_embedding_dim: int,
    config: LatentDiffusionConfig,
) -> LatentDenoiser:
    """Build the configured temporal MLP denoiser."""

    if not isinstance(config, LatentDiffusionConfig):
        raise LatentDiffusionError("config must be a LatentDiffusionConfig.")
    if config.denoiser.latent_dim is not None and config.denoiser.latent_dim != latent_dim:
        raise LatentDiffusionError(
            "denoiser.latent_dim does not match the requested latent_dim."
        )
    return LatentDenoiser(
        latent_dim=latent_dim,
        text_embedding_dim=text_embedding_dim,
        hidden_dim=config.denoiser.hidden_dim,
        timestep_embedding_dim=config.denoiser.timestep_embedding_dim,
        position_embedding_dim=config.denoiser.position_embedding_dim,
        max_positions=config.denoiser.max_positions,
        dropout=config.denoiser.dropout,
    )


class LatentLengthPredictor(nn.Module):
    """Text-conditioned positive frame-count predictor for latent generation."""

    def __init__(
        self,
        *,
        text_embedding_dim: int,
        hidden_dim: int,
        min_generated_frames: int,
        max_generated_frames: int | None,
    ) -> None:
        super().__init__()
        if text_embedding_dim <= 0:
            raise LatentDiffusionError("text_embedding_dim must be positive.")
        if hidden_dim <= 0:
            raise LatentDiffusionError("hidden_dim must be positive.")
        if min_generated_frames <= 0:
            raise LatentDiffusionError("min_generated_frames must be positive.")
        if max_generated_frames is not None and max_generated_frames < min_generated_frames:
            raise LatentDiffusionError(
                "max_generated_frames must be null or at least min_generated_frames."
            )
        self.text_embedding_dim = int(text_embedding_dim)
        self.min_generated_frames = int(min_generated_frames)
        self.max_generated_frames = (
            None if max_generated_frames is None else int(max_generated_frames)
        )
        self.net = nn.Sequential(
            nn.Linear(self.text_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        if text_embedding.ndim != 2 or text_embedding.shape[1] != self.text_embedding_dim:
            raise LatentDiffusionError(
                f"text_embedding must have shape (batch, {self.text_embedding_dim})."
            )
        return torch.nn.functional.softplus(self.net(text_embedding).squeeze(-1)) + 1.0


def build_latent_length_predictor(
    *,
    text_embedding_dim: int,
    config: LatentDiffusionConfig,
) -> LatentLengthPredictor:
    """Build the configured text-to-length predictor."""

    if not isinstance(config, LatentDiffusionConfig):
        raise LatentDiffusionError("config must be a LatentDiffusionConfig.")
    return LatentLengthPredictor(
        text_embedding_dim=text_embedding_dim,
        hidden_dim=config.denoiser.hidden_dim,
        min_generated_frames=config.length.min_generated_frames,
        max_generated_frames=config.length.max_generated_frames,
    )


def clamp_predicted_frame_count(
    predicted_length: float,
    *,
    config: LatentLengthConfig,
    max_positions: int,
) -> int:
    """Clamp a positive scalar length prediction to configured generation bounds."""

    if not math.isfinite(float(predicted_length)):
        raise LatentDiffusionError("predicted frame count must be finite.")
    if not isinstance(max_positions, int) or isinstance(max_positions, bool) or max_positions < 1:
        raise LatentDiffusionError("denoiser.max_positions must be a positive integer.")
    count = int(round(float(predicted_length)))
    count = max(int(config.min_generated_frames), count)
    if config.max_generated_frames is not None:
        count = min(int(config.max_generated_frames), count)
    if count < 1:
        raise LatentDiffusionError("generated frame count must be at least 1.")
    if count > max_positions:
        raise LatentDiffusionError(
            f"generated frame count {count} exceeds denoiser.max_positions "
            f"{max_positions}; reduce length.max_generated_frames or increase denoiser.max_positions."
        )
    return count


def compute_denoising_loss(
    *,
    predicted_noise: torch.Tensor,
    target_noise: torch.Tensor,
    validity_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute finite mask-aware MSE over valid latent features."""

    if predicted_noise.shape != target_noise.shape:
        raise LatentDiffusionError("predicted_noise and target_noise shapes must match.")
    if validity_mask.shape != predicted_noise.shape:
        raise LatentDiffusionError("validity_mask shape must match predicted_noise.")
    mask = validity_mask.to(dtype=torch.bool)
    if not torch.any(mask):
        raise LatentDiffusionError("validity_mask must contain at least one valid feature.")
    squared = (predicted_noise - target_noise).pow(2)
    loss = squared[mask].mean()
    if not torch.isfinite(loss):
        raise LatentDiffusionError("denoising loss is not finite.")
    return loss


__all__ = [
    "LatentDenoiser",
    "LatentLengthPredictor",
    "TemporalPositionEmbedding",
    "TimestepEmbedding",
    "build_latent_length_predictor",
    "build_latent_denoiser",
    "clamp_predicted_frame_count",
    "compute_denoising_loss",
]
