"""Minimal VQ-style token-unit tokenizer for full-BFH pose vectors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from text_to_sign_production.modeling.candidates.learned_pose_token.config import (
    LearnedPoseTokenConfig,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.errors import (
    LearnedPoseTokenError,
)


@dataclass(frozen=True, slots=True)
class LearnedPoseTokenizerOutput:
    reconstructed: torch.Tensor
    token_ids: torch.Tensor
    quantized: torch.Tensor
    encoder_latents: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor


class PoseFrameEncoder(nn.Module):
    """Encode one flattened full-BFH token unit into a latent vector."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        _positive_int(input_dim, "input_dim")
        _positive_int(hidden_dim, "hidden_dim")
        _positive_int(latent_dim, "latent_dim")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class VectorQuantizer(nn.Module):
    """Nearest-neighbor codebook with a straight-through estimator."""

    def __init__(self, codebook_size: int, embedding_dim: int) -> None:
        super().__init__()
        if not isinstance(codebook_size, int) or isinstance(codebook_size, bool) or codebook_size <= 1:
            raise LearnedPoseTokenError("codebook_size must be greater than 1.")
        _positive_int(embedding_dim, "embedding_dim")
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)

    def forward(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if latents.ndim != 2 or latents.shape[1] != self.embedding_dim:
            raise LearnedPoseTokenError(
                f"latents must have shape (batch, {self.embedding_dim}); got {tuple(latents.shape)}."
            )
        codebook = self.embedding.weight
        distances = (
            latents.square().sum(dim=1, keepdim=True)
            - 2.0 * latents @ codebook.t()
            + codebook.square().sum(dim=1).unsqueeze(0)
        )
        token_ids = torch.argmin(distances, dim=1)
        quantized = self.embedding(token_ids)
        codebook_loss = F.mse_loss(quantized, latents.detach())
        commitment_loss = F.mse_loss(latents, quantized.detach())
        quantized_st = latents + (quantized - latents).detach()
        return quantized_st, token_ids, codebook_loss, commitment_loss


class PoseFrameDecoder(nn.Module):
    """Decode quantized latent vectors back to flattened pose token units."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        _positive_int(latent_dim, "latent_dim")
        _positive_int(hidden_dim, "hidden_dim")
        _positive_int(output_dim, "output_dim")
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.net(latents)


class LearnedPoseTokenizer(nn.Module):
    """Token-unit VQ autoencoder for pose-token representation learning."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        codebook_size: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.encoder = PoseFrameEncoder(input_dim, hidden_dim, latent_dim)
        self.quantizer = VectorQuantizer(codebook_size, latent_dim)
        self.decoder = PoseFrameDecoder(latent_dim, hidden_dim, input_dim)

    def forward(self, values: torch.Tensor) -> LearnedPoseTokenizerOutput:
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise LearnedPoseTokenError(
                f"values must have shape (batch, {self.input_dim}); got {tuple(values.shape)}."
            )
        if not torch.isfinite(values).all():
            raise LearnedPoseTokenError("values must contain only finite numbers.")
        latents = self.encoder(values)
        quantized, token_ids, codebook_loss, commitment_loss = self.quantizer(latents)
        reconstructed = self.decoder(quantized)
        return LearnedPoseTokenizerOutput(
            reconstructed=reconstructed,
            token_ids=token_ids,
            quantized=quantized,
            encoder_latents=latents,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
        )


def build_learned_pose_tokenizer(
    *,
    input_dim: int,
    config: LearnedPoseTokenConfig,
) -> LearnedPoseTokenizer:
    """Build the configured foundation tokenizer without provider registration."""

    if not isinstance(config, LearnedPoseTokenConfig):
        raise LearnedPoseTokenError("config must be a LearnedPoseTokenConfig.")
    return LearnedPoseTokenizer(
        input_dim=input_dim,
        hidden_dim=config.tokenizer.hidden_dim,
        latent_dim=config.tokenizer.latent_dim,
        codebook_size=config.codebook.size,
    )


def compute_tokenizer_losses(
    *,
    output: LearnedPoseTokenizerOutput,
    target: torch.Tensor,
    validity_mask: torch.Tensor,
    commitment_weight: float,
    temporal_granularity: str = "frame",
    window_size: int = 1,
    feature_dim: int | None = None,
    real_frame_mask: torch.Tensor | None = None,
    velocity_loss_weight: float = 0.0,
) -> Mapping[str, torch.Tensor]:
    """Compute masked reconstruction and VQ losses for pose token units."""

    if not isinstance(output, LearnedPoseTokenizerOutput):
        raise LearnedPoseTokenError("output must be LearnedPoseTokenizerOutput.")
    if target.shape != output.reconstructed.shape:
        raise LearnedPoseTokenError(
            "target shape must match reconstructed shape: "
            f"target={tuple(target.shape)}, reconstructed={tuple(output.reconstructed.shape)}."
        )
    if validity_mask.shape != target.shape:
        raise LearnedPoseTokenError(
            f"validity_mask must have shape {tuple(target.shape)}; got {tuple(validity_mask.shape)}."
        )
    if not torch.isfinite(target).all():
        raise LearnedPoseTokenError("target must contain only finite values.")
    if not torch.isfinite(validity_mask.to(dtype=torch.float32)).all():
        raise LearnedPoseTokenError("validity_mask must contain finite values.")
    if not isinstance(commitment_weight, int | float) or commitment_weight < 0.0:
        raise LearnedPoseTokenError("commitment_weight must be non-negative.")
    if temporal_granularity not in {"frame", "window"}:
        raise LearnedPoseTokenError(
            "tokenizer.temporal_granularity must be one of {'frame', 'window'}."
        )
    _positive_int(window_size, "window_size")
    if velocity_loss_weight < 0.0:
        raise LearnedPoseTokenError("velocity_loss_weight must be non-negative.")
    mask = validity_mask.to(dtype=target.dtype)
    supervised = mask.sum()
    if bool((supervised <= 0).item()):
        raise LearnedPoseTokenError("validity_mask has no supervised pose features.")
    error = (output.reconstructed - target) * mask
    l1 = error.abs().sum() / supervised
    l2 = error.square().sum() / supervised
    reconstruction = l1 + l2
    velocity = target.new_tensor(0.0)
    if (
        temporal_granularity == "window"
        and window_size > 1
        and float(velocity_loss_weight) > 0.0
    ):
        if feature_dim is None:
            if target.shape[1] % window_size != 0:
                raise LearnedPoseTokenError(
                    "window target feature dimension must be divisible by window_size."
                )
            feature_dim = int(target.shape[1] // window_size)
        _positive_int(feature_dim, "feature_dim")
        expected_dim = window_size * feature_dim
        if target.shape[1] != expected_dim:
            raise LearnedPoseTokenError(
                f"window target must have feature dimension {expected_dim}; got {target.shape[1]}."
            )
        if real_frame_mask is None:
            real_mask = torch.ones(
                (target.shape[0], window_size),
                dtype=torch.bool,
                device=target.device,
            )
        else:
            if real_frame_mask.shape != (target.shape[0], window_size):
                raise LearnedPoseTokenError(
                    "real_frame_mask must have shape (batch, window_size)."
                )
            real_mask = real_frame_mask.to(dtype=torch.bool, device=target.device)
        target_window = target.reshape(target.shape[0], window_size, feature_dim)
        reconstructed_window = output.reconstructed.reshape(
            target.shape[0],
            window_size,
            feature_dim,
        )
        feature_mask = validity_mask.reshape(
            target.shape[0],
            window_size,
            feature_dim,
        )
        adjacent_mask = (
            feature_mask[:, 1:]
            & feature_mask[:, :-1]
            & real_mask[:, 1:, None]
            & real_mask[:, :-1, None]
        )
        supervised_velocity = adjacent_mask.to(dtype=target.dtype).sum()
        if bool((supervised_velocity > 0).item()):
            target_velocity = target_window[:, 1:] - target_window[:, :-1]
            reconstructed_velocity = (
                reconstructed_window[:, 1:] - reconstructed_window[:, :-1]
            )
            velocity_error = (reconstructed_velocity - target_velocity) * adjacent_mask.to(
                dtype=target.dtype
            )
            velocity = (
                velocity_error.abs().sum() + velocity_error.square().sum()
            ) / supervised_velocity
    total = (
        reconstruction
        + output.codebook_loss
        + float(commitment_weight) * output.commitment_loss
        + float(velocity_loss_weight) * velocity
    )
    losses = {
        "reconstruction_loss": reconstruction,
        "masked_l1": l1,
        "masked_l2": l2,
        "velocity_loss": velocity,
        "codebook_loss": output.codebook_loss,
        "commitment_loss": output.commitment_loss,
        "total_loss": total,
    }
    if any(not torch.isfinite(value).all() for value in losses.values()):
        raise LearnedPoseTokenError("tokenizer losses must be finite.")
    return losses


def _positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LearnedPoseTokenError(f"{name} must be a positive integer.")


__all__ = [
    "LearnedPoseTokenizer",
    "LearnedPoseTokenizerOutput",
    "PoseFrameDecoder",
    "PoseFrameEncoder",
    "VectorQuantizer",
    "build_learned_pose_tokenizer",
    "compute_tokenizer_losses",
]
