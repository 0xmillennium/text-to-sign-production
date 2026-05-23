"""Minimal deterministic text-to-pose-token predictor."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

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
class TextToTokenPredictorOutput:
    token_logits: torch.Tensor
    predicted_lengths: torch.Tensor
    length_values: torch.Tensor


@dataclass(frozen=True, slots=True)
class TextToTokenBatch:
    text_embeddings: torch.Tensor
    target_token_ids: torch.Tensor
    target_lengths: torch.Tensor
    token_mask: torch.Tensor


class TemporalPositionEncoding(nn.Module):
    """Learned position features for a fixed maximum token horizon."""

    def __init__(self, max_positions: int, position_dim: int) -> None:
        super().__init__()
        _positive_int(max_positions, "max_positions")
        _positive_int(position_dim, "position_dim")
        self.max_positions = max_positions
        self.position_dim = position_dim
        self.embedding = nn.Embedding(max_positions, position_dim)

    def forward(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        _positive_int(batch_size, "batch_size")
        positions = torch.arange(self.max_positions, device=device, dtype=torch.long)
        encoded = self.embedding(positions)
        return encoded.unsqueeze(0).expand(batch_size, -1, -1)


class TextToTokenPredictor(nn.Module):
    """Non-autoregressive temporal MLP over deterministic text embeddings."""

    def __init__(
        self,
        *,
        text_embedding_dim: int,
        codebook_size: int,
        max_positions: int,
        hidden_dim: int,
        position_dim: int,
        dropout: float,
        min_generated_tokens: int,
        max_generated_tokens: int,
    ) -> None:
        super().__init__()
        for value, name in (
            (text_embedding_dim, "text_embedding_dim"),
            (codebook_size, "codebook_size"),
            (max_positions, "max_positions"),
            (hidden_dim, "hidden_dim"),
            (position_dim, "position_dim"),
            (min_generated_tokens, "min_generated_tokens"),
            (max_generated_tokens, "max_generated_tokens"),
        ):
            _positive_int(value, name)
        if codebook_size <= 1:
            raise LearnedPoseTokenError("codebook_size must be greater than 1.")
        if min_generated_tokens > max_generated_tokens:
            raise LearnedPoseTokenError("min_generated_tokens cannot exceed max_generated_tokens.")
        if not isinstance(dropout, int | float) or isinstance(dropout, bool) or not 0 <= float(dropout) < 1:
            raise LearnedPoseTokenError("dropout must be in [0, 1).")
        self.text_embedding_dim = text_embedding_dim
        self.codebook_size = codebook_size
        self.max_positions = max_positions
        self.min_generated_tokens = min_generated_tokens
        self.max_generated_tokens = max_generated_tokens
        self.positions = TemporalPositionEncoding(max_positions, position_dim)
        self.token_head = nn.Sequential(
            nn.Linear(text_embedding_dim + position_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, codebook_size),
        )
        self.length_head = nn.Sequential(
            nn.Linear(text_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text_embeddings: torch.Tensor) -> TextToTokenPredictorOutput:
        if text_embeddings.ndim != 2 or text_embeddings.shape[1] != self.text_embedding_dim:
            raise LearnedPoseTokenError(
                "text_embeddings must have shape "
                f"(batch, {self.text_embedding_dim}); got {tuple(text_embeddings.shape)}."
            )
        if not torch.isfinite(text_embeddings).all():
            raise LearnedPoseTokenError("text_embeddings must contain only finite values.")
        batch_size = int(text_embeddings.shape[0])
        pos = self.positions(batch_size, device=text_embeddings.device)
        expanded = text_embeddings.unsqueeze(1).expand(-1, self.max_positions, -1)
        token_logits = self.token_head(torch.cat((expanded, pos), dim=-1))
        length_values = self.length_head(text_embeddings).squeeze(-1)
        predicted_lengths = clamp_predicted_lengths(
            torch.round(length_values).to(dtype=torch.long),
            min_tokens=self.min_generated_tokens,
            max_tokens=self.max_generated_tokens,
        )
        return TextToTokenPredictorOutput(
            token_logits=token_logits,
            predicted_lengths=predicted_lengths,
            length_values=length_values,
        )

    def generate_token_ids(self, text_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self(text_embeddings)
        token_ids = torch.argmax(output.token_logits, dim=-1)
        return token_ids, output.predicted_lengths


def build_text_to_token_predictor(
    *,
    text_embedding_dim: int,
    codebook_size: int,
    config: LearnedPoseTokenConfig,
    max_positions: int | None = None,
) -> TextToTokenPredictor:
    """Build the Phase 6 non-autoregressive predictor."""

    if not isinstance(config, LearnedPoseTokenConfig):
        raise LearnedPoseTokenError("config must be a LearnedPoseTokenConfig.")
    horizon = max_positions or config.text_to_token.max_generated_tokens
    if horizon is None:
        horizon = config.generation.max_generation_tokens
    if horizon is None:
        horizon = max(config.text_to_token.min_generated_tokens, 1)
    _positive_int(horizon, "max_positions")
    max_generation = config.text_to_token.max_generated_tokens or horizon
    if config.generation.max_generation_tokens is not None:
        max_generation = min(max_generation, config.generation.max_generation_tokens)
    return TextToTokenPredictor(
        text_embedding_dim=text_embedding_dim,
        codebook_size=codebook_size,
        max_positions=int(horizon),
        hidden_dim=config.text_to_token.hidden_dim,
        position_dim=config.text_to_token.position_dim,
        dropout=config.text_to_token.dropout,
        min_generated_tokens=config.text_to_token.min_generated_tokens,
        max_generated_tokens=max_generation,
    )


def compute_text_to_token_losses(
    *,
    output: TextToTokenPredictorOutput,
    target_token_ids: torch.Tensor,
    target_lengths: torch.Tensor,
    token_mask: torch.Tensor,
    token_loss_weight: float,
    length_loss_weight: float,
) -> Mapping[str, torch.Tensor]:
    """Compute masked token CE and finite length regression loss."""

    if not isinstance(output, TextToTokenPredictorOutput):
        raise LearnedPoseTokenError("output must be TextToTokenPredictorOutput.")
    if output.token_logits.ndim != 3:
        raise LearnedPoseTokenError("token_logits must be 3D.")
    if target_token_ids.shape != output.token_logits.shape[:2]:
        raise LearnedPoseTokenError("target_token_ids must match token logits batch/position shape.")
    if token_mask.shape != target_token_ids.shape:
        raise LearnedPoseTokenError("token_mask must match target_token_ids shape.")
    if target_lengths.shape != output.length_values.shape:
        raise LearnedPoseTokenError("target_lengths must match predicted length shape.")
    if token_loss_weight <= 0.0 or length_loss_weight < 0.0:
        raise LearnedPoseTokenError("loss weights are invalid.")
    mask = token_mask.to(dtype=torch.bool)
    if not bool(mask.any().item()):
        raise LearnedPoseTokenError("token_mask must contain at least one supervised token.")
    flat_logits = output.token_logits.reshape(-1, output.token_logits.shape[-1])
    flat_targets = target_token_ids.reshape(-1)
    flat_mask = mask.reshape(-1)
    token_loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])
    length_loss = F.mse_loss(output.length_values, target_lengths.to(dtype=output.length_values.dtype))
    total = float(token_loss_weight) * token_loss + float(length_loss_weight) * length_loss
    losses = {
        "token_loss": token_loss,
        "length_loss": length_loss,
        "total_loss": total,
    }
    if any(not torch.isfinite(value).all() for value in losses.values()):
        raise LearnedPoseTokenError("text-to-token losses must be finite.")
    return losses


def clamp_predicted_lengths(
    lengths: torch.Tensor,
    *,
    min_tokens: int,
    max_tokens: int,
) -> torch.Tensor:
    _positive_int(min_tokens, "min_tokens")
    _positive_int(max_tokens, "max_tokens")
    if min_tokens > max_tokens:
        raise LearnedPoseTokenError("min_tokens cannot exceed max_tokens.")
    return torch.clamp(lengths.to(dtype=torch.long), min=min_tokens, max=max_tokens)


def _positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LearnedPoseTokenError(f"{name} must be a positive integer.")


__all__ = [
    "TemporalPositionEncoding",
    "TextToTokenBatch",
    "TextToTokenPredictor",
    "TextToTokenPredictorOutput",
    "build_text_to_token_predictor",
    "clamp_predicted_lengths",
    "compute_text_to_token_losses",
]
