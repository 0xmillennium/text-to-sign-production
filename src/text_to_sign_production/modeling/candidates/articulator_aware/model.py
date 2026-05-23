"""Text-conditioned channel-head model for full-BFH articulator generation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import nn

from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.backbones.bfh_masks import channel_feature_slice
from text_to_sign_production.modeling.backbones.bfh_vectorization import default_bfh_tensor_layout
from text_to_sign_production.modeling.backbones.text_encoder import TextEncoderConfig
from text_to_sign_production.modeling.candidates.articulator_aware.errors import (
    ArticulatorAwareError,
)
from text_to_sign_production.modeling.candidates.articulator_aware.partitions import (
    ArticulatorChannelPartitionPolicy,
)

ARTICULATOR_TEXT_EMBEDDING_DIM = 256


def articulator_text_encoder_config() -> TextEncoderConfig:
    """Return the provider's explicit deterministic shared text encoder policy."""

    return TextEncoderConfig(
        encoder_key="deterministic_hash",
        model_name_or_path=None,
        pooling="mean",
        trainable=False,
        max_length=256,
        embedding_dim=ARTICULATOR_TEXT_EMBEDDING_DIM,
        backend="deterministic_hash",
    )


@dataclass(frozen=True, slots=True)
class ArticulatorModelOutput:
    full_pose_values: torch.Tensor
    channel_outputs: Mapping[PoseChannel, torch.Tensor]
    predicted_lengths: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.full_pose_values, torch.Tensor) or self.full_pose_values.ndim != 2:
            raise ArticulatorAwareError("full_pose_values must be a two-dimensional tensor.")
        batch = self.full_pose_values.shape[0]
        if not isinstance(self.predicted_lengths, torch.Tensor) or self.predicted_lengths.shape != (batch,):
            raise ArticulatorAwareError("predicted_lengths must have shape (batch,).")
        if not isinstance(self.channel_outputs, Mapping) or not self.channel_outputs:
            raise ArticulatorAwareError("channel_outputs must be a non-empty mapping.")
        resolved = {PoseChannel(channel): values for channel, values in self.channel_outputs.items()}
        if any(not isinstance(values, torch.Tensor) or values.shape[0] != batch for values in resolved.values()):
            raise ArticulatorAwareError("channel output tensors must share the model batch size.")
        object.__setattr__(self, "channel_outputs", MappingProxyType(resolved))


@dataclass(frozen=True, slots=True)
class ArticulatorSequenceModelOutput:
    full_pose_values: torch.Tensor
    channel_outputs: Mapping[PoseChannel, torch.Tensor]
    predicted_lengths: torch.Tensor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.full_pose_values, torch.Tensor)
            or self.full_pose_values.ndim != 3
        ):
            raise ArticulatorAwareError(
                "factorized temporal output must have shape (batch, frames, total_feature_dim)."
            )
        batch, frames, total_dim = self.full_pose_values.shape
        if frames < 1:
            raise ArticulatorAwareError("factorized temporal output must contain frames.")
        if (
            not isinstance(self.predicted_lengths, torch.Tensor)
            or self.predicted_lengths.shape != (batch,)
        ):
            raise ArticulatorAwareError("predicted_lengths must have shape (batch,).")
        if not isinstance(self.channel_outputs, Mapping) or not self.channel_outputs:
            raise ArticulatorAwareError("channel_outputs must be a non-empty mapping.")
        resolved = {PoseChannel(channel): values for channel, values in self.channel_outputs.items()}
        concatenated_dim = 0
        for channel, values in resolved.items():
            if not isinstance(values, torch.Tensor) or values.ndim != 3:
                raise ArticulatorAwareError(
                    f"{channel.value} channel output must have shape (batch, frames, features)."
                )
            if values.shape[:2] != (batch, frames):
                raise ArticulatorAwareError(
                    "channel output tensors must share the model batch and frame dimensions."
                )
            feature_slice = channel_feature_slice(default_bfh_tensor_layout(), channel)
            expected_dim = feature_slice.stop - feature_slice.start
            if values.shape[2] != expected_dim:
                raise ArticulatorAwareError(
                    f"{channel.value} channel output feature dimension must be {expected_dim}."
                )
            concatenated_dim += int(values.shape[2])
        if concatenated_dim != total_dim:
            raise ArticulatorAwareError(
                "concatenated channel output dimensions must match full_pose_values feature dimension."
            )
        object.__setattr__(self, "channel_outputs", MappingProxyType(resolved))


class ArticulatorLengthPredictor(nn.Module):
    """Positive frame-count head driven only by shared text representation."""

    def __init__(self, *, text_embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if text_embedding_dim <= 0 or hidden_dim <= 0:
            raise ArticulatorAwareError("length predictor dimensions must be positive.")
        self.text_embedding_dim = int(text_embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(text_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        if text_embedding.ndim != 2 or text_embedding.shape[1] != self.text_embedding_dim:
            raise ArticulatorAwareError(
                f"text_embedding must have shape (batch, {self.text_embedding_dim})."
            )
        return torch.nn.functional.softplus(self.net(text_embedding).squeeze(-1)) + 1.0


class ChannelFusionPoseModel(nn.Module):
    """Shared text/position trunk with one output head per canonical BFH channel."""

    def __init__(
        self,
        *,
        partition_policy: ArticulatorChannelPartitionPolicy,
        text_embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        max_positions: int,
        length_predictor: ArticulatorLengthPredictor | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(partition_policy, ArticulatorChannelPartitionPolicy):
            raise ArticulatorAwareError("partition_policy must be articulator policy.")
        if text_embedding_dim <= 0 or hidden_dim <= 0 or max_positions <= 0:
            raise ArticulatorAwareError("model dimensions and max_positions must be positive.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ArticulatorAwareError("dropout must be in [0, 1).")
        self.partition_policy = partition_policy
        self.text_embedding_dim = int(text_embedding_dim)
        self.total_feature_dim = partition_policy.layout.total_feature_dim
        self.max_positions = int(max_positions)
        self.position_embedding = nn.Embedding(self.max_positions, hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.text_embedding_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.channel_heads = nn.ModuleDict(
            {
                channel.value: nn.Linear(
                    hidden_dim,
                    channel_feature_slice(partition_policy.layout, channel).stop
                    - channel_feature_slice(partition_policy.layout, channel).start,
                )
                for channel in partition_policy.primary_channels
            }
        )
        self.length_predictor = length_predictor or ArticulatorLengthPredictor(
            text_embedding_dim=text_embedding_dim,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        text_embedding: torch.Tensor,
        position: torch.Tensor,
    ) -> ArticulatorModelOutput:
        if text_embedding.ndim != 2 or text_embedding.shape[1] != self.text_embedding_dim:
            raise ArticulatorAwareError(
                f"text_embedding must have shape (batch, {self.text_embedding_dim})."
            )
        batch = text_embedding.shape[0]
        if position.shape != (batch,):
            raise ArticulatorAwareError("position must have shape (batch,).")
        if torch.any(position < 0) or torch.any(position >= self.max_positions):
            raise ArticulatorAwareError(
                "position exceeds length.max_positions; raise the configured limit "
                "or provide shorter source sequences."
            )
        hidden = self.trunk(
            torch.cat((text_embedding, self.position_embedding(position.long())), dim=1)
        )
        channel_outputs = {
            channel: self.channel_heads[channel.value](hidden)
            for channel in self.partition_policy.primary_channels
        }
        fused = torch.cat(
            tuple(channel_outputs[channel] for channel in self.partition_policy.primary_channels),
            dim=1,
        )
        if fused.shape != (batch, self.total_feature_dim):
            raise ArticulatorAwareError("channel fusion does not match full BFH feature dimension.")
        predicted_lengths = self.length_predictor(text_embedding)
        return ArticulatorModelOutput(
            full_pose_values=fused,
            channel_outputs=channel_outputs,
            predicted_lengths=predicted_lengths,
        )


class ArticulatorFactorizedTemporalModel(nn.Module):
    """Per-channel temporal GRU branches with gated cross-channel fusion."""

    def __init__(
        self,
        *,
        partition_policy: ArticulatorChannelPartitionPolicy,
        text_embedding_dim: int,
        hidden_dim: int,
        fusion_hidden_dim: int,
        temporal_layers: int,
        dropout: float,
        max_positions: int,
        length_predictor: ArticulatorLengthPredictor | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(partition_policy, ArticulatorChannelPartitionPolicy):
            raise ArticulatorAwareError("partition_policy must be articulator policy.")
        if (
            text_embedding_dim <= 0
            or hidden_dim <= 0
            or fusion_hidden_dim <= 0
            or temporal_layers <= 0
            or max_positions <= 0
        ):
            raise ArticulatorAwareError("factorized temporal model dimensions must be positive.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ArticulatorAwareError("dropout must be in [0, 1).")
        self.partition_policy = partition_policy
        self.text_embedding_dim = int(text_embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.temporal_layers = int(temporal_layers)
        self.total_feature_dim = partition_policy.layout.total_feature_dim
        self.max_positions = int(max_positions)
        self.text_projection = nn.Linear(self.text_embedding_dim, hidden_dim)
        self.position_embedding = nn.Embedding(self.max_positions, hidden_dim)
        self.channel_embeddings = nn.ModuleDict(
            {
                channel.value: nn.Embedding(1, hidden_dim)
                for channel in partition_policy.primary_channels
            }
        )
        self.temporal_branches = nn.ModuleDict(
            {
                channel.value: nn.GRU(
                    input_size=hidden_dim,
                    hidden_size=hidden_dim,
                    num_layers=temporal_layers,
                    batch_first=True,
                    dropout=float(dropout) if temporal_layers > 1 else 0.0,
                )
                for channel in partition_policy.primary_channels
            }
        )
        self.context_projections = nn.ModuleDict(
            {
                channel.value: nn.Sequential(
                    nn.Linear(hidden_dim, fusion_hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(fusion_hidden_dim, hidden_dim),
                )
                for channel in partition_policy.primary_channels
            }
        )
        self.fusion_gates = nn.ModuleDict(
            {
                channel.value: nn.Sequential(
                    nn.Linear(hidden_dim * 2, fusion_hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(fusion_hidden_dim, hidden_dim),
                    nn.Sigmoid(),
                )
                for channel in partition_policy.primary_channels
            }
        )
        self.channel_heads = nn.ModuleDict(
            {
                channel.value: nn.Linear(
                    hidden_dim,
                    channel_feature_slice(partition_policy.layout, channel).stop
                    - channel_feature_slice(partition_policy.layout, channel).start,
                )
                for channel in partition_policy.primary_channels
            }
        )
        self.length_predictor = length_predictor or ArticulatorLengthPredictor(
            text_embedding_dim=text_embedding_dim,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        text_embedding: torch.Tensor,
        positions: torch.Tensor,
    ) -> ArticulatorSequenceModelOutput:
        if text_embedding.ndim != 2 or text_embedding.shape[1] != self.text_embedding_dim:
            raise ArticulatorAwareError(
                f"text_embedding must have shape (batch, {self.text_embedding_dim})."
            )
        if positions.ndim != 2 or positions.shape[0] != text_embedding.shape[0]:
            raise ArticulatorAwareError("positions must have shape (batch, frames).")
        if positions.shape[1] < 1:
            raise ArticulatorAwareError("positions must contain at least one frame.")
        if torch.any(positions < 0) or torch.any(positions >= self.max_positions):
            raise ArticulatorAwareError(
                "positions exceed length.max_positions; raise the configured limit "
                "or provide shorter source sequences."
            )
        batch, frames = positions.shape
        text_hidden = self.text_projection(text_embedding).unsqueeze(1).expand(batch, frames, -1)
        position_hidden = self.position_embedding(positions.long())
        base = text_hidden + position_hidden
        channel_hidden: dict[PoseChannel, torch.Tensor] = {}
        channel_index = torch.zeros((batch, frames), dtype=torch.long, device=positions.device)
        for channel in self.partition_policy.primary_channels:
            channel_input = base + self.channel_embeddings[channel.value](channel_index)
            hidden, _ = self.temporal_branches[channel.value](channel_input)
            channel_hidden[channel] = hidden
        global_context = torch.stack(
            [channel_hidden[channel] for channel in self.partition_policy.primary_channels],
            dim=0,
        ).mean(dim=0)
        channel_outputs: dict[PoseChannel, torch.Tensor] = {}
        for channel in self.partition_policy.primary_channels:
            hidden = channel_hidden[channel]
            gate = self.fusion_gates[channel.value](
                torch.cat((hidden, global_context), dim=-1)
            )
            fused = hidden + gate * self.context_projections[channel.value](global_context)
            channel_outputs[channel] = self.channel_heads[channel.value](fused)
        full = torch.cat(
            tuple(channel_outputs[channel] for channel in self.partition_policy.primary_channels),
            dim=-1,
        )
        if full.shape != (batch, frames, self.total_feature_dim):
            raise ArticulatorAwareError(
                "factorized temporal output must have shape (batch, frames, total_feature_dim)."
            )
        return ArticulatorSequenceModelOutput(
            full_pose_values=full,
            channel_outputs=channel_outputs,
            predicted_lengths=self.length_predictor(text_embedding),
        )


def clamp_articulator_predicted_frame_count(
    predicted_length: float,
    *,
    min_generated_frames: int,
    max_generated_frames: int | None,
    max_positions: int,
) -> int:
    """Round and bound one deterministic generated sequence length."""

    if not math.isfinite(float(predicted_length)):
        raise ArticulatorAwareError("predicted frame count must be finite.")
    result = max(min_generated_frames, int(round(float(predicted_length))))
    if max_generated_frames is not None:
        result = min(result, max_generated_frames)
    if result > max_positions:
        raise ArticulatorAwareError(
            f"predicted generated length {result} exceeds length.max_positions "
            f"{max_positions}; increase the configured bound or retrain the length predictor."
        )
    return result


__all__ = [
    "ARTICULATOR_TEXT_EMBEDDING_DIM",
    "ArticulatorFactorizedTemporalModel",
    "ArticulatorLengthPredictor",
    "ArticulatorModelOutput",
    "ArticulatorSequenceModelOutput",
    "ChannelFusionPoseModel",
    "articulator_text_encoder_config",
    "clamp_articulator_predicted_frame_count",
]
