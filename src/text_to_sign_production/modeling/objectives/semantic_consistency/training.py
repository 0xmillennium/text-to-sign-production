"""Differentiable semantic-consistency auxiliary training loss."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.backbones.bfh_masks import channel_feature_slice
from text_to_sign_production.modeling.backbones.bfh_vectorization import BfhTensorLayout
from text_to_sign_production.modeling.data.temporal_windows import (
    TemporalWindowSpec,
    temporal_window_starts,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.config import (
    SemanticConsistencyObjectiveConfig,
    load_semantic_consistency_config,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.errors import (
    SemanticConsistencyError,
)
from text_to_sign_production.modeling.research import ModelKey, ObjectiveKey


@dataclass(frozen=True, slots=True)
class SemanticTrainingLossResult:
    semantic_loss: torch.Tensor
    weighted_semantic_loss: torch.Tensor
    cosine_similarity: torch.Tensor
    cosine_distance: torch.Tensor
    records_count: int
    loss_weight: float


def load_semantic_training_objective_for_request(
    *,
    request,
    model_key: ModelKey,
) -> SemanticConsistencyObjectiveConfig | None:
    """Load requested semantic training config from the provider request snapshot path."""

    if ObjectiveKey.SEMANTIC_CONSISTENCY not in request.auxiliary_objectives:
        return None
    path = request.objective_config_paths.get(ObjectiveKey.SEMANTIC_CONSISTENCY)
    if path is None:
        raise SemanticConsistencyError(
            "semantic_consistency was requested but no objective config snapshot path was provided "
            "to the provider request."
        )
    config = load_semantic_consistency_config(Path(path))
    if not config.training_objective.enabled:
        raise SemanticConsistencyError(
            "semantic_consistency was requested as an auxiliary objective, but "
            "training_objective.enabled is false."
        )
    resolved_model = ModelKey(model_key)
    if resolved_model not in config.training_objective.supported_models:
        raise SemanticConsistencyError(
            f"semantic_consistency training objective does not support {resolved_model.value}."
        )
    return config


def compute_semantic_training_loss(
    *,
    text_embeddings: torch.Tensor,
    pose_values: torch.Tensor,
    pose_validity_mask: torch.Tensor,
    frame_mask: torch.Tensor,
    layout: BfhTensorLayout,
    config: SemanticConsistencyObjectiveConfig,
) -> SemanticTrainingLossResult:
    """Compute fixed-projection BFH-statistics cosine distance with gradients to pose."""

    if not config.training_objective.enabled:
        raise SemanticConsistencyError("semantic training objective must be enabled.")
    _validate_inputs(text_embeddings, pose_values, pose_validity_mask, frame_mask, layout)
    text = text_embeddings.detach() if config.training_objective.detach_text_embedding else text_embeddings
    stats = compute_differentiable_bfh_statistics(
        pose_values=pose_values,
        pose_validity_mask=pose_validity_mask,
        frame_mask=frame_mask,
        layout=layout,
        config=config,
    )
    projection = _fixed_projection(
        input_dim=stats.shape[-1],
        output_dim=text.shape[-1],
        seed=config.training_objective.pose_projection.projection_seed,
        device=pose_values.device,
        dtype=pose_values.dtype,
    )
    pose_embedding = stats @ projection
    text = text.to(device=pose_embedding.device, dtype=pose_embedding.dtype)
    if not torch.isfinite(text).all() or not torch.isfinite(pose_embedding).all():
        raise SemanticConsistencyError("semantic training loss is not finite.")
    text_norm = torch.linalg.vector_norm(text, dim=-1)
    pose_norm = torch.linalg.vector_norm(pose_embedding, dim=-1)
    if torch.any(text_norm <= 0.0) or torch.any(pose_norm <= 0.0):
        raise SemanticConsistencyError(
            "semantic training cosine alignment cannot use zero-norm text or pose embeddings."
        )
    cosine = torch.nn.functional.cosine_similarity(text, pose_embedding, dim=-1)
    distance = 1.0 - cosine
    loss = distance.mean()
    weighted = float(config.training_objective.alignment.loss_weight) * loss
    if not torch.isfinite(weighted):
        raise SemanticConsistencyError("semantic training loss is not finite.")
    return SemanticTrainingLossResult(
        semantic_loss=loss,
        weighted_semantic_loss=weighted,
        cosine_similarity=cosine.mean(),
        cosine_distance=distance.mean(),
        records_count=int(text.shape[0]),
        loss_weight=float(config.training_objective.alignment.loss_weight),
    )


def compute_differentiable_bfh_statistics(
    *,
    pose_values: torch.Tensor,
    pose_validity_mask: torch.Tensor,
    frame_mask: torch.Tensor,
    layout: BfhTensorLayout,
    config: SemanticConsistencyObjectiveConfig,
) -> torch.Tensor:
    """Return per-channel mean/std/velocity/validity statistics via torch ops."""

    features: list[torch.Tensor] = []
    valid_frame = frame_mask.unsqueeze(-1)
    full_valid = pose_validity_mask.to(torch.bool) & valid_frame
    for channel in config.training_objective.pose_projection.channel_groups:
        feature_slice = channel_feature_slice(layout, PoseChannel(channel))
        values = pose_values[:, :, feature_slice]
        valid = full_valid[:, :, feature_slice]
        count = valid.sum(dim=(1, 2)).to(dtype=values.dtype)
        if torch.any(count <= 0):
            raise SemanticConsistencyError(
                "semantic training loss requires at least one valid pose observation per record."
            )
        denom = count.clamp_min(1.0)
        masked = values * valid.to(dtype=values.dtype)
        mean = masked.sum(dim=(1, 2)) / denom
        centered = (values - mean[:, None, None]) * valid.to(dtype=values.dtype)
        std = torch.sqrt(torch.square(centered).sum(dim=(1, 2)) / denom + 1.0e-8)
        features.extend((mean, std))
        if config.training_objective.pose_projection.include_velocity_statistics:
            adjacent_valid = valid[:, 1:, :] & valid[:, :-1, :]
            adjacent_count = adjacent_valid.sum(dim=(1, 2)).to(dtype=values.dtype)
            if torch.any(adjacent_count > 0):
                velocity = (values[:, 1:, :] - values[:, :-1, :]) * adjacent_valid.to(
                    dtype=values.dtype
                )
                velocity_mean = velocity.abs().sum(dim=(1, 2)) / adjacent_count.clamp_min(1.0)
            else:
                velocity_mean = values.sum(dim=(1, 2)) * 0.0
            features.append(velocity_mean)
        if config.training_objective.pose_projection.include_validity_statistics:
            features.append(valid.to(dtype=values.dtype).mean(dim=(1, 2)))
    return torch.stack(features, dim=-1)


def merge_overlapping_windows_torch(
    windows: torch.Tensor,
    *,
    frame_count: int,
    window_size: int,
    stride: int,
    start_indices: Sequence[int] | None = None,
    source_frame_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiably average overlapping `(W, window, D)` windows into `(T, D)`."""

    if windows.ndim != 3:
        raise SemanticConsistencyError("semantic windows must have shape (windows, window_size, dim).")
    _require_positive_int(frame_count, "frame_count")
    _require_positive_int(window_size, "window_size")
    _require_positive_int(stride, "stride")
    if windows.shape[1] != window_size:
        raise SemanticConsistencyError("semantic window_size does not match decoded windows.")
    if start_indices is not None and source_frame_indices is not None:
        raise SemanticConsistencyError(
            "provide either start_indices or source_frame_indices, not both."
        )
    num_windows = int(windows.shape[0])
    source_indices = _semantic_window_source_indices(
        num_windows=num_windows,
        frame_count=frame_count,
        window_size=window_size,
        stride=stride,
        start_indices=start_indices,
        source_frame_indices=source_frame_indices,
        device=windows.device,
    )
    result = windows.new_zeros((frame_count, windows.shape[-1]))
    counts = windows.new_zeros((frame_count, 1))
    for index in range(num_windows):
        for offset in range(window_size):
            source_index = int(source_indices[index, offset].item())
            if source_index < 0:
                continue
            result[source_index] = result[source_index] + windows[index, offset]
            counts[source_index] = counts[source_index] + 1.0
    if torch.any(counts <= 0):
        raise SemanticConsistencyError("semantic window merge left frames without decoded values.")
    return result / counts


def _semantic_window_source_indices(
    *,
    num_windows: int,
    frame_count: int,
    window_size: int,
    stride: int,
    start_indices: Sequence[int] | None,
    source_frame_indices: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    if source_frame_indices is not None:
        if source_frame_indices.shape != (num_windows, window_size):
            raise SemanticConsistencyError(
                "semantic source_frame_indices must have shape (windows, window_size)."
            )
        integer_dtypes = {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        if source_frame_indices.dtype not in integer_dtypes:
            raise SemanticConsistencyError(
                "semantic source_frame_indices must contain integer frame indices."
            )
        source_indices = source_frame_indices.to(device=device, dtype=torch.long)
        if torch.any((source_indices < -1) | (source_indices >= frame_count)):
            raise SemanticConsistencyError(
                "semantic source_frame_indices contain an index outside [0, frame_count)."
            )
        return source_indices
    if start_indices is None:
        starts = _canonical_semantic_window_starts(
            frame_count=frame_count,
            window_size=window_size,
            stride=stride,
        )
        expected_label = "temporal_window_starts"
    else:
        starts = tuple(_validate_start_index(value) for value in start_indices)
        if any(start < 0 or start >= frame_count for start in starts):
            raise SemanticConsistencyError(
                "semantic start_indices contain an index outside [0, frame_count)."
            )
        expected_label = "start_indices"
    if num_windows != len(starts):
        raise SemanticConsistencyError(
            f"semantic window count={num_windows} does not match expected {expected_label} "
            f"count={len(starts)} for frame_count={frame_count}, window_size={window_size}, "
            f"stride={stride}."
        )
    indices = torch.full((num_windows, window_size), -1, dtype=torch.long, device=device)
    for index, start in enumerate(starts):
        for offset in range(window_size):
            source_index = start + offset
            if 0 <= source_index < frame_count:
                indices[index, offset] = source_index
    return indices


def _canonical_semantic_window_starts(
    *,
    frame_count: int,
    window_size: int,
    stride: int,
) -> tuple[int, ...]:
    if window_size == 1 and stride == 1:
        spec = TemporalWindowSpec.frame()
    else:
        spec = TemporalWindowSpec.window(window_size=window_size, stride=stride)
    return temporal_window_starts(frame_count=frame_count, spec=spec)


def _validate_start_index(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SemanticConsistencyError("semantic start_indices must contain integer frame indices.")
    return int(value)


def _require_positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SemanticConsistencyError(f"semantic {name} must be a positive integer.")


def _fixed_projection(
    *,
    input_dim: int,
    output_dim: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn((input_dim, output_dim), generator=generator, dtype=torch.float32)
    matrix = matrix / torch.sqrt(torch.tensor(float(input_dim), dtype=torch.float32))
    return matrix.to(device=device, dtype=dtype)


def _validate_inputs(
    text_embeddings: torch.Tensor,
    pose_values: torch.Tensor,
    pose_validity_mask: torch.Tensor,
    frame_mask: torch.Tensor,
    layout: BfhTensorLayout,
) -> None:
    if text_embeddings.ndim != 2:
        raise SemanticConsistencyError("semantic text_embeddings must have shape (batch, embedding_dim).")
    if pose_values.ndim != 3 or pose_values.shape[-1] != layout.total_feature_dim:
        raise SemanticConsistencyError(
            "semantic pose_values must have shape (batch, frames, total_feature_dim)."
        )
    if pose_values.shape[:2] != text_embeddings.shape[:1] + pose_values.shape[1:2]:
        raise SemanticConsistencyError(
            "semantic pose_values must have shape (batch, frames, total_feature_dim)."
        )
    if pose_validity_mask.shape != pose_values.shape:
        raise SemanticConsistencyError("semantic pose_validity_mask must match pose_values shape.")
    if frame_mask.shape != pose_values.shape[:2]:
        raise SemanticConsistencyError("semantic frame_mask must have shape (batch, frames).")
    valid = pose_validity_mask.to(torch.bool) & frame_mask.unsqueeze(-1).to(torch.bool)
    if not torch.isfinite(pose_values[valid]).all():
        raise SemanticConsistencyError("semantic training loss is not finite.")
    if not torch.isfinite(text_embeddings).all():
        raise SemanticConsistencyError("semantic training loss is not finite.")


__all__ = [
    "SemanticTrainingLossResult",
    "compute_differentiable_bfh_statistics",
    "compute_semantic_training_loss",
    "load_semantic_training_objective_for_request",
    "merge_overlapping_windows_torch",
]
