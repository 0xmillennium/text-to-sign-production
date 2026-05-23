"""Channel-aware reconstruction loss records and weighting policy."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import torch

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.backbones.bfh_masks import channel_feature_slice
from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    BfhVectorizedPose,
    flatten_bfh_vectorized_pose,
)
from text_to_sign_production.modeling.candidates.articulator_aware.config import (
    ChannelLossWeightingConfig,
)
from text_to_sign_production.modeling.candidates.articulator_aware.errors import (
    ArticulatorAwareError,
)
from text_to_sign_production.modeling.candidates.articulator_aware.partitions import (
    ArticulatorChannelPartitionPolicy,
)

CHANNEL_LOSS_WEIGHTING_SCHEMA_VERSION = "t2sp-channel-loss-weighting-v1"
CHANNEL_LOSS_RECORD_SCHEMA_VERSION = "t2sp-channel-loss-record-v1"


@dataclass(frozen=True, slots=True)
class ChannelLossWeightingPolicy:
    schema_version: str
    policy: str
    normalize_weights: bool
    channel_weights: Mapping[PoseChannel, float] = field(default_factory=lambda: MappingProxyType({}))
    normalized_channel_weights: Mapping[PoseChannel, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    velocity_weight: float = 0.0
    symmetry_weight: float = 0.0
    cross_channel_consistency_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.schema_version != CHANNEL_LOSS_WEIGHTING_SCHEMA_VERSION:
            raise ArticulatorAwareError("channel loss weighting schema_version is unsupported.")
        if self.policy != "static_channel_weights":
            raise ArticulatorAwareError("channel loss weighting policy must be static_channel_weights.")
        if not isinstance(self.normalize_weights, bool):
            raise ArticulatorAwareError("normalize_weights must be a boolean.")
        weights = _channel_weight_mapping(self.channel_weights, "channel_weights")
        normalized = _channel_weight_mapping(
            self.normalized_channel_weights,
            "normalized_channel_weights",
        )
        raw_total = sum(weights.values())
        if not math.isfinite(raw_total) or raw_total <= 0.0:
            raise ArticulatorAwareError("channel_weights must have a positive finite sum.")
        expected = (
            {channel: value / raw_total for channel, value in weights.items()}
            if self.normalize_weights
            else weights
        )
        for channel, value in expected.items():
            if not math.isclose(
                normalized[channel],
                value,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ArticulatorAwareError(
                    "normalized_channel_weights must match channel_weights and "
                    f"normalize_weights for channel {channel.value!r}."
                )
        for value, name in (
            (self.velocity_weight, "velocity_weight"),
            (self.symmetry_weight, "symmetry_weight"),
            (self.cross_channel_consistency_weight, "cross_channel_consistency_weight"),
        ):
            _require_non_negative(value, name)
        object.__setattr__(self, "channel_weights", MappingProxyType(weights))
        object.__setattr__(self, "normalized_channel_weights", MappingProxyType(normalized))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy,
            "normalize_weights": self.normalize_weights,
            "channel_weights": _channel_mapping_to_dict(self.channel_weights),
            "normalized_channel_weights": _channel_mapping_to_dict(
                self.normalized_channel_weights
            ),
            "velocity_weight": float(self.velocity_weight),
            "symmetry_weight": float(self.symmetry_weight),
            "cross_channel_consistency_weight": float(self.cross_channel_consistency_weight),
        }


@dataclass(frozen=True, slots=True)
class ChannelLossRecord:
    schema_version: str
    sample_id: str
    split: SampleSplit
    channel: PoseChannel
    loss_name: str
    value: float | None
    valid_observation_count: int
    skipped: bool
    reason: str | None

    def __post_init__(self) -> None:
        if self.schema_version != CHANNEL_LOSS_RECORD_SCHEMA_VERSION:
            raise ArticulatorAwareError("channel loss record schema_version is unsupported.")
        _require_text(self.sample_id, "sample_id")
        object.__setattr__(self, "split", SampleSplit(self.split))
        object.__setattr__(self, "channel", PoseChannel(self.channel))
        _require_text(self.loss_name, "loss_name")
        if self.value is not None:
            if not isinstance(self.value, int | float) or isinstance(self.value, bool):
                raise ArticulatorAwareError("channel loss record value must be numeric or null.")
            if not math.isfinite(float(self.value)):
                raise ArticulatorAwareError("channel loss record value must be finite.")
            object.__setattr__(self, "value", float(self.value))
        if not isinstance(self.valid_observation_count, int) or isinstance(self.valid_observation_count, bool) or self.valid_observation_count < 0:
            raise ArticulatorAwareError("valid_observation_count must be a non-negative integer.")
        if not isinstance(self.skipped, bool):
            raise ArticulatorAwareError("skipped must be a boolean.")
        if self.skipped:
            if self.value is not None:
                raise ArticulatorAwareError("skipped channel loss must use value=None, not fake zero.")
            _require_text(self.reason, "reason")
        elif self.value is None:
            raise ArticulatorAwareError("computed channel loss must have a finite value.")
        elif self.reason is not None:
            _require_text(self.reason, "reason")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "split": self.split.value,
            "channel": self.channel.value,
            "loss_name": self.loss_name,
            "value": self.value,
            "valid_observation_count": self.valid_observation_count,
            "skipped": self.skipped,
            "reason": self.reason,
        }


def build_channel_loss_weighting_policy(
    config: ChannelLossWeightingConfig,
) -> ChannelLossWeightingPolicy:
    """Build deterministic static channel weights."""

    if not isinstance(config, ChannelLossWeightingConfig):
        raise ArticulatorAwareError("config must be ChannelLossWeightingConfig.")
    total = sum(config.channel_weights.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ArticulatorAwareError("channel_weights must have a positive finite sum.")
    normalized = (
        {channel: weight / total for channel, weight in config.channel_weights.items()}
        if config.normalize_weights
        else dict(config.channel_weights)
    )
    return ChannelLossWeightingPolicy(
        schema_version=CHANNEL_LOSS_WEIGHTING_SCHEMA_VERSION,
        policy=config.policy,
        normalize_weights=config.normalize_weights,
        channel_weights=config.channel_weights,
        normalized_channel_weights=normalized,
        velocity_weight=config.velocity_weight,
        symmetry_weight=config.symmetry_weight,
        cross_channel_consistency_weight=config.cross_channel_consistency_weight,
    )


def compute_channel_weighted_reconstruction_loss_numpy(
    *,
    predicted: BfhVectorizedPose,
    target: BfhVectorizedPose,
    partition_policy: ArticulatorChannelPartitionPolicy,
    weighting: ChannelLossWeightingPolicy,
    split: SampleSplit,
    sample_id: str | None = None,
) -> tuple[float, tuple[ChannelLossRecord, ...]]:
    """Compute target-valid channel MSE without converting skipped channels to zero."""

    _require_compatible(predicted, target, partition_policy)
    if not isinstance(weighting, ChannelLossWeightingPolicy):
        raise ArticulatorAwareError("weighting must be ChannelLossWeightingPolicy.")
    resolved_sample_id = _loss_sample_id(predicted, target, sample_id)
    resolved_split = SampleSplit(split)
    predicted_flat = flatten_bfh_vectorized_pose(predicted)
    target_flat = flatten_bfh_vectorized_pose(target)
    target_feature_validity = np.repeat(
        target.validity_mask & target.frame_validity_mask[:, None],
        target.layout.coordinate_dimensions,
        axis=1,
    )
    records: list[ChannelLossRecord] = []
    active_values: list[tuple[PoseChannel, float]] = []
    for channel in partition_policy.primary_channels:
        feature_slice = channel_feature_slice(target.layout, channel)
        channel_validity = target_feature_validity[:, feature_slice]
        valid_count = int(np.count_nonzero(channel_validity))
        if valid_count == 0:
            records.append(
                ChannelLossRecord(
                    schema_version=CHANNEL_LOSS_RECORD_SCHEMA_VERSION,
                    sample_id=resolved_sample_id,
                    split=resolved_split,
                    channel=channel,
                    loss_name="masked_mse",
                    value=None,
                    valid_observation_count=0,
                    skipped=True,
                    reason=f"{channel.value} has no valid target observations; channel loss skipped.",
                )
            )
            continue
        delta = predicted_flat[:, feature_slice] - target_flat[:, feature_slice]
        value = float(np.mean(np.square(delta[channel_validity], dtype=np.float32)))
        if not math.isfinite(value):
            raise ArticulatorAwareError(f"{channel.value} channel loss is not finite.")
        records.append(
            ChannelLossRecord(
                schema_version=CHANNEL_LOSS_RECORD_SCHEMA_VERSION,
                sample_id=resolved_sample_id,
                split=resolved_split,
                channel=channel,
                loss_name="masked_mse",
                value=value,
                valid_observation_count=valid_count,
                skipped=False,
                reason=None,
            )
        )
        active_values.append((channel, value))
    if not active_values:
        raise ArticulatorAwareError("no valid channels remain for reconstruction loss.")
    active_weight_sum = sum(weighting.normalized_channel_weights[channel] for channel, _ in active_values)
    if active_weight_sum <= 0.0 or not math.isfinite(active_weight_sum):
        raise ArticulatorAwareError("valid channel weights must have a positive finite sum.")
    total = sum(
        weighting.normalized_channel_weights[channel] / active_weight_sum * value
        for channel, value in active_values
    )
    if not math.isfinite(total):
        raise ArticulatorAwareError("weighted channel reconstruction loss is not finite.")
    return float(total), tuple(records)


def compute_channel_weighted_sequence_reconstruction_loss_torch(
    *,
    predicted: torch.Tensor,
    target: torch.Tensor,
    channel_masks: Mapping[PoseChannel, torch.Tensor],
    weighting: ChannelLossWeightingPolicy,
) -> tuple[torch.Tensor, Mapping[PoseChannel, torch.Tensor | None]]:
    """Compute masked channel-weighted sequence MSE for `(batch, frames, features)`."""

    _require_sequence_tensors(predicted, target)
    if not isinstance(weighting, ChannelLossWeightingPolicy):
        raise ArticulatorAwareError("weighting must be ChannelLossWeightingPolicy.")
    losses: dict[PoseChannel, torch.Tensor | None] = {}
    active: list[tuple[PoseChannel, torch.Tensor]] = []
    for channel in weighting.normalized_channel_weights:
        mask = _sequence_mask(channel_masks, channel, predicted.shape)
        if not torch.any(mask):
            losses[channel] = None
            continue
        value = torch.mean(torch.square(predicted[mask] - target[mask]))
        if not torch.isfinite(value):
            raise ArticulatorAwareError(f"{channel.value} sequence reconstruction loss is not finite.")
        losses[channel] = value
        active.append((channel, value))
    if not active:
        raise ArticulatorAwareError("no valid channels remain for sequence reconstruction loss.")
    weight_sum = sum(weighting.normalized_channel_weights[channel] for channel, _ in active)
    if weight_sum <= 0.0 or not math.isfinite(weight_sum):
        raise ArticulatorAwareError("valid channel weights must have a positive finite sum.")
    total = sum(
        value * (weighting.normalized_channel_weights[channel] / weight_sum)
        for channel, value in active
    )
    if not torch.isfinite(total):
        raise ArticulatorAwareError("weighted sequence reconstruction loss is not finite.")
    return total, losses


def compute_channel_velocity_loss_torch(
    *,
    predicted: torch.Tensor,
    target: torch.Tensor,
    channel_masks: Mapping[PoseChannel, torch.Tensor],
    weighting: ChannelLossWeightingPolicy,
) -> torch.Tensor:
    """Compare predicted and target adjacent-frame deltas under adjacent valid masks."""

    _require_sequence_tensors(predicted, target)
    terms: list[torch.Tensor] = []
    active_weights: list[float] = []
    for channel in weighting.normalized_channel_weights:
        mask = _adjacent_channel_mask(channel_masks, channel, predicted.shape)
        if not torch.any(mask):
            continue
        pred_delta = predicted[:, 1:, :] - predicted[:, :-1, :]
        target_delta = target[:, 1:, :] - target[:, :-1, :]
        value = torch.mean(torch.square(pred_delta[mask] - target_delta[mask]))
        if not torch.isfinite(value):
            raise ArticulatorAwareError(f"{channel.value} velocity loss is not finite.")
        terms.append(value)
        active_weights.append(weighting.normalized_channel_weights[channel])
    return _weighted_or_zero(
        terms,
        active_weights,
        reference=predicted,
        empty_reason="velocity loss has no valid adjacent observations; contribution is skipped for this batch.",
    )


def compute_hand_motion_coordination_loss_torch(
    *,
    predicted: torch.Tensor,
    target: torch.Tensor,
    channel_masks: Mapping[PoseChannel, torch.Tensor],
) -> torch.Tensor:
    """Match target left/right hand velocity-magnitude relationship."""

    _require_sequence_tensors(predicted, target)
    left = _channel_velocity_magnitude(
        predicted,
        target,
        channel_masks,
        PoseChannel.LEFT_HAND,
    )
    right = _channel_velocity_magnitude(
        predicted,
        target,
        channel_masks,
        PoseChannel.RIGHT_HAND,
    )
    pred_left, target_left, left_mask = left
    pred_right, target_right, right_mask = right
    valid = left_mask & right_mask
    if not torch.any(valid):
        return predicted.sum() * 0.0
    pred_relation = pred_left[valid] - pred_right[valid]
    target_relation = target_left[valid] - target_right[valid]
    value = torch.mean(torch.square(pred_relation - target_relation))
    if not torch.isfinite(value):
        raise ArticulatorAwareError("hand motion coordination loss is not finite.")
    return value


def compute_cross_channel_temporal_consistency_loss_torch(
    *,
    predicted: torch.Tensor,
    target: torch.Tensor,
    channel_masks: Mapping[PoseChannel, torch.Tensor],
    channels: tuple[PoseChannel, ...],
) -> torch.Tensor:
    """Match target channel velocity-magnitude distribution over time."""

    _require_sequence_tensors(predicted, target)
    pred_magnitudes: list[torch.Tensor] = []
    target_magnitudes: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    for channel in channels:
        pred_mag, target_mag, valid = _channel_velocity_magnitude(
            predicted,
            target,
            channel_masks,
            channel,
        )
        pred_magnitudes.append(pred_mag)
        target_magnitudes.append(target_mag)
        valid_masks.append(valid)
    valid_all = torch.stack(valid_masks, dim=-1).all(dim=-1)
    if not torch.any(valid_all):
        return predicted.sum() * 0.0
    pred_stack = torch.stack(pred_magnitudes, dim=-1)[valid_all]
    target_stack = torch.stack(target_magnitudes, dim=-1)[valid_all]
    pred_distribution = pred_stack / torch.clamp(pred_stack.sum(dim=-1, keepdim=True), min=1.0e-8)
    target_distribution = target_stack / torch.clamp(target_stack.sum(dim=-1, keepdim=True), min=1.0e-8)
    value = torch.mean(torch.square(pred_distribution - target_distribution))
    if not torch.isfinite(value):
        raise ArticulatorAwareError("cross-channel temporal consistency loss is not finite.")
    return value


def _require_compatible(
    predicted: BfhVectorizedPose,
    target: BfhVectorizedPose,
    policy: ArticulatorChannelPartitionPolicy,
) -> None:
    if not isinstance(predicted, BfhVectorizedPose) or not isinstance(target, BfhVectorizedPose):
        raise ArticulatorAwareError("predicted and target must be BfhVectorizedPose values.")
    if not isinstance(policy, ArticulatorChannelPartitionPolicy):
        raise ArticulatorAwareError("partition_policy must be ArticulatorChannelPartitionPolicy.")
    if predicted.layout != target.layout or target.layout != policy.layout:
        raise ArticulatorAwareError("predicted, target, and partition policy BFH layouts must match.")
    if predicted.values.shape != target.values.shape:
        raise ArticulatorAwareError("predicted and target BFH value shapes must match.")
    if predicted.frame_count != target.frame_count:
        raise ArticulatorAwareError("predicted and target frame counts must match.")


def _require_sequence_tensors(predicted: torch.Tensor, target: torch.Tensor) -> None:
    if (
        not isinstance(predicted, torch.Tensor)
        or not isinstance(target, torch.Tensor)
        or predicted.ndim != 3
        or target.shape != predicted.shape
    ):
        raise ArticulatorAwareError("predicted and target must share shape (batch, frames, feature_dim).")
    if predicted.shape[1] < 1:
        raise ArticulatorAwareError("sequence losses require at least one frame.")


def _sequence_mask(
    channel_masks: Mapping[PoseChannel, torch.Tensor],
    channel: PoseChannel,
    expected_shape: torch.Size,
) -> torch.Tensor:
    mask = channel_masks.get(channel)
    if not isinstance(mask, torch.Tensor) or mask.shape != expected_shape:
        raise ArticulatorAwareError(
            f"channel mask for {channel.value!r} must match predicted sequence shape."
        )
    return mask.to(dtype=torch.bool)


def _adjacent_channel_mask(
    channel_masks: Mapping[PoseChannel, torch.Tensor],
    channel: PoseChannel,
    expected_shape: torch.Size,
) -> torch.Tensor:
    mask = _sequence_mask(channel_masks, channel, expected_shape)
    return mask[:, 1:, :] & mask[:, :-1, :]


def _channel_velocity_magnitude(
    predicted: torch.Tensor,
    target: torch.Tensor,
    channel_masks: Mapping[PoseChannel, torch.Tensor],
    channel: PoseChannel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = _adjacent_channel_mask(channel_masks, channel, predicted.shape)
    pred_delta = predicted[:, 1:, :] - predicted[:, :-1, :]
    target_delta = target[:, 1:, :] - target[:, :-1, :]
    pred_sq = torch.square(pred_delta) * mask.to(dtype=predicted.dtype)
    target_sq = torch.square(target_delta) * mask.to(dtype=target.dtype)
    counts = mask.sum(dim=-1)
    valid = counts > 0
    denom = torch.clamp(counts.to(dtype=predicted.dtype), min=1.0)
    pred_mag = torch.sqrt(pred_sq.sum(dim=-1) / denom + 1.0e-8)
    target_mag = torch.sqrt(target_sq.sum(dim=-1) / denom + 1.0e-8)
    return pred_mag, target_mag, valid


def _weighted_or_zero(
    terms: list[torch.Tensor],
    weights: list[float],
    *,
    reference: torch.Tensor,
    empty_reason: str,
) -> torch.Tensor:
    if not terms:
        _ = empty_reason
        return reference.sum() * 0.0
    weight_sum = sum(weights)
    if weight_sum <= 0.0 or not math.isfinite(weight_sum):
        raise ArticulatorAwareError("valid auxiliary loss weights must have a positive finite sum.")
    value = sum(term * (weight / weight_sum) for term, weight in zip(terms, weights, strict=True))
    if not torch.isfinite(value):
        raise ArticulatorAwareError("weighted auxiliary loss is not finite.")
    return value


def _loss_sample_id(
    predicted: BfhVectorizedPose,
    target: BfhVectorizedPose,
    explicit_sample_id: str | None,
) -> str:
    if predicted.source_sample_id is not None and target.source_sample_id is not None:
        if predicted.source_sample_id != target.source_sample_id:
            raise ArticulatorAwareError("predicted and target source_sample_id values must match.")
    source_sample_id = target.source_sample_id or predicted.source_sample_id
    if explicit_sample_id is not None:
        _require_text(explicit_sample_id, "sample_id")
        if source_sample_id is not None and explicit_sample_id != source_sample_id:
            raise ArticulatorAwareError(
                "explicit sample_id must match predicted/target source_sample_id identity."
            )
        return explicit_sample_id
    sample_id = source_sample_id
    if sample_id is None:
        raise ArticulatorAwareError(
            "cannot build channel loss records without source_sample_id on predicted or target pose."
        )
    return sample_id


def _channel_weight_mapping(value: Mapping[PoseChannel, float], name: str) -> dict[PoseChannel, float]:
    if not isinstance(value, Mapping):
        raise ArticulatorAwareError(f"{name} must be a mapping.")
    result: dict[PoseChannel, float] = {}
    for channel, amount in value.items():
        resolved = PoseChannel(channel)
        if not isinstance(amount, int | float) or isinstance(amount, bool) or not math.isfinite(float(amount)):
            raise ArticulatorAwareError(f"{name}.{resolved.value} must be finite.")
        if float(amount) <= 0.0:
            raise ArticulatorAwareError(f"{name}.{resolved.value} must be > 0.")
        result[resolved] = float(amount)
    expected = (
        PoseChannel.BODY,
        PoseChannel.LEFT_HAND,
        PoseChannel.RIGHT_HAND,
        PoseChannel.FACE,
    )
    if set(result) != set(expected):
        raise ArticulatorAwareError(f"{name} must define exactly body,left_hand,right_hand,face.")
    return {channel: result[channel] for channel in expected}


def _channel_mapping_to_dict(value: Mapping[PoseChannel, float]) -> dict[str, float]:
    return {channel.value: float(value[channel]) for channel in value}


def _require_non_negative(value: object, name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0.0:
        raise ArticulatorAwareError(f"{name} must be a non-negative finite number.")


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ArticulatorAwareError(f"{name} must be non-empty.")


__all__ = [
    "CHANNEL_LOSS_RECORD_SCHEMA_VERSION",
    "CHANNEL_LOSS_WEIGHTING_SCHEMA_VERSION",
    "ChannelLossRecord",
    "ChannelLossWeightingPolicy",
    "build_channel_loss_weighting_policy",
    "compute_channel_velocity_loss_torch",
    "compute_channel_weighted_sequence_reconstruction_loss_torch",
    "compute_channel_weighted_reconstruction_loss_numpy",
    "compute_cross_channel_temporal_consistency_loss_torch",
    "compute_hand_motion_coordination_loss_torch",
]
