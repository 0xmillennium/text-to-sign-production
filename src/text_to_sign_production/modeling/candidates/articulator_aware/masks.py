"""Channel mask strategy for canonical BFH vectorized poses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.backbones.bfh_masks import channel_feature_slice
from text_to_sign_production.modeling.backbones.bfh_vectorization import BfhVectorizedPose
from text_to_sign_production.modeling.candidates.articulator_aware.config import (
    ChannelMaskStrategyConfig,
)
from text_to_sign_production.modeling.candidates.articulator_aware.errors import (
    ArticulatorAwareError,
)
from text_to_sign_production.modeling.candidates.articulator_aware.partitions import (
    ArticulatorChannelPartitionPolicy,
)

CHANNEL_MASK_SUMMARY_SCHEMA_VERSION = "t2sp-channel-mask-summary-v1"


@dataclass(frozen=True, slots=True)
class ChannelMaskSummary:
    schema_version: str
    sample_id: str
    split: SampleSplit
    frame_count: int
    channel_valid_counts: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    channel_total_counts: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    channel_valid_fractions: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    all_invalid_channels: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != CHANNEL_MASK_SUMMARY_SCHEMA_VERSION:
            raise ArticulatorAwareError("channel mask summary schema_version is unsupported.")
        _require_text(self.sample_id, "sample_id")
        object.__setattr__(self, "split", SampleSplit(self.split))
        if not isinstance(self.frame_count, int) or isinstance(self.frame_count, bool) or self.frame_count < 1:
            raise ArticulatorAwareError("frame_count must be a positive integer.")
        valid_counts = _int_mapping(self.channel_valid_counts, "channel_valid_counts")
        total_counts = _int_mapping(self.channel_total_counts, "channel_total_counts")
        fractions = _float_mapping(self.channel_valid_fractions, "channel_valid_fractions")
        if set(valid_counts) != set(total_counts) or set(valid_counts) != set(fractions):
            raise ArticulatorAwareError("channel mask summary channel maps must use identical keys.")
        for channel, total in total_counts.items():
            valid = valid_counts[channel]
            if total <= 0:
                raise ArticulatorAwareError(f"channel_total_counts.{channel} must be positive.")
            if valid < 0 or valid > total:
                raise ArticulatorAwareError(f"channel_valid_counts.{channel} is outside [0, total].")
            expected_fraction = valid / total
            if abs(fractions[channel] - expected_fraction) > 1e-9:
                raise ArticulatorAwareError(f"channel_valid_fractions.{channel} is inconsistent.")
        object.__setattr__(self, "channel_valid_counts", MappingProxyType(valid_counts))
        object.__setattr__(self, "channel_total_counts", MappingProxyType(total_counts))
        object.__setattr__(self, "channel_valid_fractions", MappingProxyType(fractions))
        object.__setattr__(self, "all_invalid_channels", tuple(self.all_invalid_channels))
        object.__setattr__(self, "issues", tuple(self.issues))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "split": self.split.value,
            "frame_count": self.frame_count,
            "channel_valid_counts": {
                key: self.channel_valid_counts[key] for key in sorted(self.channel_valid_counts)
            },
            "channel_total_counts": {
                key: self.channel_total_counts[key] for key in sorted(self.channel_total_counts)
            },
            "channel_valid_fractions": {
                key: self.channel_valid_fractions[key]
                for key in sorted(self.channel_valid_fractions)
            },
            "all_invalid_channels": list(self.all_invalid_channels),
            "issues": list(self.issues),
        }


def summarize_channel_masks(
    *,
    vectorized: BfhVectorizedPose,
    split: SampleSplit,
    sample_id: str,
    policy: ArticulatorChannelPartitionPolicy,
    config: ChannelMaskStrategyConfig,
) -> ChannelMaskSummary:
    """Summarize valid BFH observations by articulator channel."""

    masks = build_channel_loss_masks(vectorized=vectorized, policy=policy, config=config)
    valid_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    fractions: dict[str, float] = {}
    all_invalid: list[str] = []
    issues: list[str] = []
    for channel in policy.primary_channels:
        mask = masks[channel]
        valid_count = int(np.count_nonzero(mask))
        total_count = int(mask.size)
        fraction = float(valid_count / total_count)
        valid_counts[channel.value] = valid_count
        total_counts[channel.value] = total_count
        fractions[channel.value] = fraction
        if valid_count == 0:
            all_invalid.append(channel.value)
            if config.missing_channel_policy == "skip_channel_loss":
                issues.append(f"{channel.value}: no valid observations; channel loss will be skipped.")
            else:
                issues.append(f"{channel.value}: no valid observations; policy requires failure.")
        threshold = config.channel_min_valid_fraction[channel]
        if fraction < threshold:
            issues.append(
                f"{channel.value}: valid fraction {fraction:.6f} is below configured minimum {threshold:.6f}."
            )
    return ChannelMaskSummary(
        schema_version=CHANNEL_MASK_SUMMARY_SCHEMA_VERSION,
        sample_id=sample_id,
        split=SampleSplit(split),
        frame_count=vectorized.frame_count,
        channel_valid_counts=valid_counts,
        channel_total_counts=total_counts,
        channel_valid_fractions=fractions,
        all_invalid_channels=tuple(all_invalid),
        issues=tuple(issues),
    )


def build_channel_loss_masks(
    *,
    vectorized: BfhVectorizedPose,
    policy: ArticulatorChannelPartitionPolicy,
    config: ChannelMaskStrategyConfig,
) -> Mapping[PoseChannel, np.ndarray]:
    """Return full flattened feature masks with shape `(frames, total_feature_dim)`."""

    if not isinstance(vectorized, BfhVectorizedPose):
        raise ArticulatorAwareError("vectorized must be BfhVectorizedPose.")
    if not isinstance(policy, ArticulatorChannelPartitionPolicy):
        raise ArticulatorAwareError("policy must be ArticulatorChannelPartitionPolicy.")
    if not isinstance(config, ChannelMaskStrategyConfig):
        raise ArticulatorAwareError("config must be ChannelMaskStrategyConfig.")
    if vectorized.layout != policy.layout:
        raise ArticulatorAwareError("vectorized BFH layout must match articulator partition policy layout.")
    joint_validity = vectorized.validity_mask & vectorized.frame_validity_mask[:, None]
    feature_validity = np.repeat(joint_validity, vectorized.layout.coordinate_dimensions, axis=1)
    if feature_validity.shape != (vectorized.frame_count, vectorized.layout.total_feature_dim):
        raise ArticulatorAwareError("expanded BFH feature validity mask has an invalid shape.")
    masks: dict[PoseChannel, np.ndarray] = {}
    total_valid = 0
    missing_channels: list[PoseChannel] = []
    for channel in policy.primary_channels:
        selected = np.zeros_like(feature_validity, dtype=np.bool_)
        feature_slice = channel_feature_slice(vectorized.layout, channel)
        selected[:, feature_slice] = feature_validity[:, feature_slice]
        selected.setflags(write=False)
        masks[channel] = selected
        valid_count = int(np.count_nonzero(selected))
        total_valid += valid_count
        if valid_count == 0:
            missing_channels.append(channel)
    if total_valid == 0 and config.all_invalid_sample_policy == "fail":
        raise ArticulatorAwareError(
            "all-invalid articulator sample: no valid BFH confidence/frame-validity observations "
            "exist across any canonical channel."
        )
    if missing_channels and config.missing_channel_policy == "fail":
        missing = ", ".join(channel.value for channel in missing_channels)
        raise ArticulatorAwareError(
            "articulator mask strategy requires valid observations for every primary "
            f"channel, but these channels are all-invalid: {missing}. Set "
            "mask_strategy.missing_channel_policy='skip_channel_loss' only when "
            "skipping missing channel loss is intended."
        )
    return MappingProxyType(masks)


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ArticulatorAwareError(f"{name} must be non-empty.")


def _int_mapping(value: Mapping[str, int], name: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ArticulatorAwareError(f"{name} must be a mapping.")
    result: dict[str, int] = {}
    for key, amount in value.items():
        _require_text(key, f"{name} key")
        if not isinstance(amount, int) or isinstance(amount, bool):
            raise ArticulatorAwareError(f"{name}.{key} must be an integer.")
        result[key] = amount
    return result


def _float_mapping(value: Mapping[str, float], name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ArticulatorAwareError(f"{name} must be a mapping.")
    result: dict[str, float] = {}
    for key, amount in value.items():
        _require_text(key, f"{name} key")
        if not isinstance(amount, int | float) or isinstance(amount, bool) or not np.isfinite(float(amount)):
            raise ArticulatorAwareError(f"{name}.{key} must be finite.")
        result[key] = float(amount)
    return result


__all__ = [
    "CHANNEL_MASK_SUMMARY_SCHEMA_VERSION",
    "ChannelMaskSummary",
    "build_channel_loss_masks",
    "summarize_channel_masks",
]
