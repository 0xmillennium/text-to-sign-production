"""Articulator-aware channel partition policy built from canonical BFH masks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.backbones.bfh_masks import (
    BfhChannelPartition,
    build_bfh_channel_partitions,
)
from text_to_sign_production.modeling.backbones.bfh_vectorization import BfhTensorLayout
from text_to_sign_production.modeling.candidates.articulator_aware.config import (
    ChannelPartitionPolicyConfig,
)
from text_to_sign_production.modeling.candidates.articulator_aware.errors import (
    ArticulatorAwareError,
)

ARTICULATOR_PARTITION_POLICY_SCHEMA_VERSION = "t2sp-articulator-partition-policy-v1"


@dataclass(frozen=True, slots=True)
class ArticulatorChannelPartitionPolicy:
    schema_version: str
    source: str
    layout: BfhTensorLayout
    primary_channels: tuple[PoseChannel, ...]
    partitions: tuple[BfhChannelPartition, ...]
    composite_groups: Mapping[str, tuple[PoseChannel, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if self.schema_version != ARTICULATOR_PARTITION_POLICY_SCHEMA_VERSION:
            raise ArticulatorAwareError("articulator partition policy schema_version is unsupported.")
        if self.source != "canonical_bfh_channels":
            raise ArticulatorAwareError("articulator partition policy source must be canonical_bfh_channels.")
        if not isinstance(self.layout, BfhTensorLayout):
            raise ArticulatorAwareError("articulator partition policy layout must be BfhTensorLayout.")
        primary = tuple(PoseChannel(channel) for channel in self.primary_channels)
        if primary != tuple(self.layout.channels):
            raise ArticulatorAwareError(
                "articulator primary channels must match the canonical BFH layout channels."
            )
        if len(set(primary)) != len(primary):
            raise ArticulatorAwareError("articulator primary channels must not contain duplicates.")
        partitions = tuple(self.partitions)
        if len(partitions) != len(primary) or any(
            not isinstance(partition, BfhChannelPartition) for partition in partitions
        ):
            raise ArticulatorAwareError("articulator partitions must contain one BfhChannelPartition per channel.")
        if tuple(partition.channel for partition in partitions) != primary:
            raise ArticulatorAwareError("articulator partitions must follow primary channel order.")
        covered = [index for part in partitions for index in range(part.joint_start, part.joint_stop)]
        if covered != list(range(self.layout.total_joint_count)):
            raise ArticulatorAwareError("articulator partitions must cover all BFH joints exactly once.")
        groups: dict[str, tuple[PoseChannel, ...]] = {}
        for name, channels in self.composite_groups.items():
            if not isinstance(name, str) or not name.strip():
                raise ArticulatorAwareError("composite group names must be non-empty strings.")
            resolved = tuple(PoseChannel(channel) for channel in channels)
            if not resolved:
                raise ArticulatorAwareError(f"composite group {name!r} must not be empty.")
            unknown = tuple(channel.value for channel in resolved if channel not in primary)
            if unknown:
                raise ArticulatorAwareError(
                    f"composite group {name!r} references non-primary channels: {', '.join(unknown)}."
                )
            groups[name] = resolved
        object.__setattr__(self, "primary_channels", primary)
        object.__setattr__(self, "partitions", partitions)
        object.__setattr__(self, "composite_groups", MappingProxyType(groups))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "layout": self.layout.to_dict(),
            "primary_channels": [channel.value for channel in self.primary_channels],
            "partitions": [partition.to_dict() for partition in self.partitions],
            "composite_groups": {
                name: [channel.value for channel in self.composite_groups[name]]
                for name in sorted(self.composite_groups)
            },
        }


def build_articulator_partition_policy(
    *,
    config: ChannelPartitionPolicyConfig,
    layout: BfhTensorLayout,
) -> ArticulatorChannelPartitionPolicy:
    """Build the Phase 8 partition policy from shared BFH channel partitions."""

    if not isinstance(config, ChannelPartitionPolicyConfig):
        raise ArticulatorAwareError("config must be ChannelPartitionPolicyConfig.")
    if not isinstance(layout, BfhTensorLayout):
        raise ArticulatorAwareError("layout must be BfhTensorLayout.")
    partitions = build_bfh_channel_partitions(layout)
    if tuple(partition.channel for partition in partitions) != config.primary_channels:
        raise ArticulatorAwareError(
            "canonical BFH partitions do not match articulator primary channels."
        )
    return ArticulatorChannelPartitionPolicy(
        schema_version=ARTICULATOR_PARTITION_POLICY_SCHEMA_VERSION,
        source=config.source,
        layout=layout,
        primary_channels=config.primary_channels,
        partitions=partitions,
        composite_groups=config.composite_groups,
    )


__all__ = [
    "ARTICULATOR_PARTITION_POLICY_SCHEMA_VERSION",
    "ArticulatorChannelPartitionPolicy",
    "build_articulator_partition_policy",
]
