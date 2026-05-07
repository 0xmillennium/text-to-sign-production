"""Typed contracts for structural processing gates."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.data._shared.types import ValidationIssue


class GateStatus(enum.Enum):
    """The outcome of a structural gate."""

    PASSED = "passed"
    DROPPED = "dropped"


class GateStage(enum.StrEnum):
    """Structural gate stages in deterministic evaluation order."""

    SOURCE = "source"
    FRAMES = "frames"
    SCHEMA = "schema"
    BODY = "body"
    HAND = "hand"
    FACE = "face"
    ARTIFACT = "artifact"


@dataclass(frozen=True, slots=True)
class ChannelGateConfig:
    """Structural configuration for a specific canonical channel."""

    min_nonzero_frames: int


@dataclass(frozen=True, slots=True)
class IntegrityGateConfig:
    """Frame and coordinate integrity gate thresholds."""

    min_valid_frames: int
    max_out_of_bounds_ratio: float
    min_num_frames: int
    min_duration_seconds: float


@dataclass(frozen=True, slots=True)
class TrackingIntegrityGateConfig:
    """Tracking-continuity integrity gate thresholds."""

    max_tracked_target_missing_frame_ratio: float
    max_zeroed_canonical_joint_frame_ratio: float
    max_person_tracking_continuity_break_ratio: float
    max_person_tracking_reanchor_ratio: float


@dataclass(frozen=True, slots=True)
class ChannelPresenceGateConfig:
    """Channel-presence gate thresholds."""

    min_any_hand_nonzero_frames: int
    channels: Mapping[str, ChannelGateConfig]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "channels",
            MappingProxyType(dict(self.channels)),
        )


@dataclass(frozen=True, slots=True)
class TextSanityGateConfig:
    """Text sanity gate thresholds."""

    min_character_count: int
    min_token_count: int


@dataclass(frozen=True, slots=True)
class GatesConfig:
    """Configuration for all structural gates."""

    integrity: IntegrityGateConfig
    tracking_integrity: TrackingIntegrityGateConfig
    channel_presence: ChannelPresenceGateConfig
    text_sanity: TextSanityGateConfig


@dataclass(frozen=True, slots=True)
class GateResult:
    """The result of a single structural gate evaluation."""

    status: GateStatus
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(self.reasons))

    @property
    def passed(self) -> bool:
        return self.status == GateStatus.PASSED


class ProcessingStatus(enum.Enum):
    """Final structural processing status."""

    PROCESSED = "processed"
    DROPPED = "dropped"


@dataclass(frozen=True, slots=True)
class ProcessingDecision:
    """The final structural processing decision for a sample."""

    status: ProcessingStatus
    drop_stage: GateStage | None
    drop_reasons: tuple[str, ...]
    can_materialize_debug: bool
    gate_results: Mapping[GateStage, GateResult]

    def __post_init__(self) -> None:
        object.__setattr__(self, "drop_reasons", tuple(self.drop_reasons))
        object.__setattr__(
            self,
            "gate_results",
            MappingProxyType(dict(self.gate_results)),
        )

    @property
    def dropped(self) -> bool:
        return self.status == ProcessingStatus.DROPPED


GateValidationIssue = ValidationIssue


@dataclass(frozen=True, slots=True)
class GateProcessingStatusCountRecord:
    """Count of final processing decisions by status."""

    status: ProcessingStatus
    count: int


@dataclass(frozen=True, slots=True)
class GateDropStageCountRecord:
    """Count of dropped decisions by structural gate stage."""

    stage: GateStage
    count: int


@dataclass(frozen=True, slots=True)
class GateReasonFrequencyRecord:
    """Frequency of machine-readable gate reasons."""

    stage: GateStage | None
    reason: str
    count: int


@dataclass(frozen=True, slots=True)
class GateStageResultCountRecord:
    """Pass/drop counts for individual gate results."""

    stage: GateStage
    status: GateStatus
    count: int


@dataclass(frozen=True, slots=True)
class GateSplitStageSummaryRecord:
    """Split-aware pass/drop counts for an individual gate stage."""

    split: SampleSplit
    stage: GateStage
    passed_count: int
    dropped_count: int


@dataclass(frozen=True, slots=True)
class GateBlockerSummaryRecord:
    """Drop-stage and reason blocker count for processing decisions."""

    stage: GateStage
    reason: str
    count: int


@dataclass(frozen=True, slots=True)
class GateSplitBlockerSummaryRecord:
    """Split-aware drop-stage and reason blocker count."""

    split: SampleSplit
    stage: GateStage
    reason: str
    count: int
