"""Typed contracts for structural processing gates."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


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


@dataclass(frozen=True, slots=True)
class GateValidationIssue:
    """A specific issue found during gate contract validation."""

    code: str
    message: str
