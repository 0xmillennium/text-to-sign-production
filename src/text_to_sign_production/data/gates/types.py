"""Typed contracts for structural processing gates."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class GateStatus(enum.Enum):
    """The outcome of a structural gate."""

    PASSED = "passed"
    DROPPED = "dropped"


@dataclass(frozen=True, slots=True)
class GateResult:
    """The result of a single structural gate evaluation."""

    status: GateStatus
    reasons: list[str] = field(default_factory=list)

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
    drop_stage: str | None
    drop_reasons: list[str]
    can_materialize_debug: bool
    gate_results: dict[str, GateResult]

    @property
    def dropped(self) -> bool:
        return self.status == ProcessingStatus.DROPPED
