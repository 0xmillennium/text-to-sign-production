"""Type surface for samples admission gates."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from text_to_sign_production.core.models import (
    GateDecision,
    GateDecisionBundle,
    GateIssueCode,
    GateName,
    GateStatus,
    PreparedSample,
)


class GateValidationIssueCode(enum.StrEnum):
    """Stable gate-domain validation issue codes."""

    INVALID_CONFIG_THRESHOLD = "invalid_config_threshold"
    CONTRADICTORY_THRESHOLDS = "contradictory_thresholds"
    INVALID_GATE_DECISION = "invalid_gate_decision"
    INVALID_DECISION_BUNDLE = "invalid_decision_bundle"


@dataclass(frozen=True, slots=True)
class GateValidationIssue:
    """Structured gate-domain validation issue."""

    code: GateValidationIssueCode
    message: str


@dataclass(frozen=True, slots=True)
class GateEvaluationInput:
    """Single authoritative input contract for samples admission gates."""

    sample: PreparedSample


@dataclass(frozen=True, slots=True)
class GateIssue:
    """One observed gate issue with optional values for diagnostics."""

    gate_name: GateName
    code: GateIssueCode
    message: str
    observed_value: int | float | str | bool | None = None
    threshold_value: int | float | str | bool | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class GateCheckResult:
    """Compatibility view over one root gate decision."""

    gate_name: GateName
    status: GateStatus
    issues: tuple[GateIssue, ...] = ()

    @property
    def passed(self) -> bool:
        """Whether this gate passed."""
        return self.status is GateStatus.PASS

    @property
    def failed(self) -> bool:
        """Whether this gate failed."""
        return self.status is GateStatus.FAIL


__all__ = [
    "GateCheckResult",
    "GateDecision",
    "GateDecisionBundle",
    "GateEvaluationInput",
    "GateIssue",
    "GateIssueCode",
    "GateName",
    "GateStatus",
    "GateValidationIssue",
    "GateValidationIssueCode",
]
