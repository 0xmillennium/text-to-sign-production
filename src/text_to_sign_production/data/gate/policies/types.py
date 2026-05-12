"""Type surface for gate admission gates."""

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
    """Single authoritative input contract for gate admission gates."""

    sample: PreparedSample


__all__ = [
    "GateDecision",
    "GateDecisionBundle",
    "GateEvaluationInput",
    "GateIssueCode",
    "GateName",
    "GateStatus",
    "GateValidationIssue",
    "GateValidationIssueCode",
]
