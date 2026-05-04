"""Source-level structural gate."""

from __future__ import annotations

from text_to_sign_production.data.gates.types import GateResult, GateStatus
from text_to_sign_production.data.sources.types import SourceCandidate
from text_to_sign_production.data.sources.validate import validate_candidate


def evaluate_source_gate(candidate: SourceCandidate) -> GateResult:
    """Evaluate if the source candidate is structurally viable.

    Uses the established source-level validation logic.
    """
    issues = validate_candidate(candidate)

    if issues:
        return GateResult(status=GateStatus.DROPPED, reasons=issues)

    return GateResult(status=GateStatus.PASSED)
