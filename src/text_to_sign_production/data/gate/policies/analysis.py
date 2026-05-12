"""Read-only analysis helpers for gate admission gates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from text_to_sign_production.core.ids import SampleStatus
from text_to_sign_production.core.models import (
    GateDecisionBundle,
    GateIssueCode,
    GateName,
    GateStatus,
)


@dataclass(frozen=True, slots=True)
class GateStatusCount:
    """Count of individual gate decisions by status."""

    status: GateStatus
    count: int


@dataclass(frozen=True, slots=True)
class GateIssueFrequency:
    """Frequency of a gate issue code."""

    gate_name: GateName
    issue_code: GateIssueCode
    count: int


@dataclass(frozen=True, slots=True)
class FailedGateSummary:
    """Compact failed-gate summary for one decision bundle."""

    passed: bool
    failed_gate_names: tuple[GateName, ...]
    issue_count: int


def summarize_failed_gates(bundle: GateDecisionBundle) -> FailedGateSummary:
    """Summarize failed gates for one gate admission decision bundle."""
    return FailedGateSummary(
        passed=bundle.final_status is SampleStatus.PASSED,
        failed_gate_names=bundle.failed_gates,
        issue_count=sum(len(decision.issue_codes) for decision in bundle.decisions),
    )


def gate_status_counts(
    bundles: Sequence[GateDecisionBundle],
) -> tuple[GateStatusCount, ...]:
    """Count per-gate statuses across decision bundles."""
    counter = Counter(decision.status for bundle in bundles for decision in bundle.decisions)
    return tuple(
        GateStatusCount(status=status, count=counter[status])
        for status in GateStatus
        if counter[status] > 0
    )


def gate_issue_frequencies(
    bundles: Sequence[GateDecisionBundle],
) -> tuple[GateIssueFrequency, ...]:
    """Count gate issue codes across decision bundles."""
    counter = Counter(
        (decision.gate, code)
        for bundle in bundles
        for decision in bundle.decisions
        for code in decision.issue_codes
    )
    return tuple(
        GateIssueFrequency(gate_name=gate_name, issue_code=code, count=count)
        for (gate_name, code), count in sorted(
            counter.items(),
            key=lambda item: (item[0][0].value, item[0][1].value),
        )
    )


def ordered_hard_fail_reasons(bundle: GateDecisionBundle) -> tuple[GateIssueCode, ...]:
    """Return ordered admission-fail issue codes from one decision bundle."""
    return tuple(code for decision in bundle.decisions for code in decision.issue_codes)


__all__ = [
    "FailedGateSummary",
    "GateIssueFrequency",
    "GateStatusCount",
    "gate_issue_frequencies",
    "gate_status_counts",
    "ordered_hard_fail_reasons",
    "summarize_failed_gates",
]
