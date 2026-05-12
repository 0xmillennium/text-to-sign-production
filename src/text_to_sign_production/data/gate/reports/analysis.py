"""Read-only analysis helpers for gate-stage report projections."""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sign_production.data.gate.reports.types import GateReportBundle
from text_to_sign_production.data.gate.reports.validate import (
    validate_gate_report_bundle,
)


@dataclass(frozen=True, slots=True)
class GateReportAudit:
    """Compact audit of report projection integrity."""

    valid: bool
    validation_issue_count: int


def audit_gate_report(bundle: GateReportBundle) -> GateReportAudit:
    """Audit report projection integrity."""
    issues = validate_gate_report_bundle(bundle)
    return GateReportAudit(
        valid=not issues,
        validation_issue_count=len(issues),
    )


__all__ = ["GateReportAudit", "audit_gate_report"]
