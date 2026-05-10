"""Source-side samples admission gate."""

from __future__ import annotations

from text_to_sign_production.core.models import (
    GateDecision,
    GateIssueCode,
    GateName,
    GateStatus,
    PreparedSample,
)
from text_to_sign_production.data.gate.policies.config import (
    SourceGateThresholds,
)


def evaluate_source_gate(
    sample: PreparedSample,
    thresholds: SourceGateThresholds,
) -> GateDecision:
    """Evaluate whether source truth is admissible into the samples checkpoint."""
    issues: list[GateIssueCode] = []
    normalized_text = sample.source.canonical_normalized_text
    if not sample.source.text.strip() or not normalized_text.strip():
        issues.append(GateIssueCode.SOURCE_TEXT_MISSING)
    if len(normalized_text.replace(" ", "")) < thresholds.min_character_count:
        issues.append(GateIssueCode.SOURCE_TEXT_TOO_SHORT)
    if len(normalized_text.split()) < thresholds.min_token_count:
        issues.append(GateIssueCode.SOURCE_TEXT_TOO_SHORT)
    if thresholds.fail_on_source_issues and sample.source.source_issue_codes:
        issues.append(GateIssueCode.SOURCE_ISSUE_PRESENT)
    issue_codes = tuple(dict.fromkeys(issues))
    return GateDecision(
        gate=GateName.SOURCE,
        status=GateStatus.FAIL if issue_codes else GateStatus.PASS,
        issue_codes=issue_codes,
    )


__all__ = ["evaluate_source_gate"]
