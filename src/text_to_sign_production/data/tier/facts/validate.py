"""Invariant validation for PreparedSample-derived quality facts."""

from __future__ import annotations

from text_to_sign_production.data.tier.facts.schema import validate_quality_facts_schema
from text_to_sign_production.data.tier.facts.types import (
    FactsValidationIssue,
    FactsValidationIssueCode,
    QualityFacts,
)


def validate_quality_facts_invariants(
    facts: QualityFacts,
) -> tuple[FactsValidationIssue, ...]:
    """Validate composed quality-facts invariants."""
    issues = list(validate_quality_facts_schema(facts))
    if facts.frame.valid_frame_count + facts.frame.invalid_frame_count != facts.frame.frame_count:
        issues.append(
            FactsValidationIssue(
                FactsValidationIssueCode.FRAME_COUNT_PARTITION_MISMATCH,
                "valid + invalid frame counts must equal frame_count.",
                "frame",
            )
        )
    for index, channel in enumerate(facts.channel):
        if channel.nonzero_frame_count + channel.zero_frame_count != facts.frame.frame_count:
            issues.append(
                FactsValidationIssue(
                    FactsValidationIssueCode.FRAME_COUNT_PARTITION_MISMATCH,
                    "channel nonzero + zero counts must equal frame_count.",
                    f"channel.{index}",
                )
            )
    return tuple(issues)


__all__ = ["validate_quality_facts_invariants"]
