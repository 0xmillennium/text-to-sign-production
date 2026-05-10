"""Lightweight source-level inspection helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from text_to_sign_production.data.gate.sources.types import (
    SourceAmbiguityFrequency,
    SourceAvailabilitySummary,
    SourceIssueFrequency,
    SourceMatchResult,
    SourceMatchStatus,
    SourceUnmatchedReasonFrequency,
)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def summarize_source_availability(
    matches: Sequence[SourceMatchResult],
) -> SourceAvailabilitySummary:
    """Summarize source matching availability without producing source truth."""
    total = len(matches)
    matched_count = sum(1 for match in matches if match.status is SourceMatchStatus.MATCHED)
    ambiguous_count = sum(1 for match in matches if match.status is SourceMatchStatus.AMBIGUOUS)
    no_match_count = sum(1 for match in matches if match.status is SourceMatchStatus.NO_MATCH)
    viable_count = sum(1 for match in matches if match.structurally_viable)
    return SourceAvailabilitySummary(
        match_count=total,
        matched_count=matched_count,
        ambiguous_count=ambiguous_count,
        no_match_count=no_match_count,
        structurally_viable_count=viable_count,
        matched_ratio=_ratio(matched_count, total),
        structurally_viable_ratio=_ratio(viable_count, total),
    )


def source_issue_frequencies(
    matches: Sequence[SourceMatchResult],
) -> tuple[SourceIssueFrequency, ...]:
    """Count source issue codes for inspection."""
    counter = Counter(
        (issue.code, issue.detail) for match in matches for issue in match.source_issues
    )
    return tuple(
        SourceIssueFrequency(issue_code=key[0], detail=key[1], count=count)
        for key, count in sorted(
            counter.items(), key=lambda item: (item[0][0].value, item[0][1] or "")
        )
    )


def unmatched_reason_frequencies(
    matches: Sequence[SourceMatchResult],
) -> tuple[SourceUnmatchedReasonFrequency, ...]:
    """Count no-match reasons for inspection."""
    counter = Counter(
        (match.unmatched_reason.code, match.unmatched_reason.detail)
        for match in matches
        if match.unmatched_reason is not None
    )
    return tuple(
        SourceUnmatchedReasonFrequency(reason=key[0], detail=key[1], count=count)
        for key, count in sorted(
            counter.items(), key=lambda item: (item[0][0].value, item[0][1] or "")
        )
    )


def ambiguity_frequencies(
    matches: Sequence[SourceMatchResult],
) -> tuple[SourceAmbiguityFrequency, ...]:
    """Count ambiguity reasons for inspection."""
    counter = Counter(
        (reason.code, reason.detail) for match in matches for reason in match.ambiguity_reasons
    )
    return tuple(
        SourceAmbiguityFrequency(reason=key[0], detail=key[1], count=count)
        for key, count in sorted(
            counter.items(), key=lambda item: (item[0][0].value, item[0][1] or "")
        )
    )


__all__ = [
    "ambiguity_frequencies",
    "source_issue_frequencies",
    "summarize_source_availability",
    "unmatched_reason_frequencies",
]
