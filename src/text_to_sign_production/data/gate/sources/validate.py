"""Higher-level invariant validation for source-domain truths."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from text_to_sign_production.data.gate.sources.schema import (
    validate_candidate as validate_candidate_schema,
)
from text_to_sign_production.data.gate.sources.schema import (
    validate_match_result as validate_match_schema,
)
from text_to_sign_production.data.gate.sources.types import (
    SourceCandidate,
    SourceMatchResult,
    SourceMatchStatus,
    SourceValidationIssue,
    SourceValidationIssueCode,
)


def _issue(code: SourceValidationIssueCode, message: str) -> SourceValidationIssue:
    return SourceValidationIssue(code=code, message=message)


def validate_match_invariants(match: SourceMatchResult) -> tuple[SourceValidationIssue, ...]:
    """Validate match-state invariants beyond structural shape."""
    issues: list[SourceValidationIssue] = list(validate_match_schema(match))

    if match.status is SourceMatchStatus.MATCHED:
        if match.unmatched_reason is not None:
            issues.append(
                _issue(
                    SourceValidationIssueCode.MATCHED_HAS_UNMATCHED_REASON,
                    "Matched result has no-match reason.",
                )
            )
        if match.ambiguity_reasons:
            issues.append(
                _issue(
                    SourceValidationIssueCode.MATCHED_HAS_AMBIGUITY,
                    "Matched result has ambiguity reasons.",
                )
            )
        if len(match.video_matches) != 1 or len(match.keypoint_matches) != 1:
            issues.append(
                _issue(
                    SourceValidationIssueCode.MATCHED_CARDINALITY_INVALID,
                    "Matched result must have exactly one video and keypoint match.",
                )
            )
        if match.candidate_identity is None:
            issues.append(
                _issue(
                    SourceValidationIssueCode.MISSING_CANDIDATE_IDENTITY,
                    "Matched result must carry explicit candidate identity.",
                )
            )
        elif len(match.video_matches) == 1 and len(match.keypoint_matches) == 1:
            if match.translation.identity != match.candidate_identity.translation:
                issues.append(
                    _issue(
                        SourceValidationIssueCode.CANDIDATE_IDENTITY_INCOHERENT,
                        "Matched candidate identity must include translation identity.",
                    )
                )
            if match.video_matches[0].identity != match.candidate_identity.video:
                issues.append(
                    _issue(
                        SourceValidationIssueCode.CANDIDATE_IDENTITY_INCOHERENT,
                        "Matched candidate identity must include video identity.",
                    )
                )
            if match.keypoint_matches[0].identity != match.candidate_identity.keypoint:
                issues.append(
                    _issue(
                        SourceValidationIssueCode.CANDIDATE_IDENTITY_INCOHERENT,
                        "Matched candidate identity must include keypoint identity.",
                    )
                )
    elif match.status is SourceMatchStatus.NO_MATCH:
        if match.unmatched_reason is None:
            issues.append(
                _issue(
                    SourceValidationIssueCode.NO_MATCH_MISSING_REASON,
                    "No-match result must carry a reason.",
                )
            )
        if match.ambiguity_reasons:
            issues.append(
                _issue(
                    SourceValidationIssueCode.NO_MATCH_HAS_AMBIGUITY,
                    "No-match result has ambiguity reasons.",
                )
            )
    elif match.status is SourceMatchStatus.AMBIGUOUS:
        if not match.ambiguity_reasons:
            issues.append(
                _issue(
                    SourceValidationIssueCode.AMBIGUITY_MISSING_REASON,
                    "Ambiguous result must carry a reason.",
                )
            )
        if match.unmatched_reason is not None:
            issues.append(
                _issue(
                    SourceValidationIssueCode.AMBIGUITY_HAS_UNMATCHED_REASON,
                    "Ambiguous result has no-match reason.",
                )
            )

    return tuple(issues)


def validate_candidate_invariants(candidate: SourceCandidate) -> tuple[SourceValidationIssue, ...]:
    """Validate candidate invariants beyond structural shape."""
    issues: list[SourceValidationIssue] = list(validate_candidate_schema(candidate))
    if candidate.source_issues and candidate.structurally_viable:
        issues.append(
            _issue(
                SourceValidationIssueCode.CANDIDATE_ISSUES_BUT_VIABLE,
                "Candidate viability conflicts with issues.",
            )
        )
    return tuple(issues)


def validate_candidate_set(
    candidates: Sequence[SourceCandidate],
) -> tuple[SourceValidationIssue, ...]:
    """Validate collection-level source candidate invariants."""
    issues: list[SourceValidationIssue] = []
    keys = [(candidate.split, candidate.sample_id) for candidate in candidates]
    counts = Counter(keys)
    for (split, sample_id), count in sorted(
        counts.items(), key=lambda item: (item[0][0].value, item[0][1])
    ):
        if count > 1:
            issues.append(
                _issue(
                    SourceValidationIssueCode.DUPLICATE_CANDIDATE_SAMPLE_ID,
                    f"Duplicate source candidate for split={split.value} sample_id={sample_id}.",
                )
            )
    for candidate in candidates:
        issues.extend(validate_candidate_invariants(candidate))
    return tuple(issues)


__all__ = [
    "validate_candidate_invariants",
    "validate_candidate_set",
    "validate_match_invariants",
]
