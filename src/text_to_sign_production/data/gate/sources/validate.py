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
    CandidateViabilityIssue,
    CandidateViabilityReport,
    CandidateViabilityStatus,
    SourceCandidate,
    SourceIssueCode,
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


def validate_candidate_identity_invariants(
    candidate: SourceCandidate,
) -> tuple[SourceValidationIssue, ...]:
    """Validate identity invariants only for a matched source candidate.

    This function intentionally does not inspect pose/source viability fields
    such as ``frame_count`` or ``source_issues``.
    """
    issues: list[SourceValidationIssue] = []
    if not candidate.sample_id.strip():
        issues.append(
            _issue(
                SourceValidationIssueCode.CANDIDATE_EMPTY_SAMPLE_ID,
                "Candidate sample_id is empty.",
            )
        )
    if not candidate.sentence_id.strip():
        issues.append(
            _issue(
                SourceValidationIssueCode.EMPTY_SENTENCE_ID,
                "Candidate sentence_id is empty.",
            )
        )
    if not candidate.video_id.strip():
        issues.append(
            _issue(
                SourceValidationIssueCode.EMPTY_VIDEO_ID,
                "Candidate video_id is empty.",
            )
        )
    if candidate.identity is None:
        issues.append(
            _issue(
                SourceValidationIssueCode.MISSING_CANDIDATE_IDENTITY,
                "Candidate must carry explicit matched source identity.",
            )
        )
        return tuple(issues)
    if candidate.identity.split is not candidate.split:
        issues.append(
            _issue(
                SourceValidationIssueCode.CANDIDATE_IDENTITY_INCOHERENT,
                "Candidate identity split must match candidate split.",
            )
        )
    if candidate.identity.translation.video != candidate.identity.video:
        issues.append(
            _issue(
                SourceValidationIssueCode.CANDIDATE_IDENTITY_INCOHERENT,
                "Candidate translation and video identities must agree.",
            )
        )
    if candidate.identity.translation.keypoint != candidate.identity.keypoint:
        issues.append(
            _issue(
                SourceValidationIssueCode.CANDIDATE_IDENTITY_INCOHERENT,
                "Candidate translation and keypoint identities must agree.",
            )
        )
    if candidate.identity.video.video_key.value != candidate.video_id:
        issues.append(
            _issue(
                SourceValidationIssueCode.CANDIDATE_IDENTITY_INCOHERENT,
                "Candidate identity video key must match candidate video_id.",
            )
        )
    if candidate.identity.translation.sentence_key.value != candidate.sentence_id:
        issues.append(
            _issue(
                SourceValidationIssueCode.CANDIDATE_IDENTITY_INCOHERENT,
                "Candidate identity sentence key must match candidate sentence_id.",
            )
        )
    if candidate.identity.keypoint.sample_key.value != candidate.sample_id:
        issues.append(
            _issue(
                SourceValidationIssueCode.CANDIDATE_IDENTITY_INCOHERENT,
                "Candidate identity keypoint sample key must match candidate sample_id.",
            )
        )
    return tuple(issues)


def validate_candidate_set(
    candidates: Sequence[SourceCandidate],
) -> tuple[SourceValidationIssue, ...]:
    """Validate collection-level source candidate identity invariants."""
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
        issues.extend(validate_candidate_identity_invariants(candidate))
    return tuple(issues)


def build_candidate_viability_report(
    candidate: SourceCandidate,
    *,
    observed_frame_count: int | None = None,
    missing_frame_files: bool | None = None,
) -> CandidateViabilityReport:
    """Build pose/source structural viability truth for one candidate."""
    issues: list[CandidateViabilityIssue] = [
        CandidateViabilityIssue(
            code=issue.code,
            message=f"Source issue present: {issue.code.value}.",
            detail=issue.detail,
        )
        for issue in candidate.source_issues
    ]
    if candidate.frame_count <= 0:
        issues.append(
            CandidateViabilityIssue(
                code=SourceIssueCode.MISSING_FRAME_JSON_FILES,
                message="Candidate keypoint frame_count is not positive.",
            )
        )
    if observed_frame_count is not None and observed_frame_count <= 0:
        issues.append(
            CandidateViabilityIssue(
                code=SourceIssueCode.MISSING_FRAME_JSON_FILES,
                message="No frame JSON files were discovered for the candidate.",
            )
        )
    if missing_frame_files:
        issues.append(
            CandidateViabilityIssue(
                code=SourceIssueCode.MISSING_FRAME_JSON_FILES,
                message="The discovered frame JSON listing is incomplete.",
            )
        )
    return CandidateViabilityReport(
        split=candidate.split,
        sample_id=candidate.sample_id,
        sentence_id=candidate.sentence_id,
        status=(
            CandidateViabilityStatus.NON_VIABLE
            if issues
            else CandidateViabilityStatus.VIABLE
        ),
        issues=tuple(issues),
    )


__all__ = [
    "build_candidate_viability_report",
    "validate_candidate_identity_invariants",
    "validate_candidate_invariants",
    "validate_candidate_set",
    "validate_match_invariants",
]
