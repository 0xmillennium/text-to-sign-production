"""Package-local analysis surfaces for source facts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.legacy_data._shared.analysis import numeric_summary, safe_ratio
from text_to_sign_production.legacy_data.sources.types import (
    SourceAvailabilitySummaryRecord,
    SourceFrameCountDistributionRecord,
    SourceIssueFrequencyRecord,
    SourceMatchResult,
    SourceSplitIssueFrequencyRecord,
    SourceSplitUnmatchedReasonCountRecord,
    SourceUnmatchedReasonCountRecord,
    SourceVideoReadabilityRecord,
)


def build_source_availability_summary_record(
    matches: Sequence[SourceMatchResult],
) -> SourceAvailabilitySummaryRecord:
    """Build source availability and match-rate summary counts."""
    matched_count = sum(1 for match in matches if match.matched)
    structurally_viable_count = sum(1 for match in matches if match.structurally_viable)
    keypoint_available_count = sum(
        1 for match in matches if match.keypoints is not None and match.keypoints.exists
    )
    readable_video_count = sum(
        1
        for match in matches
        if match.video_metadata is not None and match.video_metadata.is_readable
    )
    total = len(matches)
    return SourceAvailabilitySummaryRecord(
        match_count=total,
        matched_count=matched_count,
        structurally_viable_count=structurally_viable_count,
        keypoint_available_count=keypoint_available_count,
        readable_video_count=readable_video_count,
        matched_ratio=safe_ratio(matched_count, total),
        structurally_viable_ratio=safe_ratio(structurally_viable_count, total),
        readable_video_ratio=safe_ratio(readable_video_count, total),
    )


def build_source_issue_frequency_records(
    matches: Sequence[SourceMatchResult],
) -> tuple[SourceIssueFrequencyRecord, ...]:
    """Count source-side issue codes."""
    counter = Counter(issue for match in matches for issue in match.source_issues)
    return tuple(
        SourceIssueFrequencyRecord(issue_code=issue_code, count=count)
        for issue_code, count in sorted(counter.items())
    )


def build_source_split_issue_frequency_records(
    matches_by_split: Mapping[SampleSplit, Sequence[SourceMatchResult]],
) -> tuple[SourceSplitIssueFrequencyRecord, ...]:
    """Count source-side issue codes by split."""
    counter = Counter(
        (split, issue)
        for split, matches in matches_by_split.items()
        for match in matches
        for issue in match.source_issues
    )
    return tuple(
        SourceSplitIssueFrequencyRecord(split=split, issue_code=issue_code, count=count)
        for (split, issue_code), count in sorted(
            counter.items(), key=lambda item: (item[0][0].value, item[0][1])
        )
    )


def build_source_unmatched_reason_count_records(
    matches: Sequence[SourceMatchResult],
) -> tuple[SourceUnmatchedReasonCountRecord, ...]:
    """Count unmatched source reasons."""
    counter = Counter(match.unmatched_reason for match in matches if match.unmatched_reason)
    return tuple(
        SourceUnmatchedReasonCountRecord(reason=reason, count=count)
        for reason, count in sorted(counter.items())
    )


def build_source_split_unmatched_reason_count_records(
    matches_by_split: Mapping[SampleSplit, Sequence[SourceMatchResult]],
) -> tuple[SourceSplitUnmatchedReasonCountRecord, ...]:
    """Count unmatched source reasons by split."""
    counter = Counter(
        (split, match.unmatched_reason)
        for split, matches in matches_by_split.items()
        for match in matches
        if match.unmatched_reason
    )
    return tuple(
        SourceSplitUnmatchedReasonCountRecord(split=split, reason=reason, count=count)
        for (split, reason), count in sorted(
            counter.items(), key=lambda item: (item[0][0].value, item[0][1])
        )
    )


def build_source_frame_count_distribution_record(
    matches: Sequence[SourceMatchResult],
) -> SourceFrameCountDistributionRecord:
    """Build a keypoint frame-count distribution for source matches."""
    values = tuple(
        match.keypoints.frame_count if match.keypoints is not None else None for match in matches
    )
    summary = numeric_summary(values)
    return SourceFrameCountDistributionRecord(
        sample_count=summary.sample_count,
        missing_count=summary.missing_count,
        minimum=summary.minimum,
        p50=summary.p50,
        p95=summary.p95,
        maximum=summary.maximum,
    )


def build_source_video_readability_record(
    matches: Sequence[SourceMatchResult],
) -> SourceVideoReadabilityRecord:
    """Build readable-video coverage summary from source matches."""
    readable_count = sum(
        1
        for match in matches
        if match.video_metadata is not None and match.video_metadata.is_readable
    )
    total = len(matches)
    return SourceVideoReadabilityRecord(
        source_count=total,
        readable_count=readable_count,
        unreadable_count=total - readable_count,
        readable_ratio=safe_ratio(readable_count, total),
    )
