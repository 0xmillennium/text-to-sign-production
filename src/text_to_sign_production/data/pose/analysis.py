"""Package-local analysis surfaces for pose build outputs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from text_to_sign_production.data._shared.analysis import numeric_summary
from text_to_sign_production.data.pose.schema import CANONICAL_POSE_CHANNELS
from text_to_sign_production.data.pose.types import (
    PoseBuildDiagnosticAggregateRecord,
    PoseBuildOutput,
    PoseChannelNonzeroDistributionRecord,
    PoseIssueFrequencyRecord,
    PosePeoplePerFrameDistributionRecord,
    PosePersonSelectionFallbackCountRecord,
)


def build_pose_issue_frequency_records(
    outputs: Sequence[PoseBuildOutput],
) -> tuple[PoseIssueFrequencyRecord, ...]:
    """Count pose frame/build issue codes across outputs."""
    counter: Counter[str] = Counter()
    for output in outputs:
        counter.update(output.frame_quality.frame_issue_counts)
        if output.diagnostics.unrecoverable_error is not None:
            counter["unrecoverable_pose_error"] += 1
    return tuple(
        PoseIssueFrequencyRecord(issue_code=issue_code, count=count)
        for issue_code, count in sorted(counter.items())
    )


def build_pose_person_selection_fallback_count_records(
    outputs: Sequence[PoseBuildOutput],
) -> tuple[PosePersonSelectionFallbackCountRecord, ...]:
    """Count selected-person fallback use by selection policy."""
    counter = Counter(
        (
            output.diagnostics.person_selection_policy,
            output.diagnostics.person_selection_fallback_used,
        )
        for output in outputs
    )
    return tuple(
        PosePersonSelectionFallbackCountRecord(
            policy=policy,
            fallback_used=fallback_used,
            count=count,
        )
        for (policy, fallback_used), count in sorted(
            counter.items(), key=lambda item: (item[0][0].value, item[0][1])
        )
    )


def build_pose_channel_nonzero_distribution_records(
    outputs: Sequence[PoseBuildOutput],
) -> tuple[PoseChannelNonzeroDistributionRecord, ...]:
    """Build distributions of channel nonzero frame counts across outputs."""
    records: list[PoseChannelNonzeroDistributionRecord] = []
    for channel in CANONICAL_POSE_CHANNELS:
        values = tuple(
            output.frame_quality.channel_nonzero_frames.get(channel) for output in outputs
        )
        summary = numeric_summary(values)
        records.append(
            PoseChannelNonzeroDistributionRecord(
                channel=channel,
                sample_count=summary.sample_count,
                missing_count=summary.missing_count,
                minimum=summary.minimum,
                p50=summary.p50,
                p95=summary.p95,
                maximum=summary.maximum,
            )
        )
    return tuple(records)


def build_pose_people_per_frame_distribution_record(
    outputs: Sequence[PoseBuildOutput],
) -> PosePeoplePerFrameDistributionRecord:
    """Build the people-per-frame distribution across pose outputs."""
    values: list[int] = []
    for output in outputs:
        values.extend(int(value) for value in np.asarray(output.people_per_frame).tolist())
    summary = numeric_summary(tuple(values))
    return PosePeoplePerFrameDistributionRecord(
        output_count=len(outputs),
        frame_count=len(values),
        minimum=summary.minimum,
        p50=summary.p50,
        p95=summary.p95,
        maximum=summary.maximum,
    )


def build_pose_build_diagnostic_aggregate_record(
    outputs: Sequence[PoseBuildOutput],
) -> PoseBuildDiagnosticAggregateRecord:
    """Aggregate pose build diagnostic counts."""
    return PoseBuildDiagnosticAggregateRecord(
        output_count=len(outputs),
        unrecoverable_error_count=sum(
            1 for output in outputs if output.diagnostics.unrecoverable_error is not None
        ),
        fallback_used_count=sum(
            1 for output in outputs if output.diagnostics.person_selection_fallback_used
        ),
    )
