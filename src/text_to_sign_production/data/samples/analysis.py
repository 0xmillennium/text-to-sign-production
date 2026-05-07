"""Package-local analysis surfaces for samples and manifests."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.data._shared.analysis import numeric_summary
from text_to_sign_production.data.samples.types import (
    DroppedDebugMaterializationOutcome,
    DroppedManifestEntry,
    ManifestEntry,
    ProcessedSamplePayload,
    SampleDroppedArchivePublishableCountRecord,
    SampleDroppedMaterializationOutcomeCountRecord,
    SampleDroppedMaterializationSummaryRecord,
    SampleManifestStatusCountRecord,
    SampleNumericDistributionRecord,
    SamplePayloadCompletenessRecord,
    SampleProcessingSummaryRecord,
    SampleSplitCountRecord,
    SampleSplitStatusCountRecord,
    SampleStatus,
)


def build_manifest_status_count_records(
    entries: Sequence[ManifestEntry],
) -> tuple[SampleManifestStatusCountRecord, ...]:
    """Count manifest entries by status."""
    counter = Counter(entry.status for entry in entries)
    return tuple(
        SampleManifestStatusCountRecord(status=status, count=counter[status])
        for status in SampleStatus
        if counter[status] > 0
    )


def build_sample_split_status_count_records(
    entries: Sequence[ManifestEntry],
) -> tuple[SampleSplitStatusCountRecord, ...]:
    """Count manifest entries by split and status."""
    counter = Counter((entry.split, entry.status) for entry in entries)
    return tuple(
        SampleSplitStatusCountRecord(split=split, status=status, count=count)
        for (split, status), count in sorted(
            counter.items(), key=lambda item: (item[0][0].value, item[0][1].value)
        )
    )


def build_sample_processing_summary_records(
    entries: Sequence[ManifestEntry],
    *,
    splits: Sequence[SampleSplit] = (),
) -> tuple[SampleProcessingSummaryRecord, ...]:
    """Build split-level passed/dropped manifest processing counts."""
    split_status_counts = build_sample_split_status_count_records(entries)
    ordered_splits = tuple(
        sorted(
            {record.split for record in split_status_counts}
            | {entry_split for entry_split in splits},
            key=lambda item: item.value,
        )
    )
    counts = {
        (record.split, record.status): record.count
        for record in split_status_counts
    }
    return tuple(
        SampleProcessingSummaryRecord(
            split=split,
            passed_count=counts.get((split, SampleStatus.PASSED), 0),
            dropped_count=counts.get((split, SampleStatus.DROPPED), 0),
            total_count=sum(
                counts.get((split, status), 0)
                for status in SampleStatus
            ),
        )
        for split in ordered_splits
    )


def build_sample_split_count_records(
    entries: Sequence[ManifestEntry],
) -> tuple[SampleSplitCountRecord, ...]:
    """Count manifest entries by split."""
    counter = Counter(entry.split for entry in entries)
    return tuple(
        SampleSplitCountRecord(split=split, count=count)
        for split, count in sorted(counter.items(), key=lambda item: item[0].value)
    )


def build_sample_dropped_materialization_summary_records(
    entries: Sequence[ManifestEntry],
    *,
    splits: Sequence[SampleSplit] = (),
) -> tuple[SampleDroppedMaterializationSummaryRecord, ...]:
    """Build split-level dropped debug materialization lifecycle counts."""
    dropped_entries = tuple(
        entry for entry in entries if isinstance(entry, DroppedManifestEntry)
    )
    ordered_splits = tuple(
        sorted(
            {entry.split for entry in dropped_entries}
            | {entry_split for entry_split in splits},
            key=lambda item: item.value,
        )
    )
    return tuple(
        _dropped_materialization_summary_record(split, dropped_entries)
        for split in ordered_splits
    )


def build_sample_dropped_materialization_outcome_count_records(
    entries: Sequence[ManifestEntry],
) -> tuple[SampleDroppedMaterializationOutcomeCountRecord, ...]:
    """Count dropped debug materialization outcomes by split."""
    counter = Counter(
        (entry.split, entry.materialization.debug_materialization_outcome)
        for entry in entries
        if isinstance(entry, DroppedManifestEntry)
    )
    return tuple(
        SampleDroppedMaterializationOutcomeCountRecord(
            split=split,
            outcome=outcome,
            count=count,
        )
        for (split, outcome), count in sorted(
            counter.items(), key=lambda item: (item[0][0].value, item[0][1].value)
        )
    )


def build_sample_dropped_archive_publishable_count_records(
    entries: Sequence[ManifestEntry],
    *,
    splits: Sequence[SampleSplit] = (),
) -> tuple[SampleDroppedArchivePublishableCountRecord, ...]:
    """Build split-level archive-publishable counts for materialized dropped payloads."""
    summaries = build_sample_dropped_materialization_summary_records(
        entries,
        splits=splits,
    )
    return tuple(
        SampleDroppedArchivePublishableCountRecord(
            split=record.split,
            dropped_count=record.dropped_count,
            archive_publishable_count=record.archive_publishable_count,
        )
        for record in summaries
    )


def build_payload_completeness_records(
    payloads: Sequence[ProcessedSamplePayload],
) -> tuple[SamplePayloadCompletenessRecord, ...]:
    """Count payloads by optional diagnostic array presence."""
    counter = Counter(
        (payload.people_per_frame is not None, payload.frame_valid_mask is not None)
        for payload in payloads
    )
    return tuple(
        SamplePayloadCompletenessRecord(
            has_people_per_frame=has_people_per_frame,
            has_frame_valid_mask=has_frame_valid_mask,
            count=count,
        )
        for (has_people_per_frame, has_frame_valid_mask), count in sorted(counter.items())
    )


def build_sample_numeric_distribution_records(
    *,
    payloads: Sequence[ProcessedSamplePayload] = (),
    entries: Sequence[ManifestEntry] = (),
) -> tuple[SampleNumericDistributionRecord, ...]:
    """Build frame-count and FPS distributions for sample-owned surfaces."""
    records: list[SampleNumericDistributionRecord] = []
    if payloads:
        records.extend(
            (
                _distribution_record(
                    "payload",
                    "num_frames",
                    tuple(payload.num_frames for payload in payloads),
                ),
                _distribution_record(
                    "payload",
                    "fps",
                    tuple(payload.fps for payload in payloads),
                ),
            )
        )
    if entries:
        records.extend(
            (
                _distribution_record(
                    "manifest",
                    "num_frames",
                    tuple(entry.num_frames for entry in entries),
                ),
                _distribution_record(
                    "manifest",
                    "fps",
                    tuple(entry.fps for entry in entries),
                ),
            )
        )
    return tuple(records)


def _distribution_record(
    surface: str,
    field_name: str,
    values: Sequence[float | int | None],
) -> SampleNumericDistributionRecord:
    summary = numeric_summary(values)
    return SampleNumericDistributionRecord(
        surface=surface,
        field_name=field_name,
        sample_count=summary.sample_count,
        missing_count=summary.missing_count,
        minimum=summary.minimum,
        p50=summary.p50,
        p95=summary.p95,
        maximum=summary.maximum,
    )


def _dropped_materialization_summary_record(
    split: SampleSplit,
    dropped_entries: Sequence[DroppedManifestEntry],
) -> SampleDroppedMaterializationSummaryRecord:
    split_entries = tuple(entry for entry in dropped_entries if entry.split == split)
    outcome_counter = Counter(
        entry.materialization.debug_materialization_outcome
        for entry in split_entries
    )
    eligible_count = sum(
        1
        for entry in split_entries
        if entry.materialization.debug_materialization_eligible
    )
    attempted_count = sum(
        1
        for entry in split_entries
        if entry.materialization.debug_materialization_attempted
    )
    payload_exists_count = sum(
        1
        for entry in split_entries
        if entry.materialization.payload_exists
    )
    archive_publishable_count = sum(
        1
        for entry in split_entries
        if entry.materialization.archive_publishable
    )
    dropped_count = len(split_entries)
    return SampleDroppedMaterializationSummaryRecord(
        split=split,
        dropped_count=dropped_count,
        debug_materialization_eligible_count=eligible_count,
        debug_materialization_not_eligible_count=dropped_count - eligible_count,
        debug_materialization_attempted_count=attempted_count,
        debug_materialization_not_attempted_count=dropped_count - attempted_count,
        debug_materialization_succeeded_count=outcome_counter[
            DroppedDebugMaterializationOutcome.SUCCEEDED
        ],
        debug_materialization_failed_count=outcome_counter[
            DroppedDebugMaterializationOutcome.FAILED
        ],
        dropped_payload_exists_count=payload_exists_count,
        archive_publishable_count=archive_publishable_count,
    )
