"""Package-local analysis surfaces for gate decisions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.legacy_data.gates.types import (
    GateBlockerSummaryRecord,
    GateDropStageCountRecord,
    GateProcessingStatusCountRecord,
    GateReasonFrequencyRecord,
    GateSplitBlockerSummaryRecord,
    GateSplitStageSummaryRecord,
    GateStage,
    GateStageResultCountRecord,
    GateStatus,
    ProcessingDecision,
    ProcessingStatus,
)


def build_processing_status_count_records(
    decisions: Sequence[ProcessingDecision],
) -> tuple[GateProcessingStatusCountRecord, ...]:
    """Count final processing decisions by status."""
    counter = Counter(decision.status for decision in decisions)
    return tuple(
        GateProcessingStatusCountRecord(status=status, count=counter[status])
        for status in ProcessingStatus
        if counter[status] > 0
    )


def build_drop_stage_count_records(
    decisions: Sequence[ProcessingDecision],
) -> tuple[GateDropStageCountRecord, ...]:
    """Count dropped decisions by drop stage."""
    counter = Counter(
        decision.drop_stage for decision in decisions if decision.drop_stage is not None
    )
    return tuple(
        GateDropStageCountRecord(stage=stage, count=counter[stage])
        for stage in GateStage
        if counter[stage] > 0
    )


def build_gate_reason_frequency_records(
    decisions: Sequence[ProcessingDecision],
) -> tuple[GateReasonFrequencyRecord, ...]:
    """Count gate reason codes across per-gate results."""
    counter: Counter[tuple[GateStage | None, str]] = Counter()
    for decision in decisions:
        for stage, result in decision.gate_results.items():
            for reason in result.reasons:
                counter[(stage, reason)] += 1
    return tuple(
        GateReasonFrequencyRecord(stage=stage, reason=reason, count=count)
        for (stage, reason), count in sorted(
            counter.items(), key=lambda item: (_stage_sort_key(item[0][0]), item[0][1])
        )
    )


def build_gate_stage_result_count_records(
    decisions: Sequence[ProcessingDecision],
) -> tuple[GateStageResultCountRecord, ...]:
    """Count pass/drop outcomes for each individual gate stage."""
    counter = Counter(
        (stage, result.status)
        for decision in decisions
        for stage, result in decision.gate_results.items()
    )
    return tuple(
        GateStageResultCountRecord(stage=stage, status=status, count=count)
        for (stage, status), count in sorted(
            counter.items(), key=lambda item: (item[0][0].value, item[0][1].value)
        )
    )


def build_gate_split_stage_summary_records(
    decisions_by_split: Mapping[SampleSplit, Sequence[ProcessingDecision]],
) -> tuple[GateSplitStageSummaryRecord, ...]:
    """Build split-aware pass/drop summaries for each gate stage."""
    counter = Counter(
        (split, stage, result.status)
        for split, decisions in decisions_by_split.items()
        for decision in decisions
        for stage, result in decision.gate_results.items()
    )
    split_stages = sorted(
        {(split, stage) for split, stage, _status in counter},
        key=lambda item: (item[0].value, item[1].value),
    )
    return tuple(
        GateSplitStageSummaryRecord(
            split=split,
            stage=stage,
            passed_count=counter[(split, stage, GateStatus.PASSED)],
            dropped_count=counter[(split, stage, GateStatus.DROPPED)],
        )
        for split, stage in split_stages
    )


def build_gate_blocker_summary_records(
    decisions: Sequence[ProcessingDecision],
) -> tuple[GateBlockerSummaryRecord, ...]:
    """Count drop-stage blockers by reason."""
    counter: Counter[tuple[GateStage, str]] = Counter()
    for decision in decisions:
        if decision.drop_stage is None:
            continue
        for reason in decision.drop_reasons:
            counter[(decision.drop_stage, reason)] += 1
    return tuple(
        GateBlockerSummaryRecord(stage=stage, reason=reason, count=count)
        for (stage, reason), count in sorted(
            counter.items(), key=lambda item: (item[0][0].value, item[0][1])
        )
    )


def build_gate_split_blocker_summary_records(
    decisions_by_split: Mapping[SampleSplit, Sequence[ProcessingDecision]],
) -> tuple[GateSplitBlockerSummaryRecord, ...]:
    """Build split-aware drop-stage blocker counts by reason."""
    counter: Counter[tuple[SampleSplit, GateStage, str]] = Counter()
    for split, decisions in decisions_by_split.items():
        for decision in decisions:
            if decision.drop_stage is None:
                continue
            for reason in decision.drop_reasons:
                counter[(split, decision.drop_stage, reason)] += 1
    return tuple(
        GateSplitBlockerSummaryRecord(
            split=split,
            stage=stage,
            reason=reason,
            count=count,
        )
        for (split, stage, reason), count in sorted(
            counter.items(),
            key=lambda item: (item[0][0].value, item[0][1].value, item[0][2]),
        )
    )


def _stage_sort_key(stage: GateStage | None) -> tuple[int, str]:
    if stage is None:
        return (0, "")
    return (1, stage.value)
