"""Structural gating and decision evaluation layer."""

from __future__ import annotations

from text_to_sign_production.legacy_data.gates.analysis import (
    build_drop_stage_count_records,
    build_gate_blocker_summary_records,
    build_gate_reason_frequency_records,
    build_gate_split_blocker_summary_records,
    build_gate_split_stage_summary_records,
    build_gate_stage_result_count_records,
    build_processing_status_count_records,
)
from text_to_sign_production.legacy_data.gates.config import load_gates_config
from text_to_sign_production.legacy_data.gates.evaluate import (
    evaluate_sample_processing,
    evaluate_unmatched_source,
)
from text_to_sign_production.legacy_data.gates.types import (
    GateBlockerSummaryRecord,
    GateDropStageCountRecord,
    GateProcessingStatusCountRecord,
    GateReasonFrequencyRecord,
    GateResult,
    GatesConfig,
    GateSplitBlockerSummaryRecord,
    GateSplitStageSummaryRecord,
    GateStage,
    GateStageResultCountRecord,
    GateStatus,
    GateValidationIssue,
    ProcessingDecision,
    ProcessingStatus,
)
from text_to_sign_production.legacy_data.gates.validate import validate_decision

__all__ = [
    "GatesConfig",
    "GateBlockerSummaryRecord",
    "GateDropStageCountRecord",
    "GateProcessingStatusCountRecord",
    "GateReasonFrequencyRecord",
    "GateResult",
    "GateSplitBlockerSummaryRecord",
    "GateSplitStageSummaryRecord",
    "GateStage",
    "GateStageResultCountRecord",
    "GateStatus",
    "GateValidationIssue",
    "ProcessingDecision",
    "ProcessingStatus",
    "build_drop_stage_count_records",
    "build_gate_blocker_summary_records",
    "build_gate_split_blocker_summary_records",
    "build_gate_split_stage_summary_records",
    "build_gate_reason_frequency_records",
    "build_gate_stage_result_count_records",
    "build_processing_status_count_records",
    "evaluate_sample_processing",
    "evaluate_unmatched_source",
    "load_gates_config",
    "validate_decision",
]
