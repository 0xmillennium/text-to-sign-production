from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.workflows.tiers.contracts.config import TiersWorkflowConfig
from text_to_sign_production.workflows.tiers.contracts.runtime import (
    TiersRuntimeVerification,
    TiersWorkflowExecutionInputs,
)


@dataclass(frozen=True, slots=True)
class TiersReportArtifacts:
    summary_markdown_path: Path
    calibration_markdown_path: Path
    decision_detail_jsonl_path: Path
    calibration_surfaces_json_path: Path
    calibration_detail_json_path: Path
    index_json_path: Path


@dataclass(frozen=True, slots=True)
class TiersTieredManifestOutput:
    tier: str
    membership: str
    split: str
    path: Path


@dataclass(frozen=True, slots=True)
class TiersWorkflowOutputSummary:
    loaded_passed_sample_count: int
    quality_fact_count: int
    quality_context_count: int
    quality_metric_bundle_count: int
    leakage_sample_summary_count: int
    tier_decision_count: int
    quality_report_count: int
    tiered_manifest_outputs: tuple[TiersTieredManifestOutput, ...]
    report_artifacts: TiersReportArtifacts


@dataclass(frozen=True, slots=True)
class TiersWorkflowResult:
    config: TiersWorkflowConfig
    execution_inputs: TiersWorkflowExecutionInputs
    runtime_verification: TiersRuntimeVerification
    output_summary: TiersWorkflowOutputSummary
