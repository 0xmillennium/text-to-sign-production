from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.workflows.foundation.provenance import WrittenFileReceipt
from text_to_sign_production.workflows.tier.contracts.config import TierWorkflowConfig
from text_to_sign_production.workflows.tier.contracts.runtime import (
    TierRuntimeVerification,
    TierWorkflowExecutionInputs,
)


@dataclass(frozen=True, slots=True)
class TierPlannedReportOutputs:
    """Report paths planned by layout before report materialization."""

    summary_markdown_path: Path
    calibration_markdown_path: Path
    decision_detail_json_path: Path
    calibration_surfaces_json_path: Path
    calibration_detail_json_path: Path
    index_json_path: Path


@dataclass(frozen=True, slots=True)
class TierWrittenReportArtifacts:
    """Report artifacts after report write/materialization."""

    summary_markdown: WrittenFileReceipt
    calibration_markdown: WrittenFileReceipt
    decision_detail_json: WrittenFileReceipt
    calibration_surfaces_json: WrittenFileReceipt
    calibration_detail_json: WrittenFileReceipt
    index_json: WrittenFileReceipt

    @property
    def summary_markdown_path(self) -> Path:
        return self.summary_markdown.path

    @property
    def calibration_markdown_path(self) -> Path:
        return self.calibration_markdown.path

    @property
    def decision_detail_json_path(self) -> Path:
        return self.decision_detail_json.path

    @property
    def calibration_surfaces_json_path(self) -> Path:
        return self.calibration_surfaces_json.path

    @property
    def calibration_detail_json_path(self) -> Path:
        return self.calibration_detail_json.path

    @property
    def index_json_path(self) -> Path:
        return self.index_json.path

    @property
    def execution_id(self) -> str:
        return self.index_json.execution_id


@dataclass(frozen=True, slots=True)
class TieredManifestOutputPlan:
    """Tiered manifest path planned by layout before manifest materialization."""

    tier: str
    membership: str
    split: str
    path: Path


@dataclass(frozen=True, slots=True)
class TieredWrittenManifestArtifact:
    """Tiered manifest artifact after manifest write/materialization."""

    tier: str
    membership: str
    split: str
    receipt: WrittenFileReceipt

    @property
    def path(self) -> Path:
        return self.receipt.path

    @property
    def sha256(self) -> str:
        return self.receipt.sha256

    @property
    def execution_id(self) -> str:
        return self.receipt.execution_id


@dataclass(frozen=True, slots=True)
class TierWorkflowOutputSummary:
    loaded_passed_sample_count: int
    quality_fact_count: int
    quality_context_count: int
    quality_metric_bundle_count: int
    leakage_sample_summary_count: int
    tier_decision_count: int
    tier_report_count: int
    planned_tiered_manifest_outputs: tuple[TieredManifestOutputPlan, ...]
    planned_report_outputs: TierPlannedReportOutputs


@dataclass(frozen=True, slots=True)
class TierWorkflowResult:
    execution_id: str
    config: TierWorkflowConfig
    execution_inputs: TierWorkflowExecutionInputs
    runtime_verification: TierRuntimeVerification
    output_summary: TierWorkflowOutputSummary
