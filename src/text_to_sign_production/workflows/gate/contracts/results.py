from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.workflows.foundation.provenance import WrittenFileReceipt
from text_to_sign_production.workflows.gate.contracts.config import (
    GateWorkflowConfig,
)
from text_to_sign_production.workflows.gate.contracts.runtime import (
    GateRuntimeVerification,
    GateWorkflowExecutionInputs,
)


@dataclass(frozen=True, slots=True)
class GatePlannedReportOutputs:
    """Report paths planned by layout before report materialization."""

    summary_markdown_path: Path
    processing_summary_markdown_path: Path
    processing_detail_json_path: Path
    gate_summary_markdown_path: Path
    gate_detail_json_path: Path
    source_issue_summary_markdown_path: Path
    source_issue_detail_json_path: Path
    index_json_path: Path


@dataclass(frozen=True, slots=True)
class GateWrittenReportArtifacts:
    """Report artifacts after report write/materialization."""

    summary_markdown: WrittenFileReceipt
    processing_summary_markdown: WrittenFileReceipt
    processing_detail_json: WrittenFileReceipt
    gate_summary_markdown: WrittenFileReceipt
    gate_detail_json: WrittenFileReceipt
    source_issue_summary_markdown: WrittenFileReceipt
    source_issue_detail_json: WrittenFileReceipt
    index_json: WrittenFileReceipt

    @property
    def summary_markdown_path(self) -> Path:
        return self.summary_markdown.path

    @property
    def processing_summary_markdown_path(self) -> Path:
        return self.processing_summary_markdown.path

    @property
    def processing_detail_json_path(self) -> Path:
        return self.processing_detail_json.path

    @property
    def gate_summary_markdown_path(self) -> Path:
        return self.gate_summary_markdown.path

    @property
    def gate_detail_json_path(self) -> Path:
        return self.gate_detail_json.path

    @property
    def source_issue_summary_markdown_path(self) -> Path:
        return self.source_issue_summary_markdown.path

    @property
    def source_issue_detail_json_path(self) -> Path:
        return self.source_issue_detail_json.path

    @property
    def index_json_path(self) -> Path:
        return self.index_json.path

    @property
    def execution_id(self) -> str:
        return self.index_json.execution_id


@dataclass(frozen=True, slots=True)
class GateManifestOutputPlan:
    """Manifest path planned by layout before manifest materialization."""

    partition: str
    split: str
    path: Path


@dataclass(frozen=True, slots=True)
class GateWrittenManifestArtifact:
    """Manifest artifact after manifest write/materialization."""

    partition: str
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
class GateWrittenPayloadArtifact:
    """Gate sample payload artifact after payload materialization."""

    status: str
    split: str
    sample_id: str
    payload_ref: str
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
class GateWorkflowOutputSummary:
    planned_manifest_outputs: tuple[GateManifestOutputPlan, ...]
    passed_samples_root: Path
    dropped_samples_root: Path
    planned_report_outputs: GatePlannedReportOutputs


@dataclass(frozen=True, slots=True)
class GateWorkflowResult:
    execution_id: str
    config: GateWorkflowConfig
    execution_inputs: GateWorkflowExecutionInputs
    runtime_verification: GateRuntimeVerification
    output_summary: GateWorkflowOutputSummary
