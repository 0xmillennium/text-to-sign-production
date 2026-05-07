from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.workflows.samples.contracts.config import (
    SamplesWorkflowConfig,
)
from text_to_sign_production.workflows.samples.contracts.runtime import (
    SamplesRuntimeVerification,
    SamplesWorkflowExecutionInputs,
)


@dataclass(frozen=True, slots=True)
class SamplesReportArtifacts:
    summary_markdown_path: Path
    processing_summary_jsonl_path: Path
    processing_detail_jsonl_path: Path
    gate_summary_jsonl_path: Path
    gate_detail_jsonl_path: Path
    source_issue_summary_jsonl_path: Path
    source_issue_detail_jsonl_path: Path
    index_json_path: Path


@dataclass(frozen=True, slots=True)
class SamplesManifestOutput:
    partition: str
    split: str
    path: Path


@dataclass(frozen=True, slots=True)
class SamplesWorkflowOutputSummary:
    manifest_outputs: tuple[SamplesManifestOutput, ...]
    passed_samples_root: Path
    dropped_samples_root: Path
    report_artifacts: SamplesReportArtifacts


@dataclass(frozen=True, slots=True)
class SamplesWorkflowResult:
    config: SamplesWorkflowConfig
    execution_inputs: SamplesWorkflowExecutionInputs
    runtime_verification: SamplesRuntimeVerification
    output_summary: SamplesWorkflowOutputSummary
