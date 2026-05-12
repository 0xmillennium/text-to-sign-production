from __future__ import annotations

from uuid import uuid4

from text_to_sign_production.core.progress import ProgressSession, TqdmProgressSink
from text_to_sign_production.data.gate.policies import load_gates_config
from text_to_sign_production.workflows.foundation.provenance import (
    PhysicalSampleIdentity,
    SemanticSentenceIdentity,
    written_file_receipt,
)
from text_to_sign_production.workflows.gate.constants import GATE_WORKFLOW_NAME
from text_to_sign_production.workflows.gate.contracts import (
    GateManifestOutputPlan,
    GatePlannedReportOutputs,
    GateRuntimeVerification,
    GateWorkflowConfig,
    GateWorkflowExecutionInputs,
    GateWorkflowInvariantError,
    GateWorkflowOutputSummary,
    GateWorkflowResult,
    GateWrittenPayloadArtifact,
)
from text_to_sign_production.workflows.gate.layout import GateLayout
from text_to_sign_production.workflows.gate.processing.manifests import (
    write_gate_manifests,
)
from text_to_sign_production.workflows.gate.processing.models import (
    GateExecutionBundle,
    GatePayloadOutput,
    GateSplitProcessingResult,
)
from text_to_sign_production.workflows.gate.processing.split import process_gate_split


def execute_gate_processing(
    *,
    config: GateWorkflowConfig,
    layout: GateLayout,
    execution_inputs: GateWorkflowExecutionInputs,
    runtime_verification: GateRuntimeVerification,
    progress_session: ProgressSession | None = None,
) -> GateExecutionBundle:
    if not runtime_verification.succeeded:
        raise GateWorkflowInvariantError("Runtime verification must succeed before processing")

    progress_session = _visible_progress_session(progress_session)
    execution_id = uuid4().hex
    gates_config = load_gates_config(execution_inputs.gates_config_path)
    split_results = tuple(
        process_gate_split(
            config=config,
            split_inputs=split_inputs,
            layout=layout,
            gates_config=gates_config,
            progress_session=progress_session,
        )
        for split_inputs in execution_inputs.split_inputs
    )
    workflow_result = GateWorkflowResult(
        execution_id=execution_id,
        config=config,
        execution_inputs=execution_inputs,
        runtime_verification=runtime_verification,
        output_summary=_build_output_summary(layout),
    )
    written_manifest_artifacts = write_gate_manifests(
        layout=layout,
        config_splits=config.splits,
        split_results=split_results,
        execution_id=execution_id,
    )
    return GateExecutionBundle(
        workflow_result=workflow_result,
        split_results=split_results,
        written_payload_artifacts=_written_payload_artifacts(split_results, execution_id),
        written_manifest_artifacts=written_manifest_artifacts,
    )


def _visible_progress_session(
    progress_session: ProgressSession | None,
) -> ProgressSession:
    if progress_session is not None:
        return progress_session
    return ProgressSession(
        workflow_id=GATE_WORKFLOW_NAME,
        sink=TqdmProgressSink(),
    )


def _build_output_summary(layout: GateLayout) -> GateWorkflowOutputSummary:
    return GateWorkflowOutputSummary(
        planned_manifest_outputs=tuple(
            GateManifestOutputPlan(
                partition=manifest.partition,
                split=manifest.split,
                path=manifest.path,
            )
            for manifest in layout.outputs.manifest_outputs
        ),
        passed_samples_root=layout.outputs.passed_samples_root,
        dropped_samples_root=layout.outputs.dropped_samples_root,
        planned_report_outputs=GatePlannedReportOutputs(
            summary_markdown_path=layout.reports.summary_markdown_path,
            processing_summary_jsonl_path=layout.reports.processing_summary_jsonl_path,
            processing_detail_jsonl_path=layout.reports.processing_detail_jsonl_path,
            gate_summary_jsonl_path=layout.reports.gate_summary_jsonl_path,
            gate_detail_jsonl_path=layout.reports.gate_detail_jsonl_path,
            source_issue_summary_jsonl_path=layout.reports.source_issue_summary_jsonl_path,
            source_issue_detail_jsonl_path=layout.reports.source_issue_detail_jsonl_path,
            index_json_path=layout.reports.index_json_path,
        ),
    )


def _written_payload_artifacts(
    split_results: tuple[GateSplitProcessingResult, ...],
    execution_id: str,
) -> tuple[GateWrittenPayloadArtifact, ...]:
    artifacts: list[GateWrittenPayloadArtifact] = []
    for split_result in split_results:
        payloads = (
            *split_result.passed_payloads,
            *split_result.dropped_debug_payloads,
        )
        for payload in payloads:
            artifacts.append(_written_payload_artifact(payload, execution_id))
    return tuple(artifacts)


def _written_payload_artifact(
    payload: GatePayloadOutput,
    execution_id: str,
) -> GateWrittenPayloadArtifact:
    return GateWrittenPayloadArtifact(
        status=payload.status.value,
        split=payload.sample.source.split.value,
        sample_id=payload.sample.source.sample_id,
        payload_ref=payload.payload_ref,
        receipt=written_file_receipt(
            f"gate payload [{payload.status.value}/{payload.sample.source.split.value}/"
            f"{payload.sample.source.sample_id}]",
            payload.path,
            execution_id=execution_id,
            kind="prepared_sample_payload",
            physical_sample=PhysicalSampleIdentity(
                split=payload.sample.source.split.value,
                sample_id=payload.sample.source.sample_id,
            ),
            semantic_sentence=SemanticSentenceIdentity(
                sentence_id=payload.sample.source.source_sentence_id,
            ),
            source_lineage=(
                payload.sample.source.source_video_id,
                payload.sample.source.source_sentence_name,
            ),
        ),
    )
