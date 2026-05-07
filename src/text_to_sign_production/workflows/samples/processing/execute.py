from __future__ import annotations

from text_to_sign_production.core.progress import ProgressSession, TqdmProgressSink
from text_to_sign_production.data.gates import load_gates_config
from text_to_sign_production.workflows.samples.constants import SAMPLES_WORKFLOW_NAME
from text_to_sign_production.workflows.samples.contracts import (
    SamplesManifestOutput,
    SamplesReportArtifacts,
    SamplesRuntimeVerification,
    SamplesWorkflowConfig,
    SamplesWorkflowExecutionInputs,
    SamplesWorkflowInvariantError,
    SamplesWorkflowOutputSummary,
    SamplesWorkflowResult,
)
from text_to_sign_production.workflows.samples.layout import SamplesLayout
from text_to_sign_production.workflows.samples.processing.manifests import (
    write_samples_manifests,
)
from text_to_sign_production.workflows.samples.processing.models import (
    SamplesExecutionBundle,
)
from text_to_sign_production.workflows.samples.processing.split import process_samples_split


def execute_samples_processing(
    *,
    config: SamplesWorkflowConfig,
    layout: SamplesLayout,
    execution_inputs: SamplesWorkflowExecutionInputs,
    runtime_verification: SamplesRuntimeVerification,
    progress_session: ProgressSession | None = None,
) -> SamplesExecutionBundle:
    if not runtime_verification.succeeded:
        raise SamplesWorkflowInvariantError("Runtime verification must succeed before processing")

    progress_session = _visible_progress_session(progress_session)
    gates_config = load_gates_config(execution_inputs.gates_config_path)
    split_results = tuple(
        process_samples_split(
            config=config,
            split_inputs=split_inputs,
            layout=layout,
            gates_config=gates_config,
            progress_session=progress_session,
        )
        for split_inputs in execution_inputs.split_inputs
    )
    execution_bundle = SamplesExecutionBundle(
        workflow_result=SamplesWorkflowResult(
            config=config,
            execution_inputs=execution_inputs,
            runtime_verification=runtime_verification,
            output_summary=_build_output_summary(layout),
        ),
        split_results=split_results,
    )
    write_samples_manifests(
        layout=layout,
        execution_bundle=execution_bundle,
        progress_session=progress_session,
    )
    return execution_bundle


def _visible_progress_session(
    progress_session: ProgressSession | None,
) -> ProgressSession:
    if progress_session is not None:
        return progress_session
    return ProgressSession(
        workflow_id=SAMPLES_WORKFLOW_NAME,
        sink=TqdmProgressSink(),
    )


def _build_output_summary(layout: SamplesLayout) -> SamplesWorkflowOutputSummary:
    return SamplesWorkflowOutputSummary(
        manifest_outputs=tuple(
            SamplesManifestOutput(
                partition=manifest.partition,
                split=manifest.split,
                path=manifest.path,
            )
            for manifest in layout.outputs.manifest_outputs
        ),
        passed_samples_root=layout.outputs.passed_samples_root,
        dropped_samples_root=layout.outputs.dropped_samples_root,
        report_artifacts=SamplesReportArtifacts(
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
