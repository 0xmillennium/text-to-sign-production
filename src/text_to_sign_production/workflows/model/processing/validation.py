"""Workflow orchestration for validation-split generated/reference comparison."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.candidates import is_generated_pose_manifest_role
from text_to_sign_production.modeling.data import (
    ModelingDataError,
    load_generated_pose_surface,
    load_manifest_sample,
    read_modeling_manifest,
    resolve_prepared_payload_path,
)
from text_to_sign_production.modeling.validation import (
    ModelValidationError,
    ValidationGeneratedInput,
    ValidationPairingKey,
    ValidationReferenceInput,
    aggregate_validation_metric_results,
    build_validation_limitations,
    build_validation_pairing,
    compute_validation_channel_metric_results,
    compute_validation_metric_results,
    write_validation_aggregate_metrics_json,
    write_validation_channel_metric_results_jsonl,
    write_validation_limitations_json,
    write_validation_metric_results_jsonl,
    write_validation_pairing_jsonl,
)
from text_to_sign_production.workflows.foundation.provenance import written_file_receipt
from text_to_sign_production.workflows.foundation.review import write_markdown
from text_to_sign_production.workflows.model.contracts import (
    ModelStageExecutionWorkflowResult,
    ModelValidationArtifactResult,
    ModelWorkflowInvariantError,
)
from text_to_sign_production.workflows.model.layout import ModelLayout


def write_model_validation_artifacts(
    *,
    layout: ModelLayout,
    stage_execution: ModelStageExecutionWorkflowResult,
    execution_id: str,
) -> ModelValidationArtifactResult:
    """Compute and write workflow-owned validation-split artifacts."""

    request = stage_execution.stage_plan.request
    if request.validation_split is not SampleSplit.VAL:
        raise ModelWorkflowInvariantError("model validation harness requires validation_split='val'.")
    if not stage_execution.execution.completed:
        raise ModelWorkflowInvariantError("model validation requires completed provider stage execution.")
    generated_manifest_path = find_final_validation_generated_pose_manifest_path(
        layout,
        stage_execution,
    )
    if generated_manifest_path is None or not generated_manifest_path.is_file():
        raise ModelWorkflowInvariantError(
            "generated validation manifest is missing. The provider must export a generated-pose "
            "manifest for prediction split 'val' before model validation runs."
        )
    try:
        reference_manifest = read_modeling_manifest(
            layout.stores.runtime,
            request.manifest_family,
            SampleSplit.VAL,
        )
        generated_samples = load_generated_pose_surface(
            layout.stores.runtime,
            producer_key=request.model_key.value,
            run_name=request.run_name,
            split=SampleSplit.VAL,
        )
        reference_inputs = tuple(
            ValidationReferenceInput(
                entry,
                resolve_prepared_payload_path(layout.stores.runtime, entry),
            )
            for entry in reference_manifest.entries
        )
        generated_inputs = tuple(
            ValidationGeneratedInput(sample.entry, sample.payload_path)
            for sample in generated_samples
        )
        pairing = build_validation_pairing(
            split=SampleSplit.VAL,
            reference_entries=reference_inputs,
            generated_entries=generated_inputs,
        )
        limitations = build_validation_limitations(split=SampleSplit.VAL, pairing_entries=pairing)
        if limitations.paired_count == 0:
            raise ModelWorkflowInvariantError(
                "model validation found no paired validation samples. Check generated validation "
                "outputs and the selected manifest family; no validation artifacts were reported."
            )
        reference_by_key = {
            ValidationPairingKey(SampleSplit.VAL, value.entry.sample_id, 0): value
            for value in reference_inputs
        }
        generated_by_key = {
            ValidationPairingKey(
                SampleSplit.VAL,
                value.entry.sample_id,
                value.entry.generation_index,
            ): sample
            for value, sample in zip(generated_inputs, generated_samples, strict=True)
        }
        metric_results = []
        channel_metric_results = []
        for pair in pairing:
            if pair.status != "paired":
                continue
            reference_input = reference_by_key[pair.key]
            generated = generated_by_key[pair.key]
            if generated.sample is None:
                raise ModelWorkflowInvariantError(
                    f"paired generated validation payload was not loaded: {pair.key.sample_id!r}."
                )
            reference = load_manifest_sample(
                layout.stores.runtime,
                request.manifest_family,
                reference_manifest.manifest_path,
                reference_input.entry,
            )
            metric_results.extend(
                compute_validation_metric_results(
                    key=pair.key,
                    reference=reference.pose,
                    generated=generated.sample.pose,
                )
            )
            channel_metric_results.extend(
                compute_validation_channel_metric_results(
                    key=pair.key,
                    reference=reference.pose,
                    generated=generated.sample.pose,
                )
            )
        missing_count = (
            limitations.generated_missing_count
            + limitations.reference_missing_count
            + limitations.failed_generated_count
            + limitations.identity_mismatch_count
        )
        issue_count = (
            sum(len(pair.issues) for pair in pairing)
            + sum(len(result.issues) for result in metric_results)
            + sum(len(result.issues) for result in channel_metric_results)
        )
        aggregates = aggregate_validation_metric_results(
            metric_results,
            split=SampleSplit.VAL,
            missing_count=missing_count,
            issue_count=issue_count,
        )
        channel_aggregates = aggregate_validation_metric_results(
            channel_metric_results,
            split=SampleSplit.VAL,
            missing_count=missing_count,
            issue_count=issue_count,
        )
    except ModelWorkflowInvariantError:
        raise
    except (FileNotFoundError, OSError, ModelingDataError, ModelValidationError, ValueError) as exc:
        raise ModelWorkflowInvariantError(f"model validation inputs are invalid: {exc}") from exc

    write_validation_pairing_jsonl(layout.reports.validation_pairing_manifest_path, pairing)
    write_validation_metric_results_jsonl(layout.reports.validation_metric_results_path, metric_results)
    write_validation_channel_metric_results_jsonl(
        layout.reports.validation_channel_metric_results_path,
        channel_metric_results,
    )
    write_validation_aggregate_metrics_json(layout.reports.validation_aggregate_metrics_path, aggregates)
    write_validation_aggregate_metrics_json(
        layout.reports.validation_channel_aggregate_metrics_path,
        channel_aggregates,
    )
    write_validation_limitations_json(layout.reports.validation_limitations_path, limitations)
    write_markdown(
        layout.reports.validation_summary_report_path,
        _validation_summary_markdown(
            request.manifest_family.family_id,
            limitations,
            aggregates,
            channel_aggregates,
        ),
    )
    receipts = (
        written_file_receipt(
            "model validation pairing manifest",
            layout.reports.validation_pairing_manifest_path,
            execution_id=execution_id,
            kind="model_validation_pairing_manifest",
        ),
        written_file_receipt(
            "model validation metric results",
            layout.reports.validation_metric_results_path,
            execution_id=execution_id,
            kind="model_validation_metric_results",
        ),
        written_file_receipt(
            "model validation channel metric results",
            layout.reports.validation_channel_metric_results_path,
            execution_id=execution_id,
            kind="model_validation_channel_metric_results",
        ),
        written_file_receipt(
            "model validation aggregate metrics",
            layout.reports.validation_aggregate_metrics_path,
            execution_id=execution_id,
            kind="model_validation_aggregate_metrics",
        ),
        written_file_receipt(
            "model validation channel aggregate metrics",
            layout.reports.validation_channel_aggregate_metrics_path,
            execution_id=execution_id,
            kind="model_validation_channel_aggregate_metrics",
        ),
        written_file_receipt(
            "model validation limitations",
            layout.reports.validation_limitations_path,
            execution_id=execution_id,
            kind="model_validation_limitations",
        ),
        written_file_receipt(
            "model validation summary report",
            layout.reports.validation_summary_report_path,
            execution_id=execution_id,
            kind="model_validation_summary_report",
        ),
    )
    return ModelValidationArtifactResult(
        pairing_manifest_path=layout.reports.validation_pairing_manifest_path,
        metric_results_path=layout.reports.validation_metric_results_path,
        channel_metric_results_path=layout.reports.validation_channel_metric_results_path,
        aggregate_metrics_path=layout.reports.validation_aggregate_metrics_path,
        channel_aggregate_metrics_path=layout.reports.validation_channel_aggregate_metrics_path,
        limitations_path=layout.reports.validation_limitations_path,
        summary_markdown_path=layout.reports.validation_summary_report_path,
        paired_count=limitations.paired_count,
        missing_generated_count=limitations.generated_missing_count,
        missing_reference_count=limitations.reference_missing_count,
        failed_generated_count=limitations.failed_generated_count,
        identity_mismatch_count=limitations.identity_mismatch_count,
        receipts=receipts,
    )


def find_final_validation_generated_pose_manifest_path(
    layout: ModelLayout,
    stage_execution: ModelStageExecutionWorkflowResult,
) -> Path | None:
    """Find only a final validation generated-pose export, with old-export tolerance."""

    for artifact in stage_execution.execution.artifact_refs:
        if not is_generated_pose_manifest_role(artifact.role):
            continue
        if (
            artifact.metadata.get("split") == SampleSplit.VAL.value
            and artifact.metadata.get("artifact_subtype") == "final_validation"
        ):
            return artifact.path
    for artifact in stage_execution.execution.artifact_refs:
        if not is_generated_pose_manifest_role(artifact.role):
            continue
        if (
            artifact.metadata.get("split") == SampleSplit.VAL.value
            and artifact.metadata.get("artifact_subtype") is None
            and artifact.metadata.get("producer_stage") == "export_generated_pose"
        ):
            return artifact.path
    for split_output in layout.outputs.generated_pose_split_outputs:
        if split_output.split is SampleSplit.VAL:
            return split_output.manifest_path
    return None


def _validation_summary_markdown(manifest_family, limitations, aggregates, channel_aggregates) -> str:
    aggregate_rows = "\n".join(
        f"- `{item.metric_key}`: count=`{item.count}`, mean=`{item.mean}`"
        for item in aggregates
    )
    channel_rows = "\n".join(
        f"- `{item.metric_key}`: count=`{item.count}`, mean=`{item.mean}`"
        for item in channel_aggregates
    )
    notes = "\n".join(f"- {note}" for note in limitations.notes)
    return (
        "# Model Validation Summary\n\n"
        f"- Manifest family: `{manifest_family}`\n"
        "- Split: `val`\n"
        f"- Paired samples: `{limitations.paired_count}`\n"
        f"- Missing generated samples: `{limitations.generated_missing_count}`\n"
        f"- Missing reference samples: `{limitations.reference_missing_count}`\n"
        f"- Failed generated samples: `{limitations.failed_generated_count}`\n"
        f"- Identity mismatches: `{limitations.identity_mismatch_count}`\n\n"
        "## Aggregate Metrics\n\n"
        f"{aggregate_rows}\n\n"
        "## Channel Metrics\n\n"
        f"{channel_rows}\n\n"
        "## Limitations\n\n"
        f"{notes}\n"
    )


__all__ = [
    "find_final_validation_generated_pose_manifest_path",
    "write_model_validation_artifacts",
]
