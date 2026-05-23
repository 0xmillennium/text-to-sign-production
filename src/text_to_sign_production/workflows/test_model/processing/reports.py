"""Write human- and machine-readable test_model reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from text_to_sign_production.core.progress import ProgressSession, ProgressTaskHandle
from text_to_sign_production.workflows.foundation.provenance import written_file_receipt
from text_to_sign_production.workflows.foundation.review import write_json, write_markdown
from text_to_sign_production.workflows.test_model.contracts import (
    TestModelCheckpointSelection,
    TestModelInferenceResult,
    TestModelReferenceComparisonResult,
    TestModelReportResult,
    TestModelRunResolution,
    TestModelSampleEvidence,
    TestModelTargetResolution,
    TestModelVisualizationResult,
)
from text_to_sign_production.workflows.test_model.constants import TEST_MODEL_STAGE_REPORT_WRITE
from text_to_sign_production.workflows.test_model.layout import (
    runtime_test_model_sample_run_root,
)
from text_to_sign_production.workflows.test_model.progress import test_model_progress_stage


def write_test_model_reports(
    layout,
    *,
    model_run: TestModelRunResolution,
    checkpoint: TestModelCheckpointSelection,
    target: TestModelTargetResolution,
    evidence: TestModelSampleEvidence,
    inference: TestModelInferenceResult,
    comparison: TestModelReferenceComparisonResult,
    visualization: TestModelVisualizationResult,
    execution_id: str,
    progress_session: ProgressSession | None = None,
) -> TestModelReportResult:
    output_root = runtime_test_model_sample_run_root(
        layout,
        model_run.model_run_name,
        target.target_sentence_name,
        execution_id,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "test_model_result.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope_statement": (
            "single unseen test sample diagnostic; not full test split evaluation; "
            "not aggregate model performance"
        ),
        "model_run_name": model_run.model_run_name,
        "model_key": model_run.model_key,
        "manifest_family": (
            None
            if model_run.manifest_family is None
            else model_run.manifest_family.family_id
        ),
        "checkpoint_policy": checkpoint.policy.value,
        "checkpoint_path": checkpoint.checkpoint_path,
        "target_sentence_name": target.target_sentence_name,
        "resolved_sample_id": target.resolved_sample_id,
        "source_sentence_name": target.source_sentence_name,
        "split": target.split.value,
        "generated_pose_artifact": None
        if inference.provider_result is None
        else inference.provider_result.generated_payload_path,
        "visual_outputs": [artifact.path for artifact in visualization.artifacts if artifact.created],
        "comparison_status": comparison.status,
        "comparison_metrics": _metrics_by_key(comparison),
        "warnings": (
            model_run.warnings
            + checkpoint.warnings
            + target.warnings
            + evidence.warnings
            + inference.warnings
            + comparison.warnings
            + visualization.warnings
        ),
        "errors": (
            model_run.errors
            + checkpoint.errors
            + target.issues
            + evidence.errors
            + inference.errors
            + comparison.errors
            + visualization.errors
        ),
    }
    planned_files = (
        output_root / "test_model_summary.md",
        output_root / "test_model_result.json",
        output_root / "reference_comparison.json",
        output_root / "reference_comparison_summary.md",
        output_root / "model_run_context.json",
        output_root / "checkpoint_selection.json",
        output_root / "sample_evidence.json",
        output_root / "inference_result.json",
        output_root / "visual_artifacts.json",
        output_root / "index.json",
    )
    progress_task = (
        progress_session.task(
            test_model_progress_stage(
                stage_id=TEST_MODEL_STAGE_REPORT_WRITE,
                label="test_model report write",
                unit="file",
                owner_module=__name__,
                operation_kind="report_write",
                total_semantics="test_model report files written",
                allowed_counters=("written",),
            ),
            total=len(planned_files),
        )
        if progress_session is not None
        else None
    )
    try:
        files = (
            _record_report_write(
                progress_task, _write_markdown(planned_files[0], _summary(payload, comparison))
            ),
            _record_report_write(
                progress_task, _write_json(planned_files[1], payload),
            ),
            _record_report_write(
                progress_task,
                _write_json(planned_files[2], comparison),
            ),
            _record_report_write(
                progress_task,
                _write_markdown(planned_files[3], _reference_comparison_summary(comparison)),
            ),
            _record_report_write(
                progress_task,
                _write_json(planned_files[4], model_run),
            ),
            _record_report_write(
                progress_task,
                _write_json(planned_files[5], checkpoint),
            ),
            _record_report_write(
                progress_task,
                _write_json(planned_files[6], evidence),
            ),
            _record_report_write(
                progress_task,
                _write_json(planned_files[7], inference),
            ),
            _record_report_write(
                progress_task,
                _write_json(planned_files[8], visualization),
            ),
            _record_report_write(
                progress_task,
                _write_json(
                    planned_files[9],
                    {
                        "schema_version": "test_model_report_index.v1",
                        "files": [path.name for path in planned_files],
                        "scope_statement": payload["scope_statement"],
                        "comparison_status": comparison.status,
                        "comparison_metrics": payload["comparison_metrics"],
                    },
                ),
            ),
        )
    finally:
        if progress_task is not None:
            progress_task.close()
    receipts = tuple(
        written_file_receipt(
            f"test_model report {path.name}",
            path,
            execution_id=execution_id,
            kind="test_model_result" if path.name == "test_model_result.json" else "test_model_report",
        )
        for path in files
    )
    return TestModelReportResult(
        output_root=output_root,
        files=files,
        receipts=receipts,
        succeeded=not payload["errors"],
        warnings=tuple(str(item) for item in payload["warnings"]),
        errors=tuple(str(item) for item in payload["errors"]),
    )


def _summary(payload: dict[str, Any], comparison: TestModelReferenceComparisonResult) -> str:
    return "\n".join(
        (
            "# Test Model Summary",
            "",
            "This is a single unseen test sample diagnostic. It is not full test split evaluation and not aggregate model performance.",
            "",
            f"- MODEL_RUN_NAME: `{payload['model_run_name']}`",
            f"- Resolved model key: `{payload['model_key']}`",
            f"- Manifest family: `{payload['manifest_family']}`",
            f"- Checkpoint policy: `{payload['checkpoint_policy']}`",
            f"- Checkpoint path: `{_json_value(payload['checkpoint_path'])}`",
            f"- Target sentence name: `{payload['target_sentence_name']}`",
            f"- Resolved test sample: `{payload['resolved_sample_id']}`",
            f"- Source sentence name: `{payload['source_sentence_name']}`",
            f"- Generated pose artifact: `{_json_value(payload['generated_pose_artifact'])}`",
            f"- Visual output count: `{len(payload['visual_outputs'])}`",
            "",
            "## Reference-vs-generated diagnostic",
            "",
            "This compares the generated pose against the selected test split reference keypoints.",
            "It is a single-sample diagnostic, not aggregate test performance and not human intelligibility evaluation.",
            "",
            f"- Comparison status: `{comparison.status}`",
            f"- Reference frame count: `{_display_value(comparison.reference_frame_count)}`",
            f"- Generated frame count: `{_display_value(comparison.generated_frame_count)}`",
            f"- Aligned frame count: `{_display_value(comparison.aligned_frame_count)}`",
            f"- Sequence length absolute error: `{_metric_value(comparison, 'sequence_length_absolute_error')}`",
            f"- Masked L1 mean: `{_metric_value(comparison, 'masked_l1_mean')}`",
            f"- Masked L2 mean: `{_metric_value(comparison, 'masked_l2_mean')}`",
            f"- Velocity L1 mean: `{_metric_value(comparison, 'velocity_l1_mean')}`",
            f"- Velocity L2 mean: `{_metric_value(comparison, 'velocity_l2_mean')}`",
            f"- Valid joint coverage: `{_metric_value(comparison, 'valid_joint_coverage')}`",
            "",
            _channel_metrics_table(comparison),
            "",
        )
    )


def _reference_comparison_summary(result: TestModelReferenceComparisonResult) -> str:
    return "\n".join(
        (
            "# Reference-vs-generated Diagnostic",
            "",
            "This compares the generated pose against the selected test split reference keypoints.",
            "It is a single-sample diagnostic, not aggregate test performance and not human intelligibility evaluation.",
            "",
            f"- Comparison status: `{result.status}`",
            f"- Reference sample id: `{_display_value(result.reference_sample_id)}`",
            f"- Generated sample id: `{_display_value(result.generated_sample_id)}`",
            f"- Reference frame count: `{_display_value(result.reference_frame_count)}`",
            f"- Generated frame count: `{_display_value(result.generated_frame_count)}`",
            f"- Aligned frame count: `{_display_value(result.aligned_frame_count)}`",
            f"- Sequence length absolute error: `{_metric_value(result, 'sequence_length_absolute_error')}`",
            f"- Masked L1 mean: `{_metric_value(result, 'masked_l1_mean')}`",
            f"- Masked L2 mean: `{_metric_value(result, 'masked_l2_mean')}`",
            f"- Velocity L1 mean: `{_metric_value(result, 'velocity_l1_mean')}`",
            f"- Velocity L2 mean: `{_metric_value(result, 'velocity_l2_mean')}`",
            f"- Valid joint coverage: `{_metric_value(result, 'valid_joint_coverage')}`",
            "",
            _channel_metrics_table(result),
            "",
        )
    )


def _metrics_by_key(result: TestModelReferenceComparisonResult) -> dict[str, float | None]:
    return {metric.metric_key: metric.value for metric in result.metrics}


def _metric_value(result: TestModelReferenceComparisonResult, metric_key: str) -> str:
    for metric in result.metrics:
        if metric.metric_key == metric_key:
            return _display_value(metric.value)
    return "n/a"


def _channel_metric_value(
    result: TestModelReferenceComparisonResult,
    channel: str,
    metric_key: str,
) -> str:
    for metric in result.channel_metrics:
        if metric.channel == channel and metric.metric_key == metric_key:
            return _display_value(metric.value)
    return "n/a"


def _channel_metrics_table(result: TestModelReferenceComparisonResult) -> str:
    rows = [
        "### Channel metrics",
        "",
        "| channel | masked_l1 | masked_l2 | velocity_l1 | velocity_l2 | coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    channels = tuple(dict.fromkeys(metric.channel for metric in result.channel_metrics))
    for channel in channels:
        rows.append(
            "| "
            + " | ".join(
                (
                    channel,
                    _channel_metric_value(result, channel, "channel_masked_l1_mean"),
                    _channel_metric_value(result, channel, "channel_masked_l2_mean"),
                    _channel_metric_value(result, channel, "channel_velocity_l1_mean"),
                    _channel_metric_value(result, channel, "channel_velocity_l2_mean"),
                    _channel_metric_value(result, channel, "channel_valid_joint_coverage"),
                )
            )
            + " |"
        )
    return "\n".join(rows)


def _display_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_markdown(path: Path, text: str) -> Path:
    write_markdown(path, text)
    return path


def _write_json(path: Path, value: Any) -> Path:
    write_json(path, _json_value(value))
    return path


def _record_report_write(task: ProgressTaskHandle | None, path: Path) -> Path:
    if task is not None:
        task.advance(1, counters={"written": 1})
    return path


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if is_dataclass(value):
        to_metadata_dict = getattr(value, "to_metadata_dict", None)
        if callable(to_metadata_dict):
            return _json_value(to_metadata_dict())
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = ["write_test_model_reports"]
