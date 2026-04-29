"""Base workflow output verification helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from text_to_sign_production.core.files import require_dir, verify_output_file
from text_to_sign_production.core.paths import repo_relative_path
from text_to_sign_production.modeling.inference.qualitative import QualitativeExportResult
from text_to_sign_production.modeling.inference.schemas import validate_prediction_sample_payload
from text_to_sign_production.workflows._base.layout import _resolve_manifest_or_repo_path
from text_to_sign_production.workflows._base.types import (
    BasePredictionSplitVerification,
    BaseQualitativeVerificationSummary,
    BaseRunLayout,
    BaseRunVerificationSummary,
    BaseWorkflowError,
)
from text_to_sign_production.workflows._base.validate import _positive_optional_limit
from text_to_sign_production.workflows._shared.metadata import (
    count_jsonl_records,
    read_jsonl_records,
    verify_portable_json_file,
    verify_portable_jsonl_file,
    verify_portable_text_file,
)


def verify_base_run_outputs(
    *,
    run_layout: BaseRunLayout,
    prediction_splits: Iterable[str],
    limit_prediction_samples: int | None = None,
    qualitative_result: QualitativeExportResult | None = None,
) -> BaseRunVerificationSummary:
    """Verify canonical M0 run outputs and validate one prediction payload per split."""

    _positive_optional_limit(limit_prediction_samples, label="limit_prediction_samples")
    _verify_base_run_root_policy(run_layout)
    config_snapshot_path = verify_output_file(
        run_layout.config_snapshot_path,
        label="Base config snapshot",
    )
    baseline_config_copy_path = verify_output_file(
        run_layout.baseline_config_copy_path,
        label="Base baseline config copy",
    )
    training_metrics_path = verify_output_file(
        run_layout.training_metrics_path,
        label="Base training metrics",
    )
    training_live_log_path = verify_output_file(
        run_layout.training_live_log_path,
        label="Base training live log",
    )
    training_metric_record_count = count_jsonl_records(training_metrics_path)
    if training_metric_record_count <= 0:
        raise BaseWorkflowError(f"Base training metrics are empty: {training_metrics_path}")
    training_summary_path = verify_output_file(
        run_layout.training_summary_path,
        label="Base training summary",
    )
    best_checkpoint_path = verify_output_file(
        run_layout.best_checkpoint_path,
        label="Base best checkpoint",
    )
    last_checkpoint_path = verify_output_file(
        run_layout.last_checkpoint_path,
        label="Base last checkpoint",
    )

    split_summaries: dict[str, BasePredictionSplitVerification] = {}
    for split in prediction_splits:
        split_summaries[split] = _verify_prediction_split_outputs(
            run_layout,
            split=split,
        )

    baseline_report_json_path = verify_output_file(
        run_layout.baseline_report_json_path,
        label="Base baseline report JSON",
    )
    baseline_report_markdown_path = verify_output_file(
        run_layout.baseline_report_markdown_path,
        label="Base baseline report Markdown",
    )
    failure_modes_json_path = verify_output_file(
        run_layout.failure_modes_json_path,
        label="Base failure-mode report JSON",
    )
    failure_modes_markdown_path = verify_output_file(
        run_layout.failure_modes_markdown_path,
        label="Base failure-mode report Markdown",
    )
    qualitative = (
        None if qualitative_result is None else _verify_qualitative_outputs(qualitative_result)
    )
    run_summary_path = verify_output_file(
        run_layout.run_summary_path,
        label="Base run summary",
    )
    _verify_base_runtime_metadata_paths(
        run_layout=run_layout,
        prediction_summaries=split_summaries,
        qualitative=qualitative,
    )

    return BaseRunVerificationSummary(
        run_root=run_layout.run_root,
        config_snapshot_path=config_snapshot_path,
        baseline_config_copy_path=baseline_config_copy_path,
        training_metrics_path=training_metrics_path,
        training_live_log_path=training_live_log_path,
        training_metric_record_count=training_metric_record_count,
        training_summary_path=training_summary_path,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        prediction_splits=split_summaries,
        baseline_report_json_path=baseline_report_json_path,
        baseline_report_markdown_path=baseline_report_markdown_path,
        failure_modes_json_path=failure_modes_json_path,
        failure_modes_markdown_path=failure_modes_markdown_path,
        qualitative=qualitative,
        run_summary_path=run_summary_path,
    )


def _verify_base_run_root_policy(run_layout: BaseRunLayout) -> None:
    relative_run_root = Path(
        repo_relative_path(run_layout.run_root, repo_root=run_layout.project_root)
    )
    expected_prefix = Path("runs") / "base" / "m0-direct-text-to-full-bfh"
    if relative_run_root.parent != expected_prefix:
        raise BaseWorkflowError(
            f"Base run root must live under the canonical Base M0 runs tree: {run_layout.run_root}"
        )


def _verify_base_runtime_metadata_paths(
    *,
    run_layout: BaseRunLayout,
    prediction_summaries: Mapping[str, BasePredictionSplitVerification],
    qualitative: BaseQualitativeVerificationSummary | None,
) -> None:
    for path in (
        run_layout.config_snapshot_path,
        run_layout.training_summary_path,
        run_layout.run_summary_path,
        run_layout.baseline_report_json_path,
        run_layout.failure_modes_json_path,
    ):
        _verify_portable_json_file(path, project_root=run_layout.project_root)
    for path in (
        run_layout.baseline_report_markdown_path,
        run_layout.failure_modes_markdown_path,
    ):
        _verify_portable_text_file(path, project_root=run_layout.project_root)
    for summary in prediction_summaries.values():
        _verify_portable_jsonl_file(summary.manifest_path, project_root=run_layout.project_root)
    if qualitative is not None:
        _verify_portable_json_file(
            qualitative.panel_definition_path,
            project_root=run_layout.project_root,
        )
        _verify_portable_json_file(
            qualitative.panel_summary_path,
            project_root=run_layout.project_root,
        )
        _verify_portable_json_file(
            qualitative.evidence_bundle_path,
            project_root=run_layout.project_root,
        )
        _verify_portable_jsonl_file(
            qualitative.records_path,
            project_root=run_layout.project_root,
        )


def _verify_portable_json_file(path: Path, *, project_root: Path) -> None:
    verify_portable_json_file(path, project_root=project_root, error_factory=BaseWorkflowError)


def _verify_portable_jsonl_file(path: Path, *, project_root: Path) -> None:
    verify_portable_jsonl_file(path, project_root=project_root, error_factory=BaseWorkflowError)


def _verify_portable_text_file(path: Path, *, project_root: Path) -> None:
    verify_portable_text_file(path, project_root=project_root, error_factory=BaseWorkflowError)


def _verify_prediction_split_outputs(
    run_layout: BaseRunLayout,
    *,
    split: str,
) -> BasePredictionSplitVerification:
    manifest_path = verify_output_file(
        run_layout.prediction_manifest_path(split),
        label=f"Base {split} prediction manifest",
    )
    manifest_record_count = count_jsonl_records(manifest_path)
    if manifest_record_count <= 0:
        raise BaseWorkflowError(f"Base {split} prediction manifest is empty: {manifest_path}")
    samples_dir = require_dir(
        run_layout.prediction_samples_dir(split),
        label=f"Base {split} prediction samples",
    )
    sample_paths = sorted(samples_dir.glob("*.npz"))
    if not sample_paths:
        raise BaseWorkflowError(f"No Base {split} prediction samples were produced: {samples_dir}")
    validated_sample_path = _validate_one_prediction_payload(
        manifest_path,
        project_root=run_layout.project_root,
    )
    return BasePredictionSplitVerification(
        split=split,
        manifest_path=manifest_path,
        manifest_record_count=manifest_record_count,
        samples_dir=samples_dir,
        sample_count=len(sample_paths),
        validated_sample_path=validated_sample_path,
    )


def _validate_one_prediction_payload(
    manifest_path: Path,
    *,
    project_root: Path,
) -> Path | None:
    for record in read_jsonl_records(manifest_path):
        raw_path = record.get("prediction_sample_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise BaseWorkflowError(
                f"Prediction manifest row is missing prediction_sample_path: {manifest_path}"
            )
        prediction_path = _resolve_manifest_or_repo_path(raw_path, project_root=project_root)
        with np.load(prediction_path, allow_pickle=False) as sample:
            payload = {key: sample[key] for key in sample.files}
            validate_prediction_sample_payload(payload)
        return prediction_path
    return None


def _verify_qualitative_outputs(
    result: QualitativeExportResult,
) -> BaseQualitativeVerificationSummary:
    output_dir = require_dir(result.output_dir, label="Base qualitative output directory")
    panel_definition_path = verify_output_file(
        result.panel_definition_path,
        label="Base qualitative panel definition",
    )
    records_path = verify_output_file(
        result.records_path,
        label="Base qualitative records",
    )
    record_count = count_jsonl_records(records_path)
    if record_count != result.sample_count:
        raise BaseWorkflowError(
            "Base qualitative record count mismatch: expected "
            f"{result.sample_count}, observed {record_count}: {records_path}"
        )
    panel_summary_path = verify_output_file(
        result.panel_summary_path,
        label="Base qualitative summary",
    )
    evidence_bundle_path = verify_output_file(
        result.evidence_bundle_path,
        label="Base qualitative evidence bundle",
    )
    if result.sample_count <= 0:
        raise BaseWorkflowError("Base qualitative export must contain at least one sample.")
    if len(result.sample_ids) != result.sample_count:
        raise BaseWorkflowError(
            "Base qualitative sample id count mismatch: expected "
            f"{result.sample_count}, observed {len(result.sample_ids)}."
        )
    return BaseQualitativeVerificationSummary(
        output_dir=output_dir,
        panel_definition_path=panel_definition_path,
        records_path=records_path,
        record_count=record_count,
        panel_summary_path=panel_summary_path,
        evidence_bundle_path=evidence_bundle_path,
        sample_count=result.sample_count,
        sample_ids=result.sample_ids,
    )


__all__: list[str] = []
