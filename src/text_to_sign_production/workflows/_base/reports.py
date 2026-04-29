"""Persisted report and metadata artifact writers for the Base workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from text_to_sign_production.core.paths import repo_relative_path
from text_to_sign_production.modeling.contracts import (
    BASELINE_ID,
    BASELINE_ROLE,
    PREDICTION_MANIFEST_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    RISK_CONTROL_STATEMENT,
    baseline_architecture_spec,
    failure_mode_inventory_records,
)
from text_to_sign_production.modeling.data import M0_CHANNEL_POLICY, M0_TARGET_CHANNELS
from text_to_sign_production.modeling.inference.qualitative import (
    PREDICTION_ARTIFACTS_DIRNAME,
    REFERENCE_ARTIFACTS_DIRNAME,
    QualitativeExportResult,
)
from text_to_sign_production.modeling.inference.writer import SplitPredictionWriteResult
from text_to_sign_production.modeling.training.config import (
    BaselineTrainingConfig,
    baseline_config_to_dict,
)
from text_to_sign_production.modeling.training.train import BaselineTrainingRunResult
from text_to_sign_production.workflows._base.constants import (
    BASELINE_REPORT_SCHEMA_VERSION,
    CONFIDENCE_POLICY_WARNING,
    CONFIG_SNAPSHOT_SCHEMA_VERSION,
    FAILURE_MODE_REPORT_SCHEMA_VERSION,
    LENGTH_POLICY_WARNING,
    METRIC_LIMITATION_NOTE,
    RUN_SUMMARY_SCHEMA_VERSION,
)
from text_to_sign_production.workflows._base.types import (
    BaseRunLayout,
    BaseWorkflowConfig,
    BaseWorkflowError,
    BaseWorkflowInputError,
)
from text_to_sign_production.workflows._base.validate import _output_policy, _run_mode_policy
from text_to_sign_production.workflows._shared.metadata import (
    read_json_object,
    verify_portable_json_file,
    write_json_object,
)


def write_base_config_snapshot(
    path: Path,
    config: BaselineTrainingConfig,
    *,
    workflow_config: BaseWorkflowConfig,
    run_layout: BaseRunLayout,
    prediction_splits: Sequence[str],
    project_root: Path,
) -> Path:
    """Write source config and effective runtime settings into the run config directory."""

    run_mode_statement = _run_mode_policy(workflow_config.run_mode).run_mode_statement
    payload = {
        "schema_version": CONFIG_SNAPSHOT_SCHEMA_VERSION,
        "run_mode": workflow_config.run_mode,
        "run_mode_statement": run_mode_statement,
        "source_config_path": _portable_path(config.source_path, project_root=project_root),
        "source_config": config.raw_config,
        "effective_runtime": {
            "run_mode": workflow_config.run_mode,
            "run_mode_statement": run_mode_statement,
            "limit_train_samples": workflow_config.limit_train_samples,
            "limit_validation_samples": workflow_config.limit_validation_samples,
            "limit_prediction_samples": workflow_config.limit_prediction_samples,
            "max_epochs": workflow_config.epoch_count,
            "min_epochs": workflow_config.min_epochs,
            "early_stopping_patience": workflow_config.early_stopping_patience,
            "shuffle_train": workflow_config.shuffle_train,
            "prediction_splits": list(prediction_splits),
            "run_qualitative_export": workflow_config.run_qualitative_export,
            "panel_size": workflow_config.panel_size,
            "output_policy": _output_policy(workflow_config.output_policy).value,
            "live_log_path": _portable_path(
                run_layout.training_live_log_path,
                project_root=project_root,
            ),
        },
        "effective_paths": {
            "run_root": _portable_path(run_layout.run_root, project_root=project_root),
            "train_manifest": _portable_path(config.data.train_manifest, project_root=project_root),
            "val_manifest": _portable_path(config.data.val_manifest, project_root=project_root),
            "checkpoint_dir": _portable_path(
                config.checkpoint.output_dir,
                project_root=project_root,
            ),
            "training_output_dir": _portable_path(
                run_layout.training_dir,
                project_root=project_root,
            ),
        },
        "effective_config": _portable_config_dict(config, project_root=project_root),
    }
    write_json_object(path, payload)
    _verify_portable_json_file(path, project_root=project_root)
    return path


def write_base_run_summary(
    *,
    run_layout: BaseRunLayout,
    config: BaselineTrainingConfig,
    training_result: BaselineTrainingRunResult,
    prediction_results: Mapping[str, SplitPredictionWriteResult],
    qualitative_result: QualitativeExportResult | None,
    workflow_config: BaseWorkflowConfig,
    path_formatter: Callable[[Path], str],
) -> Path:
    """Write the root Base run summary with project-root-relative artifact paths."""

    run_mode_statement = _run_mode_policy(workflow_config.run_mode).run_mode_statement
    training_runtime_summary = _read_json_file(training_result.summary_path)
    surfaces = dict(training_runtime_summary.get("surfaces", {}))
    artifacts: dict[str, Any] = {
        "config": path_formatter(run_layout.config_snapshot_path),
        "baseline_config": path_formatter(run_layout.baseline_config_copy_path),
        "training_metrics": path_formatter(training_result.metrics_path),
        "training_summary": path_formatter(training_result.summary_path),
        "live_log": path_formatter(training_result.live_log_path),
        "checkpoints": {
            "best": (
                None
                if training_result.best_checkpoint_path is None
                else path_formatter(training_result.best_checkpoint_path)
            ),
            "last": path_formatter(training_result.last_checkpoint_path),
        },
        "prediction_manifests": {
            split: path_formatter(result.manifest_path)
            for split, result in prediction_results.items()
        },
        "reports": {
            "baseline_json": path_formatter(run_layout.baseline_report_json_path),
            "baseline_markdown": path_formatter(run_layout.baseline_report_markdown_path),
            "failure_modes_json": path_formatter(run_layout.failure_modes_json_path),
            "failure_modes_markdown": path_formatter(run_layout.failure_modes_markdown_path),
        },
    }
    if qualitative_result is None:
        artifacts["qualitative"] = None
    else:
        artifacts["qualitative"] = {
            "panel_definition": path_formatter(qualitative_result.panel_definition_path),
            "records": path_formatter(qualitative_result.records_path),
            "panel_summary": path_formatter(qualitative_result.panel_summary_path),
            "baseline_evidence_bundle": path_formatter(qualitative_result.evidence_bundle_path),
            "references_dir": path_formatter(
                qualitative_result.output_dir / REFERENCE_ARTIFACTS_DIRNAME,
            ),
            "predictions_dir": path_formatter(
                qualitative_result.output_dir / PREDICTION_ARTIFACTS_DIRNAME,
            ),
        }

    payload = {
        "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "run_name": run_layout.run_name,
        "run_mode": workflow_config.run_mode,
        "run_mode_statement": run_mode_statement,
        "baseline_id": BASELINE_ID,
        "role": BASELINE_ROLE,
        "run_root": path_formatter(run_layout.run_root),
        "splits": {
            "train": config.data.train_split,
            "validation": config.data.val_split,
            "prediction_splits": list(prediction_results),
        },
        "artifacts": artifacts,
        "risk_control": [
            "M0 is a comparison floor.",
            "M0 is not a contribution-strength claim.",
            "M0 is not a sign-intelligibility claim.",
        ],
        "training_surface": surfaces.get("training_surface"),
        "validation_surface": surfaces.get("validation_surface"),
        "primary_metric": {
            "name": training_result.best_metric_name,
            "best_value": training_result.best_metric_value,
            "best_epoch": training_result.best_epoch,
        },
        "target_standardization": {
            "enabled": workflow_config.run_mode == "complete",
            "coordinate_space": "canonical_fixed_canvas_xy",
            "fit_surface": (
                "quality train tier" if workflow_config.run_mode == "complete" else "disabled"
            ),
            "leakage_policy": "train-only fit",
            "metadata_path": surfaces.get("target_standardization_metadata_path"),
            "inverse_prediction_transform": workflow_config.run_mode == "complete",
        },
        "checkpoint_checksum_manifest_path": path_formatter(
            run_layout.checkpoints_dir / "checkpoint-manifest.json"
        ),
    }
    write_json_object(run_layout.run_summary_path, payload)
    _verify_portable_json_file(run_layout.run_summary_path, project_root=run_layout.project_root)
    return run_layout.run_summary_path


def write_baseline_report(
    *,
    run_layout: BaseRunLayout,
    config: BaselineTrainingConfig,
    training_result: BaselineTrainingRunResult,
    prediction_results: Mapping[str, SplitPredictionWriteResult],
    config_path: Path,
    workflow_config: BaseWorkflowConfig,
    sample_limits: Mapping[str, int | None],
    git_revision: str | None = None,
) -> tuple[Path, Path]:
    """Write source-level baseline report JSON and Markdown."""

    project_root = run_layout.project_root
    architecture = baseline_architecture_spec(channel_weights=config.loss.channel_weights)
    training_summary = _portable_json_paths(
        _read_json_file(training_result.summary_path),
        project_root=project_root,
    )
    surfaces = dict(cast(Mapping[str, Any], training_summary.get("surfaces", {})))
    run_mode_statement = _run_mode_policy(workflow_config.run_mode).run_mode_statement
    prediction_manifest_paths = {
        split: _portable_path(result.manifest_path, project_root=project_root)
        for split, result in prediction_results.items()
    }
    report_payload: dict[str, Any] = {
        "schema_version": BASELINE_REPORT_SCHEMA_VERSION,
        "run_name": run_layout.run_name,
        "run_mode": workflow_config.run_mode,
        "run_mode_statement": run_mode_statement,
        "git_revision": git_revision,
        "baseline": {
            "id": config.baseline.baseline_id,
            "name": config.baseline.name,
            "role": config.baseline.role,
        },
        "architecture_spec": architecture.to_dict(),
        "paths": {
            "config_path": _portable_path(config_path, project_root=project_root),
            "effective_config_snapshot_path": _portable_path(
                run_layout.config_snapshot_path,
                project_root=project_root,
            ),
            "training_metrics_path": _portable_path(
                training_result.metrics_path,
                project_root=project_root,
            ),
            "training_summary_path": _portable_path(
                training_result.summary_path,
                project_root=project_root,
            ),
            "best_checkpoint_path": (
                None
                if training_result.best_checkpoint_path is None
                else _portable_path(training_result.best_checkpoint_path, project_root=project_root)
            ),
            "last_checkpoint_path": _portable_path(
                training_result.last_checkpoint_path,
                project_root=project_root,
            ),
        },
        "splits": {
            "train": config.data.train_split,
            "validation": config.data.val_split,
            "prediction_splits": list(prediction_results),
        },
        "sample_limits": dict(sample_limits),
        "checkpoint_paths": {
            "best": (
                None
                if training_result.best_checkpoint_path is None
                else _portable_path(training_result.best_checkpoint_path, project_root=project_root)
            ),
            "last": _portable_path(training_result.last_checkpoint_path, project_root=project_root),
        },
        "training_metrics_summary": training_summary,
        "training_surface": surfaces.get("training_surface"),
        "validation_surface": surfaces.get("validation_surface"),
        "quality_tier_config_path": surfaces.get("quality_tier_config_path"),
        "quality_tier_config_hash": surfaces.get("quality_tier_config_hash"),
        "quality_tier_counts": surfaces.get("quality_tier_counts"),
        "primary_metric": {
            "name": training_result.best_metric_name,
            "best_value": training_result.best_metric_value,
            "best_epoch": training_result.best_epoch,
            "semantics": (
                "confidence-masked mean keypoint L2 in canonical fixed-canvas coordinate space; "
                "zero-filled unavailable reference keypoints do not contribute"
            ),
        },
        "prediction_manifest_paths": prediction_manifest_paths,
        "output_schema_versions": {
            "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
            "prediction_manifest_schema_version": PREDICTION_MANIFEST_SCHEMA_VERSION,
        },
        "length_policy_warning": LENGTH_POLICY_WARNING,
        "confidence_policy_warning": CONFIDENCE_POLICY_WARNING,
        "channel_policy": M0_CHANNEL_POLICY,
        "target_channels": list(M0_TARGET_CHANNELS),
        "risk_control_statement": RISK_CONTROL_STATEMENT,
        "metric_limitation_note": METRIC_LIMITATION_NOTE,
    }
    write_json_object(run_layout.baseline_report_json_path, report_payload)
    run_layout.baseline_report_markdown_path.write_text(
        _baseline_report_markdown(report_payload),
        encoding="utf-8",
    )
    return run_layout.baseline_report_json_path, run_layout.baseline_report_markdown_path


def write_failure_mode_report(run_layout: BaseRunLayout) -> tuple[Path, Path]:
    """Write expected M0 failure-mode inventory JSON and Markdown."""

    records = failure_mode_inventory_records()
    payload = {
        "schema_version": FAILURE_MODE_REPORT_SCHEMA_VERSION,
        "run_name": run_layout.run_name,
        "inventory_source": "text_to_sign_production.modeling.contracts",
        "failure_modes": records,
        "statement": (
            "This inventory is expected/diagnostic. M0 does not solve these failure modes. "
            "Observed failure analysis requires later evaluation/qualitative inspection."
        ),
    }
    write_json_object(run_layout.failure_modes_json_path, payload)
    run_layout.failure_modes_markdown_path.write_text(
        _failure_mode_report_markdown(records),
        encoding="utf-8",
    )
    return run_layout.failure_modes_json_path, run_layout.failure_modes_markdown_path


def _sample_limits(config: BaseWorkflowConfig) -> dict[str, int | None]:
    return {
        "limit_train_samples": config.limit_train_samples,
        "limit_validation_samples": config.limit_validation_samples,
        "limit_prediction_samples": config.limit_prediction_samples,
        "epoch_count": config.epoch_count,
    }


def _portable_config_dict(
    config: BaselineTrainingConfig,
    *,
    project_root: Path,
) -> dict[str, Any]:
    return baseline_config_to_dict(
        config,
        path_formatter=_project_relative_manifest_path_formatter(project_root=project_root),
    )


def _portable_path(path: Path, *, project_root: Path) -> str:
    try:
        return repo_relative_path(path, repo_root=project_root)
    except ValueError as exc:
        raise BaseWorkflowInputError(
            f"Path must stay under project_root for Base artifact metadata: {path}"
        ) from exc


def _project_relative_manifest_path_formatter(
    *,
    project_root: Path,
) -> Callable[[Path], str]:
    def format_path(path: Path) -> str:
        return _project_relative_manifest_path(path, project_root=project_root)

    return format_path


def _project_relative_manifest_path(path: Path, *, project_root: Path) -> str:
    try:
        return repo_relative_path(path, repo_root=project_root)
    except ValueError as exc:
        raise BaseWorkflowInputError(
            "Base prediction manifest paths must stay under project_root so they can be "
            f"written as project-root-relative paths: {path}"
        ) from exc


def _read_json_file(path: Path) -> dict[str, Any]:
    return read_json_object(path, error_factory=BaseWorkflowError)


def _portable_json_paths(value: Any, *, project_root: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _portable_json_paths(item, project_root=project_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_portable_json_paths(item, project_root=project_root) for item in value]
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            return _portable_path(candidate, project_root=project_root)
    return value


def _verify_portable_json_file(path: Path, *, project_root: Path) -> None:
    verify_portable_json_file(path, project_root=project_root, error_factory=BaseWorkflowError)


def _baseline_report_markdown(payload: Mapping[str, Any]) -> str:
    baseline = cast(Mapping[str, Any], payload["baseline"])
    paths = cast(Mapping[str, Any], payload["paths"])
    splits = cast(Mapping[str, Any], payload["splits"])
    prediction_manifest_paths = cast(Mapping[str, Any], payload["prediction_manifest_paths"])
    lines = [
        f"# {baseline['name']}",
        "",
        f"- Run: `{payload['run_name']}`",
        f"- Run mode: `{payload['run_mode']}`",
        f"- Run mode statement: {payload['run_mode_statement']}",
        f"- Role: `{baseline['role']}`",
        f"- Channel policy: `{payload['channel_policy']}`",
        f"- Train split: `{splits['train']}`",
        f"- Validation split: `{splits['validation']}`",
        f"- Prediction splits: `{', '.join(cast(list[str], splits['prediction_splits']))}`",
        f"- Training surface: `{payload.get('training_surface')}`",
        f"- Validation surface: `{payload.get('validation_surface')}`",
        f"- Primary metric: `{cast(Mapping[str, Any], payload['primary_metric'])['name']}`",
        f"- Config snapshot: `{paths['effective_config_snapshot_path']}`",
        f"- Training metrics: `{paths['training_metrics_path']}`",
        f"- Last checkpoint: `{paths['last_checkpoint_path']}`",
        f"- Best checkpoint: `{paths['best_checkpoint_path']}`",
        "",
        "## Prediction Manifests",
    ]
    for split, manifest_path in prediction_manifest_paths.items():
        lines.append(f"- `{split}`: `{manifest_path}`")
    lines.extend(
        [
            "",
            "## Warnings",
            f"- {payload['length_policy_warning']}",
            f"- {payload['confidence_policy_warning']}",
            f"- {payload['metric_limitation_note']}",
            "",
            "## Risk Control",
            str(payload["risk_control_statement"]),
            "",
        ]
    )
    return "\n".join(lines)


def _failure_mode_report_markdown(records: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# M0 Failure-Mode Inventory",
        "",
        "This inventory is expected/diagnostic.",
        "M0 does not solve these failure modes.",
        "Observed failure analysis requires later evaluation/qualitative inspection.",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## {record['label']}",
                "",
                f"- Identifier: `{record['identifier']}`",
                f"- Expected baseline risk: {record['expected_baseline_risk']}",
                f"- Expected qualitative observation: {record['expected_qualitative_observation']}",
                f"- solved_by_m0: `{str(record['solved_by_m0']).lower()}`",
                "",
            ]
        )
    return "\n".join(lines)


__all__: list[str] = []
