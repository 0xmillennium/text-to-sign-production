"""Dataset workflow execution orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from text_to_sign_production.artifacts.index import build_sample_index
from text_to_sign_production.artifacts.layout import DatasetArtifactLayout
from text_to_sign_production.core.files import ensure_dir, verify_output_dir
from text_to_sign_production.core.paths import ProjectLayout, repo_relative_path
from text_to_sign_production.core.progress import StdoutProgressReporter
from text_to_sign_production.data.filtering import filter_all_splits
from text_to_sign_production.data.manifests import export_final_manifests
from text_to_sign_production.data.normalize import normalize_all_splits
from text_to_sign_production.data.raw import build_raw_manifests
from text_to_sign_production.data.schemas import ValidationIssue
from text_to_sign_production.workflows._dataset.layout import (
    _dataset_output_paths,
    _dataset_run_root,
    _filter_config_path,
    _layout_from_config,
    _quality_config_path,
    _split_paths_by_split,
)
from text_to_sign_production.workflows._dataset.reports import (
    _interim_assumption_reports_root,
    _interim_filter_reports_root,
    remove_stale_processed_report_files,
    remove_stale_split_files,
    write_interim_assumption_reports,
    write_interim_filter_reports,
    write_processed_reports,
    write_quality_tier_manifests,
)
from text_to_sign_production.workflows._dataset.types import (
    DatasetWorkflowConfig,
    DatasetWorkflowError,
    DatasetWorkflowInputError,
    DatasetWorkflowResult,
)
from text_to_sign_production.workflows._dataset.validate import (
    _raise_for_validation_errors,
    _validate_splits,
    _validate_stage_outputs,
    _verify_files,
    validate_dataset_inputs,
)


def run_dataset_workflow(config: DatasetWorkflowConfig) -> DatasetWorkflowResult:
    """Run raw, normalized, filtered, and processed manifest stages in order."""

    validate_dataset_inputs(config)
    layout = _layout_from_config(config)
    artifact_layout = DatasetArtifactLayout(layout)
    splits = _validate_splits(config.splits)
    filter_config_path = _filter_config_path(config, layout)
    quality_config_path = _quality_config_path(config, layout)
    paths = _dataset_output_paths(layout=layout, splits=splits)
    run_root = _dataset_run_root(layout)
    split_paths_by_split = _split_paths_by_split(layout, splits)
    validation_issues: dict[str, tuple[ValidationIssue, ...]] = {}

    try:
        ensure_dir(run_root, label="Dataset workflow run root")
        assumption_report = build_raw_manifests(
            splits=splits,
            split_paths_by_split=split_paths_by_split,
            raw_manifests_root=artifact_layout.raw_manifests_root,
            raw_root=artifact_layout.how2sign_root,
            data_root=layout.data_root,
        )
        write_interim_assumption_reports(assumption_report, layout=layout, splits=splits)
        _verify_files(paths["raw_manifest_paths"], label="Raw manifest")
        _verify_files(paths["interim_assumption_report_paths"], label="Interim assumption report")
        remove_stale_split_files(
            artifact_layout.raw_manifests_root,
            filename_template="raw_{split}.jsonl",
            requested_splits=splits,
        )
        remove_stale_split_files(
            _interim_assumption_reports_root(layout),
            filename_template="{split}.json",
            requested_splits=splits,
        )
        _validate_stage_outputs(
            validation_issues,
            stage="raw",
            manifest_paths=paths["raw_manifest_paths"],
            kind="raw",
            data_root=layout.data_root,
            enabled=config.validate_outputs,
        )

        normalize_all_splits(
            splits=splits,
            raw_manifests_root=artifact_layout.raw_manifests_root,
            normalized_manifests_root=artifact_layout.normalized_manifests_root,
            processed_samples_root=artifact_layout.processed_v1_samples_root,
            data_root=layout.data_root,
        )
        _verify_files(paths["normalized_manifest_paths"], label="Normalized manifest")
        remove_stale_split_files(
            artifact_layout.normalized_manifests_root,
            filename_template="normalized_{split}.jsonl",
            requested_splits=splits,
        )
        _validate_stage_outputs(
            validation_issues,
            stage="normalized",
            manifest_paths=paths["normalized_manifest_paths"],
            kind="normalized",
            data_root=layout.data_root,
            enabled=config.validate_outputs,
        )

        filter_report = filter_all_splits(
            filter_config_path,
            splits=splits,
            normalized_manifests_root=artifact_layout.normalized_manifests_root,
            filtered_manifests_root=artifact_layout.filtered_manifests_root,
            data_root=layout.data_root,
        )
        write_interim_filter_reports(filter_report, layout=layout, splits=splits)
        _verify_files(paths["filtered_manifest_paths"], label="Filtered manifest")
        _verify_files(paths["interim_filter_report_paths"], label="Interim filter report")
        remove_stale_split_files(
            artifact_layout.filtered_manifests_root,
            filename_template="filtered_{split}.jsonl",
            requested_splits=splits,
        )
        remove_stale_split_files(
            _interim_filter_reports_root(layout),
            filename_template="{split}.json",
            requested_splits=splits,
        )
        _validate_stage_outputs(
            validation_issues,
            stage="filtered",
            manifest_paths=paths["filtered_manifest_paths"],
            kind="normalized",
            data_root=layout.data_root,
            enabled=config.validate_outputs,
        )
        export_report = export_final_manifests(
            assumption_report=assumption_report,
            filter_report=filter_report,
            splits=splits,
            raw_manifests_root=artifact_layout.raw_manifests_root,
            filtered_manifests_root=artifact_layout.filtered_manifests_root,
            processed_manifests_root=artifact_layout.processed_v1_manifests_root,
            processed_samples_root=artifact_layout.processed_v1_samples_root,
            data_root=layout.data_root,
            quality_config_path=quality_config_path,
            reporter=StdoutProgressReporter(prefix="[dataset]"),
        )
        write_processed_reports(
            assumption_report=assumption_report,
            export_report=export_report,
            layout=layout,
            splits=splits,
        )
        tier_manifest_paths = write_quality_tier_manifests(
            export_report=export_report,
            layout=layout,
            splits=splits,
        )
        _verify_files(paths["processed_manifest_paths"], label="Processed manifest")
        remove_stale_split_files(
            artifact_layout.processed_v1_manifests_root,
            filename_template="{split}.jsonl",
            requested_splits=splits,
        )
        _verify_files(paths["processed_report_paths"], label="Processed report")
        remove_stale_processed_report_files(layout=layout, requested_splits=splits)
        for split in splits:
            verify_output_dir(
                artifact_layout.processed_samples_split_root(split),
                label=f"{split} processed samples root",
            )
        _validate_stage_outputs(
            validation_issues,
            stage="processed",
            manifest_paths=paths["processed_manifest_paths"],
            kind="processed",
            data_root=layout.data_root,
            enabled=config.validate_outputs,
        )
        processed_sample_archive_members = _collect_processed_sample_archive_members(
            layout=layout,
            splits=splits,
        )
    except DatasetWorkflowInputError:
        raise
    except DatasetWorkflowError:
        raise
    except Exception as exc:
        raise DatasetWorkflowError(f"Dataset workflow failed: {exc}") from exc

    if config.validate_outputs and config.fail_on_validation_errors:
        _raise_for_validation_errors(validation_issues)

    return DatasetWorkflowResult(
        project_root=layout.root,
        data_root=layout.data_root,
        run_root=run_root,
        splits=splits,
        raw_manifest_paths=paths["raw_manifest_paths"],
        normalized_manifest_paths=paths["normalized_manifest_paths"],
        filtered_manifest_paths=paths["filtered_manifest_paths"],
        processed_manifest_paths=paths["processed_manifest_paths"],
        processed_samples_root=paths["processed_samples_root"],
        interim_report_paths=paths["interim_report_paths"],
        processed_report_paths=paths["processed_report_paths"],
        quality_report_paths=paths["quality_report_paths"],
        leakage_report_paths=paths["leakage_report_paths"],
        tier_manifest_paths=tier_manifest_paths,
        tier_counts=_tier_counts(export_report),
        blocking_leakage=_blocking_leakage(export_report),
        quality_config_hash=_quality_config_hash(export_report),
        processed_sample_archive_members=processed_sample_archive_members,
        assumption_report=assumption_report,
        filter_report=filter_report,
        export_report=export_report,
        validation_issues=validation_issues,
    )


def _collect_processed_sample_archive_members(
    *,
    layout: ProjectLayout,
    splits: tuple[str, ...],
) -> dict[str, tuple[Path, ...]]:
    artifact_layout = DatasetArtifactLayout(layout)
    sample_index = build_sample_index(
        layout.data_root,
        splits=splits,
        require_sample_files=True,
        processed_manifests_root=artifact_layout.processed_v1_manifests_root,
        processed_samples_root=artifact_layout.processed_v1_samples_root,
        filtered_manifests_root=artifact_layout.filtered_manifests_root,
    )
    members_by_split: dict[str, list[Path]] = {split: [] for split in splits}
    for record in sample_index.records:
        project_relative = Path(repo_relative_path(record.sample_path, repo_root=layout.root))
        expected_root = Path("data") / "processed" / "v1" / "samples" / record.split
        if not project_relative.is_relative_to(expected_root) or project_relative.suffix != ".npz":
            raise ValueError(
                "Processed sample archive member must be a project-relative .npz under "
                f"{expected_root.as_posix()}: {project_relative.as_posix()}"
            )
        members_by_split[record.split].append(project_relative)
    return {split: tuple(members_by_split[split]) for split in splits}


def _tier_counts(export_report: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    quality_outputs = export_report.get("quality_outputs")
    if not isinstance(quality_outputs, dict):
        return {}
    tier_counts = quality_outputs.get("tier_counts")
    if not isinstance(tier_counts, dict):
        return {}
    return {
        str(split): {
            str(tier): int(count) for tier, count in dict(counts).items() if isinstance(count, int)
        }
        for split, counts in tier_counts.items()
        if isinstance(counts, dict)
    }


def _blocking_leakage(export_report: Mapping[str, Any]) -> bool:
    quality_outputs = export_report.get("quality_outputs")
    if not isinstance(quality_outputs, dict):
        return False
    leakage_report = quality_outputs.get("leakage_report")
    if not isinstance(leakage_report, dict):
        return False
    return bool(leakage_report.get("blocking_for_complete", False))


def _quality_config_hash(export_report: Mapping[str, Any]) -> str | None:
    quality_outputs = export_report.get("quality_outputs")
    if not isinstance(quality_outputs, dict):
        return None
    quality_config = quality_outputs.get("quality_config")
    if not isinstance(quality_config, dict):
        return None
    value = quality_config.get("sha256")
    return value if isinstance(value, str) else None


__all__ = ["run_dataset_workflow"]
