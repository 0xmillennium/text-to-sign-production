from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.artifacts.store import (
    ArtifactStores,
    build_artifact_stores,
)
from text_to_sign_production.core import build_repo_roots
from text_to_sign_production.workflows.tiers.contracts.config import TiersWorkflowConfig


@dataclass(frozen=True, slots=True)
class TiersDriveSplitLayout:
    split: str
    passed_manifest_path: Path
    passed_archive_path: Path


@dataclass(frozen=True, slots=True)
class TiersRuntimeSplitLayout:
    split: str
    passed_manifest_path: Path
    passed_samples_split_root: Path


@dataclass(frozen=True, slots=True)
class TiersRuntimeLayout:
    filters_config_path: Path
    tiers_config_path: Path
    passed_samples_root: Path
    splits: tuple[TiersRuntimeSplitLayout, ...]


@dataclass(frozen=True, slots=True)
class TiersOutputLayout:
    tiered_manifests_root: Path


@dataclass(frozen=True, slots=True)
class TiersReportLayout:
    summary_markdown_path: Path
    calibration_markdown_path: Path
    decision_detail_jsonl_path: Path
    calibration_surfaces_json_path: Path
    calibration_detail_json_path: Path
    index_json_path: Path


@dataclass(frozen=True, slots=True)
class TiersPublishLayout:
    root: Path
    report_targets_root: Path
    summary_markdown_target_path: Path
    calibration_markdown_target_path: Path
    decision_detail_target_path: Path
    calibration_surfaces_target_path: Path
    calibration_detail_target_path: Path
    index_json_target_path: Path
    tiered_manifest_targets_root: Path


@dataclass(frozen=True, slots=True)
class TiersLayout:
    config: TiersWorkflowConfig
    stores: ArtifactStores
    drive_splits: tuple[TiersDriveSplitLayout, ...]
    runtime: TiersRuntimeLayout
    outputs: TiersOutputLayout
    reports: TiersReportLayout
    publish: TiersPublishLayout


def build_tiers_layout(config: TiersWorkflowConfig) -> TiersLayout:
    stores = build_artifact_stores(
        build_repo_roots(config.project_root),
        build_repo_roots(config.drive_project_root),
    )
    return TiersLayout(
        config=config,
        stores=stores,
        drive_splits=_build_drive_split_layouts(config, stores),
        runtime=_build_runtime_layout(config, stores),
        outputs=_build_output_layout(stores),
        reports=_build_report_layout(stores),
        publish=_build_publish_layout(config, stores),
    )


def _build_drive_split_layouts(
    config: TiersWorkflowConfig,
    stores: ArtifactStores,
) -> tuple[TiersDriveSplitLayout, ...]:
    return tuple(
        TiersDriveSplitLayout(
            split=split,
            passed_manifest_path=stores.drive.manifests.untiered_passed_manifest(split).path,
            passed_archive_path=stores.drive.samples.split_archive("passed", split).path,
        )
        for split in config.splits
    )


def _build_runtime_split_layouts(
    config: TiersWorkflowConfig,
    stores: ArtifactStores,
) -> tuple[TiersRuntimeSplitLayout, ...]:
    return tuple(
        TiersRuntimeSplitLayout(
            split=split,
            passed_manifest_path=stores.runtime.manifests.untiered_passed_manifest(split).path,
            passed_samples_split_root=stores.runtime.samples.passed_split_dir(split).path,
        )
        for split in config.splits
    )


def _build_runtime_layout(
    config: TiersWorkflowConfig,
    stores: ArtifactStores,
) -> TiersRuntimeLayout:
    return TiersRuntimeLayout(
        filters_config_path=config.project_root / config.filters_config_relpath,
        tiers_config_path=config.project_root / config.tiers_config_relpath,
        passed_samples_root=stores.runtime.samples.passed_root,
        splits=_build_runtime_split_layouts(config, stores),
    )


def _build_output_layout(stores: ArtifactStores) -> TiersOutputLayout:
    return TiersOutputLayout(
        tiered_manifests_root=stores.runtime.manifests.tiered_root,
    )


def _build_report_layout(stores: ArtifactStores) -> TiersReportLayout:
    reports = stores.runtime.reports
    return TiersReportLayout(
        summary_markdown_path=reports.tiers_summary().path,
        calibration_markdown_path=reports.tiers_calibration().path,
        decision_detail_jsonl_path=reports.tiers_decision_detail().path,
        calibration_surfaces_json_path=reports.tiers_calibration_surfaces().path,
        calibration_detail_json_path=reports.tiers_calibration_detail().path,
        index_json_path=reports.tiers_index().path,
    )


def _build_publish_layout(
    config: TiersWorkflowConfig,
    stores: ArtifactStores,
) -> TiersPublishLayout:
    reports = stores.drive.reports
    return TiersPublishLayout(
        root=config.drive_project_root,
        report_targets_root=reports.tiers_root,
        summary_markdown_target_path=reports.tiers_summary().path,
        calibration_markdown_target_path=reports.tiers_calibration().path,
        decision_detail_target_path=reports.tiers_decision_detail().path,
        calibration_surfaces_target_path=reports.tiers_calibration_surfaces().path,
        calibration_detail_target_path=reports.tiers_calibration_detail().path,
        index_json_target_path=reports.tiers_index().path,
        tiered_manifest_targets_root=stores.drive.manifests.tiered_root,
    )
