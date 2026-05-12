from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.artifacts.store import (
    ArtifactTopology,
    ReportsTopology,
    build_artifact_topology,
)
from text_to_sign_production.core import build_repo_roots
from text_to_sign_production.workflows.gate.constants import (
    GATE_DROPPED_PARTITION,
    GATE_PASSED_PARTITION,
    GATE_RUNTIME_DIRNAME,
    GATE_WORKFLOW_DIRNAME,
)
from text_to_sign_production.workflows.gate.contracts.config import (
    GateWorkflowConfig,
)


@dataclass(frozen=True, slots=True)
class GateDriveSplitLayout:
    split: str
    translation_csv_path: Path
    keypoint_archive_path: Path


@dataclass(frozen=True, slots=True)
class GateRuntimeSplitLayout:
    split: str
    translation_csv_path: Path
    keypoint_root: Path
    keypoint_json_root: Path
    keypoint_video_root: Path


@dataclass(frozen=True, slots=True)
class GateRuntimeLayout:
    root: Path
    gates_config_original_path: Path
    gates_config_path: Path
    keypoint_extract_root: Path
    splits: tuple[GateRuntimeSplitLayout, ...]


@dataclass(frozen=True, slots=True)
class GateManifestLayout:
    partition: str
    split: str
    path: Path


@dataclass(frozen=True, slots=True)
class GateOutputLayout:
    root: Path
    manifest_outputs: tuple[GateManifestLayout, ...]
    passed_samples_root: Path
    dropped_samples_root: Path


@dataclass(frozen=True, slots=True)
class GateReportLayout:
    root: Path
    summary_markdown_path: Path
    processing_summary_jsonl_path: Path
    processing_detail_jsonl_path: Path
    gate_summary_jsonl_path: Path
    gate_detail_jsonl_path: Path
    source_issue_summary_jsonl_path: Path
    source_issue_detail_jsonl_path: Path
    index_json_path: Path


@dataclass(frozen=True, slots=True)
class GatePublishSplitLayout:
    split: str
    passed_manifest_target_path: Path
    dropped_manifest_target_path: Path
    passed_archive_path: Path
    dropped_archive_path: Path


@dataclass(frozen=True, slots=True)
class GatePublishLayout:
    root: Path
    report_targets_root: Path
    summary_markdown_target_path: Path
    processing_summary_jsonl_target_path: Path
    processing_detail_jsonl_target_path: Path
    gate_summary_jsonl_target_path: Path
    gate_detail_jsonl_target_path: Path
    source_issue_summary_jsonl_target_path: Path
    source_issue_detail_jsonl_target_path: Path
    index_json_target_path: Path
    manifest_targets_root: Path
    samples_targets_root: Path
    splits: tuple[GatePublishSplitLayout, ...]


@dataclass(frozen=True, slots=True)
class GateLayout:
    config: GateWorkflowConfig
    drive_splits: tuple[GateDriveSplitLayout, ...]
    runtime: GateRuntimeLayout
    outputs: GateOutputLayout
    reports: GateReportLayout
    publish: GatePublishLayout


def build_gate_layout(config: GateWorkflowConfig) -> GateLayout:
    runtime_root = _runtime_root(config)
    return GateLayout(
        config=config,
        drive_splits=_build_drive_split_layouts(config),
        runtime=_build_runtime_layout(config, runtime_root),
        outputs=_build_output_layout(config, runtime_root),
        reports=_build_report_layout(runtime_root),
        publish=_build_publish_layout(config),
    )


def _build_drive_split_layouts(
    config: GateWorkflowConfig,
) -> tuple[GateDriveSplitLayout, ...]:
    drive_topology = build_artifact_topology(build_repo_roots(config.drive_project_root))
    return tuple(
        GateDriveSplitLayout(
            split=split,
            translation_csv_path=drive_topology.assets.translation_csv(split).path,
            keypoint_archive_path=drive_topology.assets.keypoint_archive(split).path,
        )
        for split in config.splits
    )


def _build_runtime_split_layouts(
    config: GateWorkflowConfig,
    runtime_root: Path,
) -> tuple[GateRuntimeSplitLayout, ...]:
    runtime_topology = build_artifact_topology(build_repo_roots(runtime_root))
    return tuple(
        GateRuntimeSplitLayout(
            split=split,
            translation_csv_path=runtime_topology.assets.translation_csv(split).path,
            keypoint_root=runtime_topology.assets.keypoint_split_root(split).path,
            keypoint_json_root=runtime_topology.assets.keypoint_json_dir(split).path,
            keypoint_video_root=runtime_topology.assets.keypoint_video_dir(split).path,
        )
        for split in config.splits
    )


def _build_runtime_layout(
    config: GateWorkflowConfig,
    runtime_root: Path,
) -> GateRuntimeLayout:
    runtime_topology = build_artifact_topology(build_repo_roots(runtime_root))
    return GateRuntimeLayout(
        root=runtime_root,
        gates_config_original_path=config.project_root / config.gates_config_relpath,
        gates_config_path=runtime_root / "provenance" / "config" / "gates.yaml",
        keypoint_extract_root=runtime_topology.assets.keypoint_extract_root().path,
        splits=_build_runtime_split_layouts(config, runtime_root),
    )


def _build_output_layout(
    config: GateWorkflowConfig,
    runtime_root: Path,
) -> GateOutputLayout:
    runtime_topology = build_artifact_topology(build_repo_roots(runtime_root))
    return GateOutputLayout(
        root=runtime_topology.samples_root,
        manifest_outputs=_build_manifest_layouts(config, runtime_topology),
        passed_samples_root=runtime_topology.samples.passed_root,
        dropped_samples_root=runtime_topology.samples.dropped_root,
    )


def _build_manifest_layouts(
    config: GateWorkflowConfig,
    topology: ArtifactTopology,
) -> tuple[GateManifestLayout, ...]:
    manifest_outputs: list[GateManifestLayout] = []
    for split in config.splits:
        manifest_outputs.append(
            GateManifestLayout(
                partition=GATE_PASSED_PARTITION,
                split=split,
                path=topology.manifests.untiered_passed_manifest(split).path,
            )
        )
        manifest_outputs.append(
            GateManifestLayout(
                partition=GATE_DROPPED_PARTITION,
                split=split,
                path=topology.manifests.untiered_dropped_manifest(split).path,
            )
        )
    return tuple(manifest_outputs)


def _build_report_layout(runtime_root: Path) -> GateReportLayout:
    reports = build_artifact_topology(build_repo_roots(runtime_root)).reports
    return _report_layout_from_topology(reports)


def _report_layout_from_topology(reports: ReportsTopology) -> GateReportLayout:
    return GateReportLayout(
        root=reports.samples_root,
        summary_markdown_path=reports.samples_summary().path,
        processing_summary_jsonl_path=reports.samples_processing_summary().path,
        processing_detail_jsonl_path=reports.samples_processing_detail().path,
        gate_summary_jsonl_path=reports.samples_gate_summary().path,
        gate_detail_jsonl_path=reports.samples_gate_detail().path,
        source_issue_summary_jsonl_path=reports.samples_source_issue_summary().path,
        source_issue_detail_jsonl_path=reports.samples_source_issue_detail().path,
        index_json_path=reports.samples_index().path,
    )


def _build_publish_layout(config: GateWorkflowConfig) -> GatePublishLayout:
    drive_topology = build_artifact_topology(build_repo_roots(config.drive_project_root))
    reports = drive_topology.reports
    manifest_targets_root = drive_topology.manifests.untiered_root
    samples_targets_root = drive_topology.samples_root
    return GatePublishLayout(
        root=config.drive_project_root,
        report_targets_root=reports.samples_root,
        summary_markdown_target_path=reports.samples_summary().path,
        processing_summary_jsonl_target_path=reports.samples_processing_summary().path,
        processing_detail_jsonl_target_path=reports.samples_processing_detail().path,
        gate_summary_jsonl_target_path=reports.samples_gate_summary().path,
        gate_detail_jsonl_target_path=reports.samples_gate_detail().path,
        source_issue_summary_jsonl_target_path=reports.samples_source_issue_summary().path,
        source_issue_detail_jsonl_target_path=reports.samples_source_issue_detail().path,
        index_json_target_path=reports.samples_index().path,
        manifest_targets_root=manifest_targets_root,
        samples_targets_root=samples_targets_root,
        splits=tuple(
            GatePublishSplitLayout(
                split=split,
                passed_manifest_target_path=(
                    drive_topology.manifests.untiered_passed_manifest(split).path
                ),
                dropped_manifest_target_path=(
                    drive_topology.manifests.untiered_dropped_manifest(split).path
                ),
                passed_archive_path=(
                    drive_topology.samples.split_archive(GATE_PASSED_PARTITION, split).path
                ),
                dropped_archive_path=(
                    drive_topology.samples.split_archive(GATE_DROPPED_PARTITION, split).path
                ),
            )
            for split in config.splits
        ),
    )


def _runtime_root(config: GateWorkflowConfig) -> Path:
    return config.project_root / GATE_RUNTIME_DIRNAME / GATE_WORKFLOW_DIRNAME
