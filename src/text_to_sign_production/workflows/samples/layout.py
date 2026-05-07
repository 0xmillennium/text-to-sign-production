from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.artifacts.store import (
    ArtifactTopology,
    ReportsTopology,
    build_artifact_topology,
)
from text_to_sign_production.core import build_repo_roots
from text_to_sign_production.workflows.samples.constants import (
    SAMPLES_DROPPED_PARTITION,
    SAMPLES_PASSED_PARTITION,
    SAMPLES_RUNTIME_DIRNAME,
    SAMPLES_WORKFLOW_DIRNAME,
)
from text_to_sign_production.workflows.samples.contracts.config import (
    SamplesWorkflowConfig,
)


@dataclass(frozen=True, slots=True)
class SamplesDriveSplitLayout:
    split: str
    translation_csv_path: Path
    keypoint_archive_path: Path


@dataclass(frozen=True, slots=True)
class SamplesRuntimeSplitLayout:
    split: str
    translation_csv_path: Path
    keypoint_root: Path
    keypoint_json_root: Path
    keypoint_video_root: Path


@dataclass(frozen=True, slots=True)
class SamplesRuntimeLayout:
    root: Path
    gates_config_path: Path
    keypoint_extract_root: Path
    splits: tuple[SamplesRuntimeSplitLayout, ...]


@dataclass(frozen=True, slots=True)
class SamplesManifestLayout:
    partition: str
    split: str
    path: Path


@dataclass(frozen=True, slots=True)
class SamplesOutputLayout:
    root: Path
    manifest_outputs: tuple[SamplesManifestLayout, ...]
    passed_samples_root: Path
    dropped_samples_root: Path


@dataclass(frozen=True, slots=True)
class SamplesReportLayout:
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
class SamplesPublishSplitLayout:
    split: str
    passed_manifest_target_path: Path
    dropped_manifest_target_path: Path
    passed_archive_path: Path
    dropped_archive_path: Path


@dataclass(frozen=True, slots=True)
class SamplesPublishLayout:
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
    splits: tuple[SamplesPublishSplitLayout, ...]


@dataclass(frozen=True, slots=True)
class SamplesLayout:
    config: SamplesWorkflowConfig
    drive_splits: tuple[SamplesDriveSplitLayout, ...]
    runtime: SamplesRuntimeLayout
    outputs: SamplesOutputLayout
    reports: SamplesReportLayout
    publish: SamplesPublishLayout


def build_samples_layout(config: SamplesWorkflowConfig) -> SamplesLayout:
    runtime_root = _runtime_root(config)
    return SamplesLayout(
        config=config,
        drive_splits=_build_drive_split_layouts(config),
        runtime=_build_runtime_layout(config, runtime_root),
        outputs=_build_output_layout(config, runtime_root),
        reports=_build_report_layout(runtime_root),
        publish=_build_publish_layout(config),
    )


def _build_drive_split_layouts(
    config: SamplesWorkflowConfig,
) -> tuple[SamplesDriveSplitLayout, ...]:
    drive_topology = build_artifact_topology(build_repo_roots(config.drive_project_root))
    return tuple(
        SamplesDriveSplitLayout(
            split=split,
            translation_csv_path=drive_topology.assets.translation_csv(split).path,
            keypoint_archive_path=drive_topology.assets.keypoint_archive(split).path,
        )
        for split in config.splits
    )


def _build_runtime_split_layouts(
    config: SamplesWorkflowConfig,
    runtime_root: Path,
) -> tuple[SamplesRuntimeSplitLayout, ...]:
    runtime_topology = build_artifact_topology(build_repo_roots(runtime_root))
    return tuple(
        SamplesRuntimeSplitLayout(
            split=split,
            translation_csv_path=runtime_topology.assets.translation_csv(split).path,
            keypoint_root=runtime_topology.assets.keypoint_split_root(split).path,
            keypoint_json_root=runtime_topology.assets.keypoint_json_dir(split).path,
            keypoint_video_root=runtime_topology.assets.keypoint_video_dir(split).path,
        )
        for split in config.splits
    )


def _build_runtime_layout(
    config: SamplesWorkflowConfig,
    runtime_root: Path,
) -> SamplesRuntimeLayout:
    runtime_topology = build_artifact_topology(build_repo_roots(runtime_root))
    return SamplesRuntimeLayout(
        root=runtime_root,
        gates_config_path=config.project_root / config.gates_config_relpath,
        keypoint_extract_root=runtime_topology.assets.keypoint_extract_root().path,
        splits=_build_runtime_split_layouts(config, runtime_root),
    )


def _build_output_layout(
    config: SamplesWorkflowConfig,
    runtime_root: Path,
) -> SamplesOutputLayout:
    runtime_topology = build_artifact_topology(build_repo_roots(runtime_root))
    return SamplesOutputLayout(
        root=runtime_topology.samples_root,
        manifest_outputs=_build_manifest_layouts(config, runtime_topology),
        passed_samples_root=runtime_topology.samples.passed_root,
        dropped_samples_root=runtime_topology.samples.dropped_root,
    )


def _build_manifest_layouts(
    config: SamplesWorkflowConfig,
    topology: ArtifactTopology,
) -> tuple[SamplesManifestLayout, ...]:
    manifest_outputs: list[SamplesManifestLayout] = []
    for split in config.splits:
        manifest_outputs.append(
            SamplesManifestLayout(
                partition=SAMPLES_PASSED_PARTITION,
                split=split,
                path=topology.manifests.untiered_passed_manifest(split).path,
            )
        )
        manifest_outputs.append(
            SamplesManifestLayout(
                partition=SAMPLES_DROPPED_PARTITION,
                split=split,
                path=topology.manifests.untiered_dropped_manifest(split).path,
            )
        )
    return tuple(manifest_outputs)


def _build_report_layout(runtime_root: Path) -> SamplesReportLayout:
    reports = build_artifact_topology(build_repo_roots(runtime_root)).reports
    return _report_layout_from_topology(reports)


def _report_layout_from_topology(reports: ReportsTopology) -> SamplesReportLayout:
    return SamplesReportLayout(
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


def _build_publish_layout(config: SamplesWorkflowConfig) -> SamplesPublishLayout:
    drive_topology = build_artifact_topology(build_repo_roots(config.drive_project_root))
    reports = drive_topology.reports
    manifest_targets_root = drive_topology.manifests.untiered_root
    samples_targets_root = drive_topology.samples_root
    return SamplesPublishLayout(
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
            SamplesPublishSplitLayout(
                split=split,
                passed_manifest_target_path=(
                    drive_topology.manifests.untiered_passed_manifest(split).path
                ),
                dropped_manifest_target_path=(
                    drive_topology.manifests.untiered_dropped_manifest(split).path
                ),
                passed_archive_path=(
                    drive_topology.samples.split_archive(SAMPLES_PASSED_PARTITION, split).path
                ),
                dropped_archive_path=(
                    drive_topology.samples.split_archive(SAMPLES_DROPPED_PARTITION, split).path
                ),
            )
            for split in config.splits
        ),
    )


def _runtime_root(config: SamplesWorkflowConfig) -> Path:
    return config.project_root / SAMPLES_RUNTIME_DIRNAME / SAMPLES_WORKFLOW_DIRNAME
