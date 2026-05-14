from __future__ import annotations

from pathlib import Path

from text_to_sign_production.artifacts.store import (
    ArtifactTopology,
    build_artifact_topology,
    resolve_samples_relative,
)
from text_to_sign_production.core import build_repo_roots
from dataclasses import dataclass

from text_to_sign_production.core.ids import SampleSplit, TierMembership, TierName
from text_to_sign_production.workflows.debug.contracts import DebugWorkflowConfig


def runtime_topology(config: DebugWorkflowConfig) -> ArtifactTopology:
    return build_artifact_topology(build_repo_roots(config.runtime_root))


def drive_topology(config: DebugWorkflowConfig) -> ArtifactTopology:
    return build_artifact_topology(build_repo_roots(config.drive_project_root))


@dataclass(frozen=True, slots=True)
class DebugLayout:
    config: DebugWorkflowConfig
    runtime: ArtifactTopology
    drive: ArtifactTopology


def build_debug_layout(config: DebugWorkflowConfig) -> DebugLayout:
    return DebugLayout(
        config=config,
        runtime=runtime_topology(config),
        drive=drive_topology(config),
    )


def project_config_path(config: DebugWorkflowConfig, name: str) -> Path:
    relpaths = {
        "gates.yaml": config.gates_config_relpath,
        "filters.yaml": config.filters_config_relpath,
        "tiers.yaml": config.tiers_config_relpath,
    }
    return config.project_root / relpaths[name]


def runtime_translation_path(layout: DebugLayout, split: SampleSplit) -> Path:
    return layout.runtime.assets.translation_csv(split).path


def runtime_keypoint_json_dir(
    layout: DebugLayout,
    split: SampleSplit,
    sentence_name: str,
) -> Path:
    return layout.runtime.assets.keypoint_json_dir(split).path / sentence_name


def runtime_keypoint_video_path(
    layout: DebugLayout,
    split: SampleSplit,
    sentence_name: str,
) -> Path:
    return layout.runtime.assets.keypoint_video_dir(split).path / f"{sentence_name}.mp4"


def runtime_passed_manifest_path(layout: DebugLayout, split: SampleSplit) -> Path:
    return layout.runtime.manifests.untiered_passed_manifest(split).path


def runtime_dropped_manifest_path(layout: DebugLayout, split: SampleSplit) -> Path:
    return layout.runtime.manifests.untiered_dropped_manifest(split).path


def runtime_tier_manifest_path(
    layout: DebugLayout,
    *,
    tier: TierName,
    membership: TierMembership,
    split: SampleSplit,
) -> Path:
    return layout.runtime.manifests.tiered_manifest(tier, membership, split).path


def runtime_sample_payload_path(layout: DebugLayout, payload_ref: str | Path) -> Path:
    return resolve_samples_relative(layout.runtime, payload_ref).path


def runtime_debug_samples_root(layout: DebugLayout) -> Path:
    return layout.config.runtime_root / layout.config.debug_reports_relroot


def drive_debug_samples_root(layout: DebugLayout) -> Path:
    return layout.config.drive_project_root / layout.config.debug_reports_relroot


def runtime_debug_sample_run_root(
    layout: DebugLayout,
    *,
    sentence_name: str,
    run_id: str,
) -> Path:
    return runtime_debug_samples_root(layout) / sentence_name / run_id


def drive_debug_sample_run_root(
    layout: DebugLayout,
    *,
    sentence_name: str,
    run_id: str,
) -> Path:
    return drive_debug_samples_root(layout) / sentence_name / run_id


def debug_sample_run_root(
    layout: DebugLayout,
    *,
    sentence_name: str,
    run_id: str,
) -> Path:
    return runtime_debug_sample_run_root(
        layout,
        sentence_name=sentence_name,
        run_id=run_id,
    )


__all__ = [
    "debug_sample_run_root",
    "DebugLayout",
    "build_debug_layout",
    "drive_topology",
    "drive_debug_sample_run_root",
    "drive_debug_samples_root",
    "project_config_path",
    "runtime_debug_sample_run_root",
    "runtime_debug_samples_root",
    "runtime_dropped_manifest_path",
    "runtime_keypoint_json_dir",
    "runtime_keypoint_video_path",
    "runtime_passed_manifest_path",
    "runtime_sample_payload_path",
    "runtime_tier_manifest_path",
    "runtime_topology",
    "runtime_translation_path",
]
