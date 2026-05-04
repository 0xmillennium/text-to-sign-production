"""Runtime restore planning for the samples workflow."""

from __future__ import annotations

from text_to_sign_production.artifacts.store import build_artifact_stores
from text_to_sign_production.foundation.roots import build_repo_roots
from text_to_sign_production.workflows._shared.archives import build_tar_zstd_extract_spec
from text_to_sign_production.workflows._shared.transfers import build_byte_progress_copy_spec
from text_to_sign_production.workflows.samples.types import (
    SamplesRuntimePlan,
    SamplesWorkflowConfig,
)


def build_samples_runtime_plan(config: SamplesWorkflowConfig) -> SamplesRuntimePlan:
    runtime_roots = build_repo_roots(config.project_root)
    drive_roots = build_repo_roots(config.drive_project_root)
    stores = build_artifact_stores(runtime_roots, drive_roots)

    file_transfers = []
    archive_extracts = []
    for split in config.splits:
        file_transfers.append(
            build_byte_progress_copy_spec(
                label=f"restore_samples_translation_{split.value}",
                source_path=stores.drive.assets.translation_csv(split).path,
                target_path=stores.runtime.assets.translation_csv(split).path,
            )
        )
        archive_extracts.append(
            build_tar_zstd_extract_spec(
                label=f"restore_samples_keypoints_{split.value}",
                archive_path=stores.drive.assets.keypoint_archive(split).path,
                extraction_root=stores.runtime.assets.how2sign_bfh_keypoints_root,
                members=(),
            )
        )

    return SamplesRuntimePlan(
        project_root=config.project_root,
        drive_project_root=config.drive_project_root,
        restore_file_transfers=tuple(file_transfers),
        restore_archive_extracts=tuple(archive_extracts),
    )


__all__ = ["build_samples_runtime_plan"]
