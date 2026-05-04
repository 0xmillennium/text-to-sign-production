"""Runtime restore planning for the tiers workflow."""

from __future__ import annotations

from text_to_sign_production.artifacts.store import SampleStatus, build_artifact_stores
from text_to_sign_production.foundation.roots import build_repo_roots
from text_to_sign_production.workflows._shared.archives import build_tar_zstd_extract_spec
from text_to_sign_production.workflows._shared.transfers import build_byte_progress_copy_spec
from text_to_sign_production.workflows.tiers.types import (
    TiersRuntimePlan,
    TiersWorkflowConfig,
)


def build_tiers_runtime_plan(config: TiersWorkflowConfig) -> TiersRuntimePlan:
    runtime_roots = build_repo_roots(config.project_root)
    drive_roots = build_repo_roots(config.drive_project_root)
    stores = build_artifact_stores(runtime_roots, drive_roots)

    file_transfers = []
    archive_extracts = []
    for split in config.splits:
        file_transfers.append(
            build_byte_progress_copy_spec(
                label=f"restore_tiers_untiered_passed_manifest_{split.value}",
                source_path=stores.drive.manifests.untiered_passed_manifest(split).path,
                target_path=stores.runtime.manifests.untiered_passed_manifest(split).path,
            )
        )
        archive_extracts.append(
            build_tar_zstd_extract_spec(
                label=f"restore_tiers_passed_samples_{split.value}",
                archive_path=stores.drive.samples.split_archive(SampleStatus.PASSED, split).path,
                extraction_root=stores.runtime.samples.split_extract_root(
                    SampleStatus.PASSED
                ).path,
                members=(),
            )
        )

    return TiersRuntimePlan(
        project_root=config.project_root,
        drive_project_root=config.drive_project_root,
        restore_file_transfers=tuple(file_transfers),
        restore_archive_extracts=tuple(archive_extracts),
    )


__all__ = ["build_tiers_runtime_plan"]
