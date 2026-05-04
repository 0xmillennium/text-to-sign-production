"""Publish planning for samples workflow outputs."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.artifacts.store import SampleStatus, build_artifact_stores
from text_to_sign_production.foundation.roots import build_repo_roots
from text_to_sign_production.workflows._shared.archives import (
    archive_member_from_path,
    build_archive_member_list_spec,
    build_tar_zstd_create_spec,
    build_tar_zstd_verify_spec,
    write_archive_member_list,
)
from text_to_sign_production.workflows._shared.transfers import build_byte_progress_copy_spec
from text_to_sign_production.workflows.samples.types import (
    SamplesPublishPlan,
    SamplesWorkflowConfig,
    SamplesWorkflowResult,
)


def build_samples_publish_plan(
    config: SamplesWorkflowConfig,
    result: SamplesWorkflowResult,
) -> SamplesPublishPlan:
    stores = build_artifact_stores(
        build_repo_roots(config.project_root),
        build_repo_roots(config.drive_project_root),
    )

    file_transfers = []
    for runtime_path in result.output_summary.untiered_manifest_paths:
        file_transfers.append(
            build_byte_progress_copy_spec(
                label=f"publish_samples_manifest_{runtime_path.stem}_{runtime_path.parent.name}",
                source_path=runtime_path,
                target_path=(
                    stores.drive.repo_root / runtime_path.relative_to(stores.runtime.repo_root)
                ),
            )
        )
    for runtime_path in result.output_summary.report_paths:
        file_transfers.append(
            build_byte_progress_copy_spec(
                label=f"publish_samples_report_{runtime_path.stem}",
                source_path=runtime_path,
                target_path=(
                    stores.drive.repo_root / runtime_path.relative_to(stores.runtime.repo_root)
                ),
            )
        )

    archive_creates = []
    archive_member_lists = []
    archive_verifies = []
    for status in (SampleStatus.PASSED, SampleStatus.DROPPED):
        source_root = stores.runtime.samples.split_extract_root(status).path
        for split in config.splits:
            split_dir = stores.runtime.samples.sample_dir(status, split).path
            members = tuple(
                archive_member_from_path(path, source_root)
                for path in sorted(split_dir.glob("*.npz"), key=lambda item: item.name)
            )
            archive_path = stores.drive.samples.split_archive(status, split).path
            member_list_path = _member_list_path(
                stores.runtime.repo_root,
                status.value,
                split.value,
            )
            create_spec = build_tar_zstd_create_spec(
                label=f"publish_samples_{status.value}_{split.value}",
                archive_path=archive_path,
                source_root=source_root,
                member_list_path=member_list_path,
                members=members,
            )
            archive_member_lists.append(
                build_archive_member_list_spec(
                    label=f"write_samples_archive_members_{status.value}_{split.value}",
                    member_list_path=member_list_path,
                    members=members,
                )
            )
            archive_creates.append(create_spec)
            archive_verifies.append(
                build_tar_zstd_verify_spec(
                    label=f"verify_samples_{status.value}_{split.value}",
                    archive_path=archive_path,
                    expected_member_list_path=member_list_path,
                    observed_member_list_path=_observed_member_list_path(
                        stores.runtime.repo_root,
                        status.value,
                        split.value,
                    ),
                )
            )

    return SamplesPublishPlan(
        file_transfers=tuple(file_transfers),
        archive_member_lists=tuple(archive_member_lists),
        archive_creates=tuple(archive_creates),
        archive_verifies=tuple(archive_verifies),
    )


def materialize_samples_publish_metadata(plan: SamplesPublishPlan) -> None:
    for spec in plan.archive_member_lists:
        write_archive_member_list(spec)


def _member_list_path(project_root: Path, status: str, split: str) -> Path:
    return (
        project_root
        / ".workflow"
        / "samples"
        / "archive_members"
        / status
        / f"{split}.txt"
    )


def _observed_member_list_path(project_root: Path, status: str, split: str) -> Path:
    return (
        project_root
        / ".workflow"
        / "samples"
        / "archive_members"
        / status
        / f"{split}.observed.txt"
    )


__all__ = ["build_samples_publish_plan", "materialize_samples_publish_metadata"]
