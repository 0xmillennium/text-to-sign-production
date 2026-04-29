"""Dataset workflow publish plan construction and verification."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from text_to_sign_production.artifacts.layout import DatasetArtifactLayout
from text_to_sign_production.core.files import (
    ensure_dir,
    require_non_empty_file,
    verify_output_file,
)
from text_to_sign_production.core.paths import ProjectLayout
from text_to_sign_production.workflows._dataset.layout import (
    _dataset_publish_root,
    _dataset_run_root,
    _mirrored_drive_data_path,
    _split_from_publish_key,
)
from text_to_sign_production.workflows._dataset.types import (
    DatasetArchiveSpec,
    DatasetFilePublishSpec,
    DatasetPublishPlan,
    DatasetPublishSummary,
    DatasetWorkflowError,
    DatasetWorkflowResult,
)
from text_to_sign_production.workflows._shared.archives import (
    archive_member_from_path,
    build_tar_zstd_create_from_member_list_command,
    build_tar_zstd_list_command,
    validate_archive_members,
)
from text_to_sign_production.workflows._shared.transfer import build_byte_progress_copy_command
from text_to_sign_production.workflows.commands import CommandSpec


def build_dataset_publish_plan(
    result: DatasetWorkflowResult,
    *,
    drive_project_root: Path | str,
    project_root: Path | str | None = None,
) -> DatasetPublishPlan:
    """Build notebook-visible dataset publish specs without executing commands."""

    runtime_layout = ProjectLayout(Path(project_root or result.project_root))
    drive_layout = ProjectLayout(Path(drive_project_root))
    drive_artifacts = DatasetArtifactLayout(drive_layout)
    expected_run_root = _dataset_run_root(runtime_layout)
    if result.run_root != expected_run_root:
        raise DatasetWorkflowError(
            "Dataset result run root does not match the canonical Dataset Build run root: "
            f"{result.run_root}"
        )

    interim_manifest_files = _dataset_file_publish_specs(
        {
            **{f"raw_{split}": result.raw_manifest_paths[split] for split in result.splits},
            **{
                f"normalized_{split}": result.normalized_manifest_paths[split]
                for split in result.splits
            },
            **{
                f"filtered_{split}": result.filtered_manifest_paths[split]
                for split in result.splits
            },
        },
        label_prefix="interim manifest",
        runtime_layout=runtime_layout,
        drive_layout=drive_layout,
    )
    processed_manifest_files = _dataset_file_publish_specs(
        {split: result.processed_manifest_paths[split] for split in result.splits},
        label_prefix="processed manifest",
        runtime_layout=runtime_layout,
        drive_layout=drive_layout,
    )
    interim_report_files = _dataset_file_publish_specs(
        result.interim_report_paths,
        label_prefix="interim report",
        runtime_layout=runtime_layout,
        drive_layout=drive_layout,
    )
    processed_report_files = _dataset_file_publish_specs(
        {
            **result.processed_report_paths,
            **{
                f"tier_{split}_{tier}": path
                for split, tier_paths in result.tier_manifest_paths.items()
                for tier, path in tier_paths.items()
            },
        },
        label_prefix="processed report",
        runtime_layout=runtime_layout,
        drive_layout=drive_layout,
    )

    sample_archives: dict[str, DatasetArchiveSpec] = {}
    publish_root = _dataset_publish_root(runtime_layout)
    for split in result.splits:
        sample_members = _archive_members_from_paths(
            result.processed_sample_archive_members[split],
            source_root=runtime_layout.root,
        )
        sample_archives[split] = _dataset_archive_spec(
            label=f"{split} processed samples",
            archive_path=drive_artifacts.processed_sample_archive(split),
            source_root=runtime_layout.root,
            member_list_path=publish_root / f"{split}_sample_archive_members.txt",
            observed_member_list_path=publish_root / f"{split}_sample_archive_observed_members.txt",
            members=sample_members,
            split=split,
        )

    return DatasetPublishPlan(
        interim_manifest_files=interim_manifest_files,
        processed_manifest_files=processed_manifest_files,
        interim_report_files=interim_report_files,
        processed_report_files=processed_report_files,
        sample_archives=sample_archives,
    )


def write_dataset_archive_member_list(spec: DatasetArchiveSpec) -> Path:
    """Write a tar member list for a notebook-executed archive command."""

    ensure_dir(spec.member_list_path.parent, label=f"{spec.label} member list directory")
    spec.member_list_path.write_text("\n".join(spec.members) + "\n", encoding="utf-8")
    return verify_output_file(spec.member_list_path, label=f"{spec.label} member list")


def verify_dataset_publish_plan(plan: DatasetPublishPlan) -> DatasetPublishSummary:
    """Verify all Dataset publish outputs, including observed archive members."""

    file_paths: dict[str, Path] = {}
    file_sizes: dict[str, int] = {}
    sample_archive_paths: dict[str, Path] = {}
    sample_archive_sizes: dict[str, int] = {}
    sample_archive_member_counts: dict[str, int] = {}
    for file_spec in _all_dataset_file_publish_specs(plan):
        file_path = _verify_publish_file(file_spec)
        file_paths[file_spec.label] = file_path
        file_sizes[file_spec.label] = file_path.stat().st_size
    for archive_spec in plan.sample_archives.values():
        archive_path = _verify_publish_archive_file(archive_spec)
        observed_members = _verify_archive_member_listing(archive_spec)
        sample_archive_paths[archive_spec.label] = archive_path
        sample_archive_sizes[archive_spec.label] = archive_path.stat().st_size
        sample_archive_member_counts[archive_spec.label] = len(observed_members)
    return DatasetPublishSummary(
        file_paths=file_paths,
        file_sizes=file_sizes,
        sample_archive_paths=sample_archive_paths,
        sample_archive_sizes=sample_archive_sizes,
        sample_archive_member_counts=sample_archive_member_counts,
    )


def _dataset_file_publish_specs(
    paths: Mapping[str, Path],
    *,
    label_prefix: str,
    runtime_layout: ProjectLayout,
    drive_layout: ProjectLayout,
) -> dict[str, DatasetFilePublishSpec]:
    specs: dict[str, DatasetFilePublishSpec] = {}
    for key, source_path in paths.items():
        label = f"{label_prefix} {key}"
        verified_source_path = verify_output_file(source_path, label=f"Runtime {label}")
        target_path = _mirrored_drive_data_path(
            verified_source_path,
            runtime_layout=runtime_layout,
            drive_layout=drive_layout,
            label=label,
        )
        expected_input_bytes = verified_source_path.stat().st_size
        specs[key] = DatasetFilePublishSpec(
            label=label,
            split=_split_from_publish_key(key),
            source_path=verified_source_path,
            target_path=target_path,
            expected_input_bytes=expected_input_bytes,
            publish_command=_publish_file_command_spec(
                label=label,
                source_path=verified_source_path,
                target_path=target_path,
                expected_input_bytes=expected_input_bytes,
            ),
        )
    return specs


def _publish_file_command_spec(
    *,
    label: str,
    source_path: Path,
    target_path: Path,
    expected_input_bytes: int,
) -> CommandSpec:
    return build_byte_progress_copy_command(
        label=label,
        source_path=source_path,
        target_path=target_path,
        expected_input_bytes=expected_input_bytes,
        failure=f"Failed to publish {label}: {source_path} -> {target_path}",
    )


def _all_dataset_file_publish_specs(
    plan: DatasetPublishPlan,
) -> tuple[DatasetFilePublishSpec, ...]:
    return (
        *plan.interim_manifest_files.values(),
        *plan.processed_manifest_files.values(),
        *plan.interim_report_files.values(),
        *plan.processed_report_files.values(),
    )


def _dataset_archive_spec(
    *,
    label: str,
    archive_path: Path,
    source_root: Path,
    member_list_path: Path,
    observed_member_list_path: Path,
    members: tuple[str, ...],
    split: str | None,
) -> DatasetArchiveSpec:
    validated_members = _validate_dataset_archive_members(members, label=label)
    archive_parent = archive_path.parent
    expected_member_count = len(validated_members)
    return DatasetArchiveSpec(
        label=label,
        archive_path=archive_path,
        archive_parent=archive_parent,
        source_root=source_root,
        member_list_path=member_list_path,
        observed_member_list_path=observed_member_list_path,
        members=validated_members,
        expected_member_count=expected_member_count,
        split=split,
        create_command=build_tar_zstd_create_from_member_list_command(
            label=label,
            archive_path=archive_path,
            archive_parent=archive_parent,
            source_root=source_root,
            member_list_path=member_list_path,
            expected_member_count=expected_member_count,
            failure=f"Failed to create {label} archive: {archive_path}",
        ),
        verify_command=build_tar_zstd_list_command(
            label=label,
            archive_path=archive_path,
            observed_member_list_path=observed_member_list_path,
            expected_member_count=expected_member_count,
            failure=f"Failed to verify {label} archive members: {archive_path}",
        ),
    )


def _archive_members_from_paths(
    paths: Iterable[Path],
    *,
    source_root: Path,
) -> tuple[str, ...]:
    return tuple(archive_member_from_path(path, source_root=source_root) for path in paths)


def _validate_dataset_archive_members(members: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    return validate_archive_members(
        members,
        label=label,
        required_prefix="data/",
        error_factory=DatasetWorkflowError,
    )


def _verify_publish_file(spec: DatasetFilePublishSpec) -> Path:
    target_path = verify_output_file(spec.target_path, label=spec.label)
    observed_size = target_path.stat().st_size
    if observed_size != spec.expected_input_bytes:
        raise DatasetWorkflowError(
            f"{spec.label} byte count mismatch: expected "
            f"{spec.expected_input_bytes}, observed {observed_size}: {target_path}"
        )
    return target_path


def _verify_publish_archive_file(spec: DatasetArchiveSpec) -> Path:
    return verify_output_file(spec.archive_path, label=f"{spec.label} archive")


def _verify_archive_member_listing(spec: DatasetArchiveSpec) -> tuple[str, ...]:
    observed_path = require_non_empty_file(
        spec.observed_member_list_path,
        label=f"{spec.label} observed archive member list",
    )
    observed_members = tuple(
        line.strip()
        for line in observed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if observed_members != spec.members:
        raise DatasetWorkflowError(
            f"{spec.label} archive members do not match the workflow spec. "
            f"Expected {spec.members}, observed {observed_members}."
        )
    return observed_members


__all__ = ["build_dataset_publish_plan", "write_dataset_archive_member_list"]
