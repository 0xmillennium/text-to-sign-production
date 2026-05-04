"""Catalog-level invariant validation."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.artifacts.catalog.types import (
    SampleRef,
    SamplesCatalog,
    TieredCatalog,
)
from text_to_sign_production.artifacts.store.resolve import resolve_samples_relative
from text_to_sign_production.artifacts.store.topology import (
    ArtifactStores,
    sample_manifest_relative_path,
)
from text_to_sign_production.artifacts.store.types import (
    ArchiveMemberPathRef,
    ArchivePathRef,
    SamplePathRef,
    SampleStatus,
    SplitName,
)
from text_to_sign_production.artifacts.store.validate import (
    validate_sample_archive_member_path,
    validate_sample_archive_relative_path,
    validate_samples_relative_path,
)


def validate_samples_catalog(catalog: SamplesCatalog, stores: ArtifactStores) -> list[str]:
    """Validate logical invariants for a samples catalog."""

    errors: list[str] = []
    for ref, handle in catalog.items.items():
        label = _format_ref(ref)
        if handle.ref != ref:
            errors.append(f"{label}: handle ref must match the catalog key.")
        if handle.status is not catalog.status:
            errors.append(f"{label}: handle status must match catalog status.")
        if handle.manifest_entry.sample_id != ref.sample_id:
            errors.append(f"{label}: manifest sample_id must match ref sample_id.")
        if handle.manifest_entry.split != ref.split.value:
            errors.append(f"{label}: manifest split must match ref split.")
        declared_sample_path = handle.manifest_entry.sample_path
        if catalog.status is SampleStatus.PASSED and declared_sample_path is None:
            errors.append(f"{label}: passed manifest entry must declare a sample path.")
        if declared_sample_path is None:
            if handle.runtime_sample is not None:
                errors.append(
                    f"{label}: runtime_sample must be absent when manifest omits it."
                )
            if handle.drive_archive is not None:
                errors.append(
                    f"{label}: drive_archive must be absent when manifest omits it."
                )
            if handle.drive_archive_member is not None:
                errors.append(
                    f"{label}: drive_archive_member must be absent when manifest omits it."
                )
        else:
            manifest_path_errors = _validate_manifest_sample_path(
                declared_sample_path,
                label=label,
                required_status=catalog.status,
                expected_split=ref.split,
                expected_sample_id=ref.sample_id,
            )
            errors.extend(manifest_path_errors)
            errors.extend(
                _validate_runtime_sample_binding(
                    stores=stores,
                    sample=handle.runtime_sample,
                    label=label,
                    declared_sample_path=declared_sample_path,
                    manifest_path_is_valid=not manifest_path_errors,
                )
            )
            errors.extend(
                _validate_drive_sample_binding(
                    stores=stores,
                    archive=handle.drive_archive,
                    member=handle.drive_archive_member,
                    label=label,
                    status=catalog.status,
                    split=ref.split,
                    sample_id=ref.sample_id,
                )
            )
    return errors


def validate_tiered_catalog(catalog: TieredCatalog, stores: ArtifactStores) -> list[str]:
    """Validate logical invariants for a tiered catalog."""

    errors: list[str] = []
    for ref, handle in catalog.items.items():
        label = _format_ref(ref)
        if handle.ref != ref:
            errors.append(f"{label}: handle ref must match the catalog key.")
        if handle.tier is not catalog.tier:
            errors.append(f"{label}: handle tier must match catalog tier.")
        if handle.membership is not catalog.membership:
            errors.append(f"{label}: handle membership must match catalog membership.")
        if handle.manifest_entry.sample_id != ref.sample_id:
            errors.append(f"{label}: manifest sample_id must match ref sample_id.")
        if handle.manifest_entry.split != ref.split.value:
            errors.append(f"{label}: manifest split must match ref split.")
        declared_sample_path = handle.manifest_entry.sample_path
        manifest_path_errors = _validate_manifest_sample_path(
            declared_sample_path,
            label=label,
            required_status=SampleStatus.PASSED,
            expected_split=ref.split,
            expected_sample_id=ref.sample_id,
        )
        errors.extend(manifest_path_errors)
        errors.extend(
            _validate_runtime_sample_binding(
                stores=stores,
                sample=handle.runtime_sample,
                label=label,
                declared_sample_path=declared_sample_path,
                manifest_path_is_valid=not manifest_path_errors,
            )
        )
        errors.extend(
            _validate_drive_sample_binding(
                stores=stores,
                archive=handle.drive_archive,
                member=handle.drive_archive_member,
                label=label,
                status=SampleStatus.PASSED,
                split=ref.split,
                sample_id=ref.sample_id,
            )
        )
    return errors


def _validate_runtime_sample_binding(
    *,
    stores: ArtifactStores,
    sample: object | None,
    label: str,
    declared_sample_path: str,
    manifest_path_is_valid: bool,
) -> list[str]:
    errors: list[str] = []
    if sample is None:
        errors.append(f"{label}: runtime_sample must resolve the manifest-declared path.")
        return errors
    if not isinstance(sample, SamplePathRef):
        errors.append(f"{label}: runtime_sample must be a SamplePathRef.")
        return errors

    sample_relative_path = _root_relative_path(sample.path, stores.runtime.samples_root)
    if sample_relative_path is None:
        errors.append(f"{label}: runtime_sample must be under the runtime samples root.")
    else:
        errors.extend(
            f"{label}: {error}" for error in validate_samples_relative_path(sample_relative_path)
        )

    if manifest_path_is_valid:
        expected_sample = resolve_samples_relative(stores.runtime, declared_sample_path)
        if sample != expected_sample:
            errors.append(f"{label}: runtime_sample must match manifest sample_path.")
    return errors


def _validate_manifest_sample_path(
    sample_path: str | None,
    *,
    label: str,
    required_status: SampleStatus,
    expected_split: SplitName,
    expected_sample_id: str,
) -> list[str]:
    if sample_path is None:
        return [f"{label}: manifest entry must declare a sample path."]

    errors = [f"{label}: {error}" for error in validate_samples_relative_path(sample_path)]
    if errors:
        return errors

    expected_path = sample_manifest_relative_path(
        required_status,
        expected_split,
        expected_sample_id,
    )
    if Path(sample_path) != expected_path:
        errors.append(
            f"{label}: manifest sample_path must be {expected_path.as_posix()}."
        )
    return errors


def _validate_drive_sample_binding(
    *,
    stores: ArtifactStores,
    archive: object | None,
    member: object | None,
    label: str,
    status: SampleStatus,
    split: SplitName,
    sample_id: str,
) -> list[str]:
    errors: list[str] = []
    expected_archive = stores.drive.samples.split_archive(status, split)
    if archive is None:
        errors.append(f"{label}: drive_archive must be present.")
    elif not isinstance(archive, ArchivePathRef):
        errors.append(f"{label}: drive_archive must be an ArchivePathRef.")
    else:
        archive_relative_path = _root_relative_path(archive.path, stores.drive.repo_root)
        if archive_relative_path is None:
            errors.append(f"{label}: drive_archive must be under the Drive repo root.")
        else:
            errors.extend(
                f"{label}: {error}"
                for error in validate_sample_archive_relative_path(archive_relative_path)
            )
        if archive != expected_archive:
            errors.append(f"{label}: drive_archive must point to {expected_archive.path}.")

    expected_member = stores.drive.samples.archive_member(split, sample_id)
    if member is None:
        errors.append(f"{label}: drive_archive_member must be present.")
    elif not isinstance(member, ArchiveMemberPathRef):
        errors.append(f"{label}: drive_archive_member must be an ArchiveMemberPathRef.")
    else:
        errors.extend(
            f"{label}: {error}" for error in validate_sample_archive_member_path(member.path)
        )
        if member != expected_member:
            errors.append(
                f"{label}: drive_archive_member must be {expected_member.path.as_posix()}."
            )
    return errors


def _root_relative_path(path: Path, root: Path) -> Path | None:
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def _format_ref(ref: SampleRef) -> str:
    return f"SampleRef(split={ref.split.value}, sample_id={ref.sample_id})"


__all__ = [
    "validate_samples_catalog",
    "validate_tiered_catalog",
]
