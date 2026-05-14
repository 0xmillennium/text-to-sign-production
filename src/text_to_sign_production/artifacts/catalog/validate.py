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
)
from text_to_sign_production.artifacts.store.validate import (
    validate_sample_archive_member_path,
    validate_sample_archive_relative_path,
    validate_samples_relative_path,
)
from text_to_sign_production.core.ids import SampleSplit, SampleStatus


def _expected_archive_member_for_status(
    *,
    stores: ArtifactStores,
    status: SampleStatus,
    split: SampleSplit,
    sample_id: str,
) -> ArchiveMemberPathRef:
    """Return the expected archive member path for the given sample status.

    Passed PreparedSample payloads use ``.npz``.
    Dropped DroppedSample payloads use ``.json``.
    Status determines the physical member extension.
    """
    if status is SampleStatus.PASSED:
        return stores.drive.samples.passed_archive_member(split, sample_id)
    if status is SampleStatus.DROPPED:
        return stores.drive.samples.dropped_archive_member(split, sample_id)
    raise ValueError(f"Unsupported sample status for archive member validation: {status}")


def validate_samples_catalog(catalog: SamplesCatalog, stores: ArtifactStores) -> list[str]:
    """Validate logical invariants for a samples catalog."""

    errors: list[str] = []
    for ref, handle in catalog.items.items():
        label = _format_ref(ref)
        if handle.ref != ref:
            errors.append(f"{label}: handle ref must match the catalog key.")
        if handle.status is not catalog.status:
            errors.append(f"{label}: handle status must match catalog status.")
        if handle.manifest.sample_id != ref.sample_id:
            errors.append(f"{label}: manifest sample_id must match ref sample_id.")
        if handle.manifest.split != ref.split:
            errors.append(f"{label}: manifest split must match ref split.")
        if handle.manifest.projected_status is not catalog.status:
            errors.append(f"{label}: manifest projected status must match catalog status.")
        declared_payload_ref = handle.manifest.payload_ref
        if catalog.status is SampleStatus.PASSED and declared_payload_ref is None:
            errors.append(f"{label}: passed manifest entry must declare a payload ref.")
        if handle.manifest.archive_publishable and not handle.manifest.payload_declared_present:
            errors.append(
                f"{label}: archive-publishable payload must be declared physically present."
            )
        if handle.manifest.archive_publishable and declared_payload_ref is None:
            errors.append(f"{label}: archive-publishable payload must declare a payload ref.")
        if declared_payload_ref is None:
            if handle.runtime_sample is not None:
                errors.append(f"{label}: runtime_sample must be absent when manifest omits it.")
            if handle.drive_archive is not None:
                errors.append(f"{label}: drive_archive must be absent when manifest omits it.")
            if handle.drive_archive_member is not None:
                errors.append(
                    f"{label}: drive_archive_member must be absent when manifest omits it."
                )
        else:
            manifest_path_errors = _validate_manifest_payload_ref(
                declared_payload_ref,
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
                    declared_payload_ref=declared_payload_ref,
                    manifest_path_is_valid=not manifest_path_errors,
                )
            )
            if handle.manifest.archive_publishable:
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
            else:
                if handle.drive_archive is not None:
                    errors.append(
                        f"{label}: drive_archive must be absent when payload is not "
                        "archive-publishable."
                    )
                if handle.drive_archive_member is not None:
                    errors.append(
                        f"{label}: drive_archive_member must be absent when payload is not "
                        "archive-publishable."
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
        if handle.manifest.sample_id != ref.sample_id:
            errors.append(f"{label}: manifest sample_id must match ref sample_id.")
        if handle.manifest.split != ref.split:
            errors.append(f"{label}: manifest split must match ref split.")
        if handle.manifest.projected_status is not SampleStatus.PASSED:
            errors.append(f"{label}: tiered manifest projection must be passed.")
        declared_payload_ref = handle.manifest.payload_ref
        manifest_path_errors = _validate_manifest_payload_ref(
            declared_payload_ref,
            label=label,
            required_status=SampleStatus.PASSED,
            expected_split=ref.split,
            expected_sample_id=ref.sample_id,
        )
        errors.extend(manifest_path_errors)
        if declared_payload_ref is not None:
            errors.extend(
                _validate_runtime_sample_binding(
                    stores=stores,
                    sample=handle.runtime_sample,
                    label=label,
                    declared_payload_ref=declared_payload_ref,
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
    sample: SamplePathRef | None,
    label: str,
    declared_payload_ref: str,
    manifest_path_is_valid: bool,
) -> list[str]:
    errors: list[str] = []
    if sample is None:
        errors.append(f"{label}: runtime_sample must resolve the manifest-declared payload ref.")
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
        expected_sample = resolve_samples_relative(stores.runtime, declared_payload_ref)
        if sample != expected_sample:
            errors.append(f"{label}: runtime_sample must match manifest payload_ref.")
    return errors


def _validate_manifest_payload_ref(
    payload_ref: str | None,
    *,
    label: str,
    required_status: SampleStatus,
    expected_split: SampleSplit,
    expected_sample_id: str,
) -> list[str]:
    if payload_ref is None:
        return [f"{label}: manifest entry must declare a payload ref."]

    errors = [f"{label}: {error}" for error in validate_samples_relative_path(payload_ref)]
    if errors:
        return errors

    expected_path = sample_manifest_relative_path(
        required_status,
        expected_split,
        expected_sample_id,
    )
    if Path(payload_ref) != expected_path:
        errors.append(f"{label}: manifest payload_ref must be {expected_path.as_posix()}.")
    return errors


def _validate_drive_sample_binding(
    *,
    stores: ArtifactStores,
    archive: ArchivePathRef | None,
    member: ArchiveMemberPathRef | None,
    label: str,
    status: SampleStatus,
    split: SampleSplit,
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

    expected_member = _expected_archive_member_for_status(
        stores=stores,
        status=status,
        split=split,
        sample_id=sample_id,
    )
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
