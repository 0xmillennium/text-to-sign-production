"""Private baseline-modeling archive helpers."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from text_to_sign_production.data.utils import ensure_directory
from text_to_sign_production.ops.archive_ops import (
    create_tar_zst_archive_from_snapshot,
    extract_tar_zst_with_progress,
)


def create_baseline_archive(
    *,
    archive_path: Path,
    members: tuple[Path, ...],
    cwd: Path,
    label: str,
    artifact_description: str,
) -> Path:
    """Create a baseline-modeling archive with baseline-specific missing-artifact errors."""

    _require_baseline_archive_members(
        members=members,
        cwd=cwd,
        artifact_description=artifact_description,
    )
    try:
        return create_tar_zst_archive_from_snapshot(
            archive_path=archive_path,
            members=members,
            cwd=cwd,
            label=label,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            _baseline_archive_missing_members_message(
                members=members,
                cwd=cwd,
                artifact_description=artifact_description,
            )
        ) from exc


def extract_baseline_archive_into_run_root(
    archive_path: Path,
    *,
    run_root: Path,
    label: str,
    source_config_name: str,
) -> None:
    """Extract a baseline-modeling archive into an existing run root."""

    ensure_directory(run_root)
    with tempfile.TemporaryDirectory(
        prefix=f".{archive_path.name}.extract-",
        dir=run_root.parent,
    ) as temp_root:
        extraction_destination = Path(temp_root) / "extracted"
        extract_tar_zst_with_progress(archive_path, extraction_destination, label=label)
        if not extraction_destination.is_dir():
            raise RuntimeError(f"Archive extraction did not create outputs: {archive_path}")
        extracted_members = sorted(extraction_destination.iterdir(), key=lambda path: path.name)
        if not extracted_members:
            raise RuntimeError(f"Archive did not contain any outputs: {archive_path}")
        _validate_extracted_config_compatible(
            extracted_config_dir=extraction_destination / "config",
            current_config_dir=run_root / "config",
            source_config_name=source_config_name,
        )
        for source in extracted_members:
            _move_extracted_member_into_run_root(source, run_root=run_root)


def _require_baseline_archive_members(
    *,
    members: tuple[Path, ...],
    cwd: Path,
    artifact_description: str,
) -> None:
    missing_members = _missing_baseline_archive_members(members=members, cwd=cwd)
    if missing_members:
        formatted = "\n".join(f"- {member}" for member in missing_members)
        raise FileNotFoundError(
            f"Cannot archive {artifact_description}; missing required baseline-modeling "
            f"run artifacts:\n{formatted}"
        )


def _baseline_archive_missing_members_message(
    *,
    members: tuple[Path, ...],
    cwd: Path,
    artifact_description: str,
) -> str:
    missing_members = _missing_baseline_archive_members(members=members, cwd=cwd)
    if missing_members:
        formatted = "\n".join(f"- {member}" for member in missing_members)
        return (
            f"Cannot archive {artifact_description}; missing required baseline-modeling "
            f"run artifacts:\n{formatted}"
        )
    formatted = "\n".join(f"- {member}" for member in members)
    return (
        f"Cannot archive {artifact_description}; baseline-modeling archive member "
        f"validation failed while reading run artifacts under {cwd}. Expected members:\n"
        f"{formatted}"
    )


def _missing_baseline_archive_members(
    *,
    members: tuple[Path, ...],
    cwd: Path,
) -> tuple[Path, ...]:
    return tuple(
        member for member in members if not _baseline_archive_member_path(member, cwd).exists()
    )


def _baseline_archive_member_path(member: Path, cwd: Path) -> Path:
    return member if member.is_absolute() else cwd / member


def _move_extracted_member_into_run_root(source: Path, *, run_root: Path) -> None:
    destination = run_root / source.name
    if source.name == "config" and destination.exists():
        return
    if source.name == "archives" and destination.exists():
        if not destination.is_dir():
            raise FileExistsError(
                f"Baseline archives path exists and is not a directory: {destination}"
            )
        return
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    shutil.move(str(source), str(destination))


def _validate_extracted_config_compatible(
    *,
    extracted_config_dir: Path,
    current_config_dir: Path,
    source_config_name: str,
) -> None:
    if not extracted_config_dir.exists():
        return
    if not extracted_config_dir.is_dir():
        raise RuntimeError(
            f"Archived baseline config member is not a directory: {extracted_config_dir}"
        )
    if not current_config_dir.exists():
        return
    if not current_config_dir.is_dir():
        raise FileExistsError(
            f"Baseline config path exists and is not a directory: {current_config_dir}"
        )

    current_source_config = current_config_dir / source_config_name
    archived_source_config = extracted_config_dir / source_config_name
    if not current_source_config.is_file() or not archived_source_config.is_file():
        raise ValueError(
            "Cannot extract archived baseline config over an existing config directory "
            f"because {source_config_name} is missing. Use a new run_name or "
            f"repair the run directory: {current_config_dir}"
        )
    if current_source_config.read_bytes() != archived_source_config.read_bytes():
        raise ValueError(
            "Archived baseline source config differs from the current run config. "
            "Use a new run_name for a different config or remove the existing run "
            f"directory: {current_config_dir}"
        )
