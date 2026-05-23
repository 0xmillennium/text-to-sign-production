"""Archive packaging for generated-pose sample payloads."""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.core.integrity import sha256_file
from text_to_sign_production.core.io import write_json
from text_to_sign_production.modeling.artifacts.generated_pose_io import (
    read_generated_pose_manifest_jsonl,
)

GENERATED_POSE_SAMPLE_ARCHIVE_SCHEMA_VERSION = "model.generated_pose_samples_archive.v1"
GENERATED_POSE_SAMPLE_ARCHIVE_NAME = "samples.tar.zst"
GENERATED_POSE_SAMPLE_ARCHIVE_MANIFEST_NAME = "samples_archive_manifest.json"
GENERATED_POSE_SAMPLE_ARCHIVE_SHA256_NAME = "samples_archive_sha256.txt"


class GeneratedPoseSampleArchiveError(RuntimeError):
    """Raised when generated-pose sample archive packaging fails."""


@dataclass(frozen=True, slots=True)
class GeneratedPoseSampleArchive:
    model_run_name: str
    model_key: str
    manifest_family: str
    split: SampleSplit
    manifest_path: Path
    archive_path: Path
    archive_manifest_path: Path
    archive_sha256_path: Path
    sample_count: int
    member_count: int
    archive_sha256: str
    generated_manifest_sha256: str
    members: tuple[dict[str, object], ...]

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "model_run_name": self.model_run_name,
            "model_key": self.model_key,
            "manifest_family": self.manifest_family,
            "split": self.split.value,
            "generated_sample_count": self.sample_count,
            "archive_path": str(self.archive_path),
            "archive_manifest_path": str(self.archive_manifest_path),
            "archive_sha256_path": str(self.archive_sha256_path),
            "archive_member_count": self.member_count,
            "archive_sha256": self.archive_sha256,
            "generated_manifest_path": str(self.manifest_path),
            "generated_manifest_sha256": self.generated_manifest_sha256,
            "individual_sample_publish": False,
        }


def build_generated_pose_samples_archive(
    *,
    manifest_path: Path,
    samples_root: Path,
    model_run_name: str,
    model_key: str,
    manifest_family: str,
    split: SampleSplit | str = SampleSplit.VAL,
    on_event: Callable[[str, dict[str, object]], None] | None = None,
    on_member: Callable[[int, int, str, int], None] | None = None,
) -> GeneratedPoseSampleArchive:
    """Create tar.zst archive and sidecar manifest for one generated-pose split."""

    resolved_split = SampleSplit(split)
    manifest_path = Path(manifest_path)
    samples_root = Path(samples_root)
    split_root = samples_root.parent
    archive_path = split_root / GENERATED_POSE_SAMPLE_ARCHIVE_NAME
    archive_manifest_path = split_root / GENERATED_POSE_SAMPLE_ARCHIVE_MANIFEST_NAME
    archive_sha256_path = split_root / GENERATED_POSE_SAMPLE_ARCHIVE_SHA256_NAME
    entries = read_generated_pose_manifest_jsonl(
        manifest_path,
        expected_split=resolved_split,
    )
    if not entries:
        raise GeneratedPoseSampleArchiveError(
            f"generated-pose manifest has no entries: {manifest_path}"
        )
    members: list[dict[str, object]] = []
    member_paths: list[str] = []
    _emit_archive_event(on_event, "archive_start", {"member_count": len(entries)})
    for index, entry in enumerate(entries, start=1):
        filename = _generated_payload_filename(entry)
        member_path = f"samples/{filename}"
        sample_path = samples_root / filename
        if not sample_path.is_file():
            raise GeneratedPoseSampleArchiveError(
                f"generated-pose sample listed by manifest does not exist: {sample_path}"
            )
        stat = sample_path.stat()
        member_paths.append(member_path)
        digest = sha256_file(sample_path)
        members.append(
            {
                "member_path": member_path,
                "sample_id": entry.sample_id,
                "source_sentence_name": entry.source_sentence_name,
                "sha256": digest,
                "size_bytes": stat.st_size,
            }
        )
        if on_member is not None:
            on_member(index, len(entries), entry.sample_id, stat.st_size)
    _emit_archive_event(on_event, "archive_compression_start", {"member_count": len(entries)})
    _create_tar_zst_archive(
        source_root=split_root,
        archive_path=archive_path,
        members=tuple(member_paths),
    )
    _emit_archive_event(on_event, "archive_compression_done", {"archive_path": str(archive_path)})
    archive_sha256 = sha256_file(archive_path)
    generated_manifest_sha256 = sha256_file(manifest_path)
    archive_sha256_path.parent.mkdir(parents=True, exist_ok=True)
    archive_sha256_path.write_text(archive_sha256 + "\n", encoding="utf-8")
    _emit_archive_event(on_event, "sha256_sidecar_write_done", {"path": str(archive_sha256_path)})
    payload = {
        "schema_version": GENERATED_POSE_SAMPLE_ARCHIVE_SCHEMA_VERSION,
        "model_run_name": model_run_name,
        "model_key": model_key,
        "manifest_family": manifest_family,
        "split": resolved_split.value,
        "sample_count": len(entries),
        "archive_format": "tar.zst",
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "generated_manifest_path": str(manifest_path),
        "generated_manifest_sha256": generated_manifest_sha256,
        "member_count": len(members),
        "members": members,
    }
    write_json(archive_manifest_path, payload)
    _emit_archive_event(on_event, "archive_manifest_write_done", {"path": str(archive_manifest_path)})
    _emit_archive_event(on_event, "archive_done", {"archive_sha256": archive_sha256})
    return GeneratedPoseSampleArchive(
        model_run_name=model_run_name,
        model_key=model_key,
        manifest_family=manifest_family,
        split=resolved_split,
        manifest_path=manifest_path,
        archive_path=archive_path,
        archive_manifest_path=archive_manifest_path,
        archive_sha256_path=archive_sha256_path,
        sample_count=len(entries),
        member_count=len(members),
        archive_sha256=archive_sha256,
        generated_manifest_sha256=generated_manifest_sha256,
        members=tuple(members),
    )


def _generated_payload_filename(entry) -> str:
    if entry.generated_payload_ref is not None:
        return Path(entry.generated_payload_ref).name
    raise GeneratedPoseSampleArchiveError(
        f"generated-pose manifest entry has no generated_payload_ref: {entry.sample_id}"
    )


def _emit_archive_event(
    callback: Callable[[str, dict[str, object]], None] | None,
    event: str,
    payload: dict[str, object],
) -> None:
    if callback is not None:
        callback(event, payload)


def _create_tar_zst_archive(
    *,
    source_root: Path,
    archive_path: Path,
    members: tuple[str, ...],
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive_path.with_name(f"{archive_path.name}.creating")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as member_file:
        member_file.write("\n".join(members) + "\n")
        member_file_path = Path(member_file.name)
    try:
        if temporary_archive.exists():
            temporary_archive.unlink()
        subprocess.run(
            [
                "tar",
                "--use-compress-program=zstd",
                "-cf",
                str(temporary_archive),
                "-C",
                str(source_root),
                "--verbatim-files-from",
                "--files-from",
                str(member_file_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        os.replace(temporary_archive, archive_path)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            detail = f": {exc.stderr.strip()}"
        raise GeneratedPoseSampleArchiveError(
            f"failed to create generated-pose samples archive {archive_path}{detail}"
        ) from exc
    finally:
        try:
            member_file_path.unlink()
        except FileNotFoundError:
            pass
        try:
            temporary_archive.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "GENERATED_POSE_SAMPLE_ARCHIVE_MANIFEST_NAME",
    "GENERATED_POSE_SAMPLE_ARCHIVE_NAME",
    "GENERATED_POSE_SAMPLE_ARCHIVE_SCHEMA_VERSION",
    "GENERATED_POSE_SAMPLE_ARCHIVE_SHA256_NAME",
    "GeneratedPoseSampleArchive",
    "GeneratedPoseSampleArchiveError",
    "build_generated_pose_samples_archive",
]
