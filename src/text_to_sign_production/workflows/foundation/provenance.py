"""Small provenance and receipt DTOs for workflow file boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from text_to_sign_production.core.integrity import sha256_file, sha256_json

MaterializedArtifactKind: TypeAlias = Literal[
    "prepared_sample_payload",
    "gate_manifest",
    "tiered_manifest",
    "gate_report",
    "tier_report",
    "archive_file",
]


@dataclass(frozen=True, slots=True)
class SemanticSentenceIdentity:
    """Sentence-level semantic grouping identity.

    This value is allowed to repeat across physical views and must not be used
    as a payload, manifest, or archive member identity.
    """

    sentence_id: str


@dataclass(frozen=True, slots=True)
class PhysicalSampleIdentity:
    """Physical sample identity used for payloads and manifests."""

    split: str
    sample_id: str


@dataclass(frozen=True, slots=True)
class MaterializationIdentity:
    """Identity for one materialized artifact/member."""

    kind: str
    path: Path
    physical_sample: PhysicalSampleIdentity | None = None


@dataclass(frozen=True, slots=True)
class FileProvenance:
    """Resolved input-file provenance recorded by workflow execution.

    ``original_path`` is where the workflow found the input. ``execution_path``
    is the path processing is allowed to read. The digest is the planned content
    fingerprint that verification must enforce at ``execution_path``.
    """

    label: str
    original_path: Path
    execution_path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class WrittenFileReceipt:
    """Receipt for a file materialized by the current workflow execution."""

    label: str
    kind: str
    path: Path
    sha256: str
    execution_id: str
    physical_sample: PhysicalSampleIdentity | None = None
    semantic_sentence: SemanticSentenceIdentity | None = None
    source_lineage: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceMemberTreeReceipt:
    """Receipt for the unarchived source member tree planned for archive creation."""

    label: str
    source_root: Path
    member_count: int
    member_listing_digest: str
    member_tree_digest: str
    execution_id: str
    members_preview: tuple[str, ...]
    source_lineage: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ArchiveFileReceipt:
    """Receipt for an archive file's compressed bytes."""

    label: str
    kind: str
    path: Path
    sha256: str
    execution_id: str


@dataclass(frozen=True, slots=True)
class PublishedArchiveReceipt:
    """Receipt for a published target archive and its observable member listing."""

    label: str
    kind: str
    path: Path
    sha256: str
    execution_id: str
    member_count: int
    member_listing_digest: str
    members_preview: tuple[str, ...]


def file_provenance(
    label: str,
    original_path: Path,
    *,
    execution_path: Path | None = None,
) -> FileProvenance:
    """Fingerprint a planned input file and bind it to an execution path."""
    resolved_original = original_path.expanduser().resolve()
    resolved_execution = (
        resolved_original
        if execution_path is None
        else execution_path.expanduser().resolve()
    )
    return FileProvenance(
        label=label,
        original_path=resolved_original,
        execution_path=resolved_execution,
        sha256=sha256_file(resolved_original),
    )


def written_file_receipt(
    label: str,
    path: Path,
    *,
    execution_id: str,
    kind: str,
    physical_sample: PhysicalSampleIdentity | None = None,
    semantic_sentence: SemanticSentenceIdentity | None = None,
    source_lineage: tuple[str, ...] = (),
) -> WrittenFileReceipt:
    """Build a digest-bearing receipt for a materialized file."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Written file receipt target is not a file: {resolved}")
    return WrittenFileReceipt(
        label=label,
        kind=kind,
        path=resolved,
        sha256=sha256_file(resolved),
        execution_id=execution_id,
        physical_sample=physical_sample,
        semantic_sentence=semantic_sentence,
        source_lineage=tuple(source_lineage),
    )


def source_member_tree_receipt(
    label: str,
    source_root: Path,
    members: tuple[str, ...],
    *,
    execution_id: str,
    source_lineage: tuple[str, ...] = (),
) -> SourceMemberTreeReceipt:
    """Build source-tree integrity truth for an unarchived member set."""
    resolved_root = source_root.expanduser().resolve()
    if len(set(members)) != len(members):
        raise ValueError("source member tree members must be unique")
    member_tree_rows = tuple(
        {
            "member": member,
            "sha256": sha256_file(resolved_root / member),
        }
        for member in members
    )
    return SourceMemberTreeReceipt(
        label=label,
        source_root=resolved_root,
        member_count=len(members),
        member_listing_digest=archive_member_listing_fingerprint(members),
        member_tree_digest=sha256_json(member_tree_rows),
        execution_id=execution_id,
        members_preview=members[:10],
        source_lineage=tuple(source_lineage),
    )


def archive_file_receipt(
    label: str,
    path: Path,
    *,
    execution_id: str,
    kind: str = "archive_file",
) -> ArchiveFileReceipt:
    """Build compressed archive-file byte integrity truth."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Archive receipt target is not a file: {resolved}")
    return ArchiveFileReceipt(
        label=label,
        kind=kind,
        path=resolved,
        sha256=sha256_file(resolved),
        execution_id=execution_id,
    )


def archive_member_listing_fingerprint(members: tuple[str, ...]) -> str:
    """Fingerprint an ordered archive member listing deterministically."""
    return sha256_json(tuple(members))


__all__ = [
    "ArchiveFileReceipt",
    "FileProvenance",
    "MaterializationIdentity",
    "MaterializedArtifactKind",
    "PhysicalSampleIdentity",
    "PublishedArchiveReceipt",
    "SemanticSentenceIdentity",
    "SourceMemberTreeReceipt",
    "WrittenFileReceipt",
    "archive_member_listing_fingerprint",
    "archive_file_receipt",
    "file_provenance",
    "source_member_tree_receipt",
    "written_file_receipt",
]
