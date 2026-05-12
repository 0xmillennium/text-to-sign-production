from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, TypeAlias

from text_to_sign_production.core.progress import ProgressStageSpec

OperationKind: TypeAlias = Literal[
    "file_copy",
    "archive_extract",
    "archive_create",
    "archive_verify",
]
OperationLiveOwner: TypeAlias = Literal["shell", "python"]
OperationOverwritePolicy: TypeAlias = Literal["forbid", "replace", "atomic_replace"]
ArchiveCompressionKind: TypeAlias = Literal["tar_zst"]
RuntimeCheckScope: TypeAlias = Literal["asset", "domain"]


class ReadinessLevel(enum.StrEnum):
    """Named runtime readiness contract levels."""

    FAST_READINESS = "fast_readiness"
    STRUCTURAL_READINESS = "structural_readiness"
    EXHAUSTIVE_READINESS = "exhaustive_readiness"


@dataclass(frozen=True, slots=True)
class RuntimeReadinessScope:
    """Typed statement of what a runtime verify pass checked."""

    readiness_level: ReadinessLevel
    checked_semantics: tuple[str, ...]
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "readiness_level", ReadinessLevel(self.readiness_level))
        object.__setattr__(
            self,
            "checked_semantics",
            tuple(
                semantic.strip()
                for semantic in self.checked_semantics
                if isinstance(semantic, str) and semantic.strip()
            ),
        )
        object.__setattr__(
            self,
            "limitations",
            tuple(
                limitation.strip()
                for limitation in self.limitations
                if isinstance(limitation, str) and limitation.strip()
            ),
        )


@dataclass(frozen=True, slots=True)
class OperationProgressSpec:
    stage: ProgressStageSpec
    expected_total: int | None = None
    live_owner: OperationLiveOwner = "shell"

    def __post_init__(self) -> None:
        _ensure_non_negative_int("expected_total", self.expected_total)


@dataclass(frozen=True, slots=True)
class FileCopyOperation:
    label: str
    source_path: Path
    target_path: Path
    failure_message: str
    overwrite_policy: OperationOverwritePolicy = "atomic_replace"
    expected_input_bytes: int | None = None
    progress: OperationProgressSpec | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        _validate_non_empty_text("failure_message", self.failure_message)
        if not isinstance(self.source_path, Path):
            raise TypeError("source_path must be a Path")
        if not isinstance(self.target_path, Path):
            raise TypeError("target_path must be a Path")
        if self.source_path == self.target_path:
            raise ValueError("source_path and target_path must differ")
        _ensure_non_negative_int("expected_input_bytes", self.expected_input_bytes)


@dataclass(frozen=True, slots=True)
class ArchiveExtractOperation:
    label: str
    archive_path: Path
    extraction_root: Path
    failure_message: str
    compression_kind: ArchiveCompressionKind = "tar_zst"
    strip_components: int | None = None
    members: tuple[str, ...] = ()
    expected_input_bytes: int | None = None
    progress: OperationProgressSpec | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        _validate_non_empty_text("failure_message", self.failure_message)
        if not isinstance(self.archive_path, Path):
            raise TypeError("archive_path must be a Path")
        if not isinstance(self.extraction_root, Path):
            raise TypeError("extraction_root must be a Path")
        _ensure_non_negative_int("strip_components", self.strip_components)
        _ensure_non_negative_int("expected_input_bytes", self.expected_input_bytes)
        object.__setattr__(self, "members", _normalize_archive_members(self.members))


@dataclass(frozen=True, slots=True)
class ArchiveCreateOperation:
    label: str
    archive_path: Path
    source_root: Path
    members: tuple[str, ...]
    failure_message: str
    compression_kind: ArchiveCompressionKind = "tar_zst"
    overwrite_policy: OperationOverwritePolicy = "atomic_replace"
    progress: OperationProgressSpec | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        _validate_non_empty_text("failure_message", self.failure_message)
        if not isinstance(self.archive_path, Path):
            raise TypeError("archive_path must be a Path")
        if not isinstance(self.source_root, Path):
            raise TypeError("source_root must be a Path")
        normalized_members = _normalize_archive_members(self.members)
        if not normalized_members:
            raise ValueError("members must not be empty")
        object.__setattr__(self, "members", normalized_members)


@dataclass(frozen=True, slots=True)
class ArchiveVerifyOperation:
    label: str
    archive_path: Path
    expected_members: tuple[str, ...]
    failure_message: str
    compression_kind: ArchiveCompressionKind = "tar_zst"
    progress: OperationProgressSpec | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        _validate_non_empty_text("failure_message", self.failure_message)
        if not isinstance(self.archive_path, Path):
            raise TypeError("archive_path must be a Path")
        normalized_members = _normalize_archive_members(self.expected_members)
        if not normalized_members:
            raise ValueError("expected_members must not be empty")
        object.__setattr__(self, "expected_members", normalized_members)


WorkflowOperation: TypeAlias = (
    FileCopyOperation | ArchiveExtractOperation | ArchiveCreateOperation | ArchiveVerifyOperation
)


@dataclass(frozen=True, slots=True)
class RuntimeCheckResult:
    """One typed runtime readiness check result."""

    label: str
    scope: RuntimeCheckScope
    exists: bool
    valid: bool = True
    message: str | None = None


def operation_kind(operation: WorkflowOperation) -> OperationKind:
    if isinstance(operation, FileCopyOperation):
        return "file_copy"
    if isinstance(operation, ArchiveExtractOperation):
        return "archive_extract"
    if isinstance(operation, ArchiveCreateOperation):
        return "archive_create"
    if isinstance(operation, ArchiveVerifyOperation):
        return "archive_verify"
    raise TypeError(f"Unsupported workflow operation: {type(operation).__name__}")


def expected_operation_outputs(operation: WorkflowOperation) -> tuple[Path, ...]:
    if isinstance(operation, FileCopyOperation):
        return (operation.target_path,)
    if isinstance(operation, ArchiveExtractOperation):
        return (operation.extraction_root,)
    if isinstance(operation, ArchiveCreateOperation):
        return (operation.archive_path,)
    if isinstance(operation, ArchiveVerifyOperation):
        return ()
    raise TypeError(f"Unsupported workflow operation: {type(operation).__name__}")


def _validate_non_empty_text(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _ensure_non_negative_int(field_name: str, value: int | None) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} must be >= 0")


def _normalize_archive_member(member: str) -> str:
    if not isinstance(member, str) or not member.strip():
        raise ValueError("archive members must be non-empty strings")
    if "\\" in member:
        raise ValueError(f"archive member {member!r} must use POSIX separators")
    if "\n" in member or "\r" in member:
        raise ValueError(f"archive member {member!r} must not contain newlines")

    path = PurePosixPath(member)
    if path.is_absolute():
        raise ValueError(f"archive member {member!r} must be relative")
    if ".." in path.parts:
        raise ValueError(f"archive member {member!r} must not contain '..' path segments")

    normalized = PurePosixPath(*path.parts).as_posix()
    if normalized in {"", "."}:
        raise ValueError("archive members must not be empty")
    return normalized


def _normalize_archive_members(members: tuple[str, ...]) -> tuple[str, ...]:
    normalized_members = tuple(_normalize_archive_member(member) for member in members)
    if len(set(normalized_members)) != len(normalized_members):
        raise ValueError("archive members must not contain duplicates")
    return normalized_members
