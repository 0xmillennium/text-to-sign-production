"""Tar.zst archive specifications and member list helpers."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from text_to_sign_production.workflows._shared.commands import (
    CommandSpec,
    command,
    shell_quote,
)


@dataclass(frozen=True, slots=True)
class ArchiveExtractSpec:
    label: str
    archive_path: Path
    extraction_root: Path
    mkdir_path: Path
    expected_input_bytes: int
    strip_components: int | None
    members: tuple[str, ...]
    command: CommandSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", validate_archive_members(self.members))


@dataclass(frozen=True, slots=True)
class ArchiveCreateSpec:
    label: str
    archive_path: Path
    archive_parent: Path
    source_root: Path
    member_list_path: Path
    members: tuple[str, ...]
    command: CommandSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", validate_archive_members(self.members))


@dataclass(frozen=True, slots=True)
class ArchiveMemberListSpec:
    label: str
    member_list_path: Path
    members: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str):
            raise ValueError("label must be a string")
        label = self.label.strip()
        if not label:
            raise ValueError("label must not be empty")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "members", validate_archive_members(self.members))


@dataclass(frozen=True, slots=True)
class ArchiveVerifySpec:
    label: str
    archive_path: Path
    expected_member_list_path: Path
    observed_member_list_path: Path
    command: CommandSpec


def archive_member_from_path(path: Path, source_root: Path) -> str:
    try:
        relative_path = path.relative_to(source_root)
    except ValueError as error:
        raise ValueError(f"path must be inside source_root: {path}") from error

    member = relative_path.as_posix()
    if not member or member == ".":
        raise ValueError("archive member must not be empty")
    if ".." in PurePosixPath(member).parts:
        raise ValueError(f"archive member must not contain '..': {member}")
    return member


def validate_archive_members(members: Iterable[str]) -> tuple[str, ...]:
    validated: list[str] = []
    seen: set[str] = set()
    for member in members:
        normalized = str(member).strip()
        if not normalized:
            raise ValueError("archive member must not be empty")
        member_path = PurePosixPath(normalized)
        canonical_member = member_path.as_posix()
        if member_path.is_absolute():
            raise ValueError(f"archive member must be relative: {normalized}")
        if canonical_member == ".":
            raise ValueError("archive member must not be empty")
        if ".." in member_path.parts:
            raise ValueError(f"archive member must not contain '..': {normalized}")
        if canonical_member in seen:
            raise ValueError(f"duplicate archive member: {canonical_member}")
        seen.add(canonical_member)
        validated.append(canonical_member)
    return tuple(sorted(validated))


def write_archive_member_list(spec: ArchiveMemberListSpec) -> None:
    spec.member_list_path.parent.mkdir(parents=True, exist_ok=True)
    spec.member_list_path.write_text(
        "".join(f"{member}\n" for member in spec.members),
        encoding="utf-8",
    )


def build_archive_member_list_spec(
    *,
    label: str,
    member_list_path: Path,
    members: Iterable[str],
) -> ArchiveMemberListSpec:
    return ArchiveMemberListSpec(
        label=label,
        member_list_path=member_list_path,
        members=validate_archive_members(members),
    )


def build_tar_zstd_extract_spec(
    *,
    label: str,
    archive_path: Path,
    extraction_root: Path,
    strip_components: int | None = None,
    members: Iterable[str] = (),
) -> ArchiveExtractSpec:
    if not archive_path.is_file():
        raise ValueError(f"archive_path must exist and be a file: {archive_path}")

    member_tuple = tuple(members)
    validated_members = (
        validate_archive_members(member_tuple) if member_tuple else ()
    )
    expected_input_bytes = archive_path.stat().st_size
    strip_option = (
        f" --strip-components {strip_components}" if strip_components is not None else ""
    )
    member_args = (
        " " + " ".join(shell_quote(member) for member in validated_members)
        if validated_members
        else ""
    )
    extract_command = command(
        label=label,
        bash=(
            f"mkdir -p {shell_quote(extraction_root)} && "
            f"tqdm --bytes --total {expected_input_bytes} "
            f"< {shell_quote(archive_path)} | "
            f"tar --zstd -xf - -C {shell_quote(extraction_root)}"
            f"{strip_option}{member_args}"
        ),
        failure=f"Failed to extract archive for {label}.",
    )
    return ArchiveExtractSpec(
        label=extract_command.label,
        archive_path=archive_path,
        extraction_root=extraction_root,
        mkdir_path=extraction_root,
        expected_input_bytes=expected_input_bytes,
        strip_components=strip_components,
        members=validated_members,
        command=extract_command,
    )


def build_tar_zstd_create_spec(
    *,
    label: str,
    archive_path: Path,
    source_root: Path,
    member_list_path: Path,
    members: Iterable[str],
) -> ArchiveCreateSpec:
    if not source_root.is_dir():
        raise ValueError(f"source_root must exist and be a directory: {source_root}")

    validated_members = validate_archive_members(members)
    archive_parent = archive_path.parent
    # Empty archives are intentional only when an explicit empty member list was
    # written before this command runs. A missing member list must fail.
    expected_member_count = len(validated_members)
    create_command = command(
        label=label,
        bash=(
            f"mkdir -p {shell_quote(archive_parent)} && "
            f"tqdm --total {expected_member_count} --unit file "
            f"--desc {shell_quote(label)} < {shell_quote(member_list_path)} | "
            f"tar -cf - -C {shell_quote(source_root)} -T - | "
            f"zstd -f -q -o {shell_quote(archive_path)} -"
        ),
        failure=f"Failed to create archive for {label}.",
    )
    return ArchiveCreateSpec(
        label=create_command.label,
        archive_path=archive_path,
        archive_parent=archive_parent,
        source_root=source_root,
        member_list_path=member_list_path,
        members=validated_members,
        command=create_command,
    )


def build_tar_zstd_verify_spec(
    *,
    label: str,
    archive_path: Path,
    expected_member_list_path: Path,
    observed_member_list_path: Path,
) -> ArchiveVerifySpec:
    verify_command = command(
        label=label,
        bash=(
            f"mkdir -p {shell_quote(observed_member_list_path.parent)} && "
            f"expected_member_count=$(wc -l < {shell_quote(expected_member_list_path)}) && "
            f"tar --zstd -tf {shell_quote(archive_path)} | "
            f"tee {shell_quote(observed_member_list_path)} | "
            f"tqdm --total \"$expected_member_count\" --unit file "
            f"--desc {shell_quote(f'{label} verify')} > /dev/null && "
            f"diff -u {shell_quote(expected_member_list_path)} "
            f"{shell_quote(observed_member_list_path)}"
        ),
        failure=f"Failed to verify archive for {label}.",
    )
    return ArchiveVerifySpec(
        label=verify_command.label,
        archive_path=archive_path,
        expected_member_list_path=expected_member_list_path,
        observed_member_list_path=observed_member_list_path,
        command=verify_command,
    )
