"""Domain-free tar/zstd command specs and archive-member helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from text_to_sign_production.core.paths import repo_relative_path
from text_to_sign_production.workflows._shared.shell import shell_quote
from text_to_sign_production.workflows.commands import CommandSpec, command


def build_tar_zstd_extract_command(
    *,
    label: str,
    archive_path: Path,
    extraction_root: Path,
    mkdir_path: Path,
    expected_input_bytes: int,
    failure: str,
    strip_components: int | None = None,
    members: Iterable[str] = (),
) -> CommandSpec:
    """Build a visible tar/zstd extraction command without executing it."""

    strip_option = "" if strip_components is None else f" --strip-components={strip_components}"
    member_args = "".join(f" {shell_quote(member)}" for member in members)
    bash = (
        f"mkdir -p {shell_quote(mkdir_path)} && "
        f"tqdm --bytes --total {expected_input_bytes} "
        f"< {shell_quote(archive_path)} "
        f"| tar --zstd{strip_option} -xf - -C {shell_quote(extraction_root)}"
        f"{member_args}"
    )
    return command(label=label, bash=bash, failure=failure)


def build_tar_zstd_create_from_member_list_command(
    *,
    label: str,
    archive_path: Path,
    archive_parent: Path,
    source_root: Path,
    member_list_path: Path,
    expected_member_count: int,
    failure: str,
) -> CommandSpec:
    """Build a visible tar/zstd archive creation command without executing it."""

    bash = (
        f"mkdir -p {shell_quote(archive_parent)} && "
        f"tqdm --total {expected_member_count} --unit file --desc {shell_quote(label)} "
        f"< {shell_quote(member_list_path)} "
        f"| tar -cf - -C {shell_quote(source_root)} -T - "
        f"| zstd -f -q -o {shell_quote(archive_path)} -"
    )
    return command(label=label, bash=bash, failure=failure)


def build_tar_zstd_list_command(
    *,
    label: str,
    archive_path: Path,
    observed_member_list_path: Path,
    expected_member_count: int,
    failure: str,
) -> CommandSpec:
    """Build a visible tar/zstd member-list command without executing it."""

    bash = (
        f"mkdir -p {shell_quote(observed_member_list_path.parent)} && "
        f"tar --zstd -tf {shell_quote(archive_path)} "
        f"| tee {shell_quote(observed_member_list_path)} "
        f"| tqdm --total {expected_member_count} --unit file "
        f"--desc {shell_quote(label + ' verify')} > /dev/null"
    )
    return command(label=f"{label} verify", bash=bash, failure=failure)


def archive_member_from_path(path: Path, *, source_root: Path) -> str:
    """Return a source-root-relative archive member path.

    The returned value is project-root-relative only when ``source_root`` is the
    project root.
    """

    if path.is_absolute():
        return repo_relative_path(path, repo_root=source_root)
    return path.as_posix()


def validate_archive_members(
    members: tuple[str, ...],
    *,
    label: str,
    required_prefix: str,
    error_factory: Callable[[str], Exception] = ValueError,
) -> tuple[str, ...]:
    """Validate primitive archive member path safety constraints."""

    if not members:
        raise error_factory(f"{label} archive must include at least one member.")
    seen_members: set[str] = set()
    invalid_members: list[str] = []
    for member in members:
        if (
            not member
            or member in seen_members
            or member.startswith("/")
            or "\\" in member
            or ".." in member.split("/")
            or not member.startswith(required_prefix)
        ):
            invalid_members.append(member)
        seen_members.add(member)
    if invalid_members:
        preview = "\n".join(invalid_members[:10])
        raise error_factory(
            "Archive member paths must be unique, relative, and start with "
            f"{required_prefix!r} for {label}. Invalid members:\n{preview}"
        )
    return members


__all__ = [
    "archive_member_from_path",
    "build_tar_zstd_create_from_member_list_command",
    "build_tar_zstd_extract_command",
    "build_tar_zstd_list_command",
    "validate_archive_members",
]
