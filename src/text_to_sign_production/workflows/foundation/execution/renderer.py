from __future__ import annotations

import shlex
from pathlib import Path
from typing import Iterable  # noqa: UP035

from text_to_sign_production.workflows.foundation.execution.contracts import (
    ArchiveCreateOperation,
    ArchiveExtractOperation,
    ArchiveVerifyOperation,
    FileCopyOperation,
    WorkflowOperation,
    expected_operation_outputs,
    operation_kind,
)
from text_to_sign_production.workflows.foundation.execution.results import RenderedShellCommand


def render_operation_to_shell(operation: WorkflowOperation) -> RenderedShellCommand:
    if isinstance(operation, FileCopyOperation):
        return _render_file_copy(operation)
    if isinstance(operation, ArchiveExtractOperation):
        return _render_archive_extract(operation)
    if isinstance(operation, ArchiveCreateOperation):
        return _render_archive_create(operation)
    if isinstance(operation, ArchiveVerifyOperation):
        return _render_archive_verify(operation)
    raise TypeError(f"Unsupported workflow operation: {type(operation).__name__}")


def _render_file_copy(operation: FileCopyOperation) -> RenderedShellCommand:
    target_path = operation.target_path
    copy_target = (
        _temporary_path(target_path, ".copying")
        if operation.overwrite_policy == "atomic_replace"
        else target_path
    )
    lines = [
        "set -euo pipefail",
        f"mkdir -p -- {_shell_quote(target_path.parent)}",
    ]
    if operation.overwrite_policy == "forbid":
        lines.extend(
            [
                f"if [ -e {_shell_quote(target_path)} ]; then",
                f"  echo {_shell_quote(f'Target already exists: {target_path}')} >&2",
                "  exit 1",
                "fi",
            ]
        )
    if operation.overwrite_policy == "replace":
        lines.append(f"rm -f -- {_shell_quote(target_path)}")
    if operation.overwrite_policy == "atomic_replace":
        lines.append(f"rm -f -- {_shell_quote(copy_target)}")

    if operation.expected_input_bytes is None:
        lines.append(f"cp -- {_shell_quote(operation.source_path)} {_shell_quote(copy_target)}")
    else:
        lines.append(
            "python -m tqdm --bytes "
            f"--total {_shell_quote(str(operation.expected_input_bytes))} "
            f"--desc {_shell_quote(operation.label)} "
            f"< {_shell_quote(operation.source_path)} > {_shell_quote(copy_target)}"
        )

    if operation.overwrite_policy == "atomic_replace":
        lines.append(f"mv -f -- {_shell_quote(copy_target)} {_shell_quote(target_path)}")

    return RenderedShellCommand(
        label=operation.label,
        operation_kind=operation_kind(operation),
        shell_script=_render_multiline_script(lines),
        display_command=_display_command_for_operation(operation),
        failure_message=operation.failure_message,
        progress=operation.progress,
        expected_outputs=expected_operation_outputs(operation),
    )


def _render_archive_extract(operation: ArchiveExtractOperation) -> RenderedShellCommand:
    if operation.compression_kind != "tar_zst":
        raise ValueError(f"Unsupported archive compression: {operation.compression_kind}")

    tar_options = [
        "tar",
        "--use-compress-program=zstd",
        "-xf",
        str(operation.archive_path) if operation.expected_input_bytes is None else "-",
        "-C",
        str(operation.extraction_root),
    ]
    if operation.strip_components is not None:
        tar_options.append(f"--strip-components={operation.strip_components}")
    if operation.members:
        tar_options.append("--")
    tar_options.extend(operation.members)

    tar_command = " ".join(_shell_quote(option) for option in tar_options)
    lines = [
        "set -euo pipefail",
        f"mkdir -p -- {_shell_quote(operation.extraction_root)}",
    ]
    if operation.expected_input_bytes is None:
        lines.append(tar_command)
    else:
        progress_command = (
            "python -m tqdm --bytes "
            f"--total {_shell_quote(str(operation.expected_input_bytes))} "
            f"--desc {_shell_quote(operation.label)} "
            f"< {_shell_quote(operation.archive_path)}"
        )
        lines.append(f"{progress_command} | {tar_command}")

    return RenderedShellCommand(
        label=operation.label,
        operation_kind=operation_kind(operation),
        shell_script=_render_multiline_script(lines),
        display_command=_display_command_for_operation(operation),
        failure_message=operation.failure_message,
        progress=operation.progress,
        expected_outputs=expected_operation_outputs(operation),
    )


def _render_archive_create(operation: ArchiveCreateOperation) -> RenderedShellCommand:
    if operation.compression_kind != "tar_zst":
        raise ValueError(f"Unsupported archive compression: {operation.compression_kind}")

    archive_path = operation.archive_path
    temporary_archive_path = _temporary_path(archive_path, ".creating")
    member_var_name = "member_file"
    lines = [
        "set -euo pipefail",
        f"mkdir -p -- {_shell_quote(archive_path.parent)}",
        f"tmp_archive={_shell_quote(temporary_archive_path)}",
        f"{member_var_name}=",
        "_cleanup() {",
        f'  if [ -n "${{{member_var_name}:-}}" ]; then rm -f -- "${member_var_name}"; fi',
        '  rm -f -- "$tmp_archive"',
        "}",
        "trap _cleanup EXIT",
    ]
    if operation.overwrite_policy == "forbid":
        lines.extend(
            [
                f"if [ -e {_shell_quote(archive_path)} ]; then",
                f"  echo {_shell_quote(f'Archive already exists: {archive_path}')} >&2",
                "  exit 1",
                "fi",
            ]
        )
    lines.append('rm -f -- "$tmp_archive"')
    lines.extend(_render_member_file_script(member_var_name, operation.members))
    lines.append(
        "python -m tqdm "
        f"--total {_shell_quote(str(len(operation.members)))} "
        "--unit member "
        f"--desc {_shell_quote(operation.label)} "
        f'< "${member_var_name}" | '
        "tar --use-compress-program=zstd "
        f'-cf "$tmp_archive" -C {_shell_quote(operation.source_root)} '
        "--files-from -"
    )
    if operation.overwrite_policy == "replace":
        lines.append(f"rm -f -- {_shell_quote(archive_path)}")
    lines.append(f'mv -f -- "$tmp_archive" {_shell_quote(archive_path)}')

    return RenderedShellCommand(
        label=operation.label,
        operation_kind=operation_kind(operation),
        shell_script=_render_multiline_script(lines),
        display_command=_display_command_for_operation(operation),
        failure_message=operation.failure_message,
        progress=operation.progress,
        expected_outputs=expected_operation_outputs(operation),
    )


def _render_archive_verify(operation: ArchiveVerifyOperation) -> RenderedShellCommand:
    if operation.compression_kind != "tar_zst":
        raise ValueError(f"Unsupported archive compression: {operation.compression_kind}")

    expected_var_name = "expected_members_file"
    lines = [
        "set -euo pipefail",
        f"{expected_var_name}=",
        "observed_members_file=",
        "_cleanup() {",
        f'  if [ -n "${{{expected_var_name}:-}}" ]; then rm -f -- "${expected_var_name}"; fi',
        '  if [ -n "${observed_members_file:-}" ]; then rm -f -- "$observed_members_file"; fi',
        "}",
        "trap _cleanup EXIT",
    ]
    lines.extend(_render_member_file_script(expected_var_name, operation.expected_members))
    lines.extend(
        [
            "observed_members_file=$(mktemp)",
            f'LC_ALL=C sort "${expected_var_name}" -o "${expected_var_name}"',
            "tar --use-compress-program=zstd "
            f"-tf {_shell_quote(operation.archive_path)} "
            "| sed 's#^\\./##' "
            "| python -m tqdm "
            f"--total {_shell_quote(str(len(operation.expected_members)))} "
            "--unit member "
            f"--desc {_shell_quote(operation.label)} "
            '| LC_ALL=C sort > "$observed_members_file"',
            f'diff -u "${expected_var_name}" "$observed_members_file"',
        ]
    )

    return RenderedShellCommand(
        label=operation.label,
        operation_kind=operation_kind(operation),
        shell_script=_render_multiline_script(lines),
        display_command=_display_command_for_operation(operation),
        failure_message=operation.failure_message,
        progress=operation.progress,
        expected_outputs=expected_operation_outputs(operation),
    )


def _shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _render_multiline_script(lines: Iterable[str]) -> str:
    return "\n".join(line for line in lines if line).strip()


def _temporary_path(path: Path, suffix: str) -> Path:
    return path.with_name(f".{path.name}{suffix}")


def _render_member_file_script(member_var_name: str, members: tuple[str, ...]) -> list[str]:
    quoted_members = " ".join(_shell_quote(member) for member in members)
    return [
        f"{member_var_name}=$(mktemp)",
        f"printf '%s\\n' {quoted_members} > \"${member_var_name}\"",
    ]


def _display_command_for_operation(operation: WorkflowOperation) -> str:
    if isinstance(operation, FileCopyOperation):
        return (
            f"copy {_shell_quote(operation.source_path)} -> {_shell_quote(operation.target_path)}"
        )
    if isinstance(operation, ArchiveExtractOperation):
        return (
            f"extract {_shell_quote(operation.archive_path)} "
            f"-> {_shell_quote(operation.extraction_root)}"
        )
    if isinstance(operation, ArchiveCreateOperation):
        return (
            f"archive {_shell_quote(operation.source_root)} "
            f"-> {_shell_quote(operation.archive_path)}"
        )
    if isinstance(operation, ArchiveVerifyOperation):
        return f"verify archive {_shell_quote(operation.archive_path)}"
    raise TypeError(f"Unsupported workflow operation: {type(operation).__name__}")
