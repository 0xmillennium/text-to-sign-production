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
from text_to_sign_production.workflows.foundation.execution.results import (
    RenderedShellCommand,
    RenderedShellInputFile,
)


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

    input_files: tuple[RenderedShellInputFile, ...] = ()
    member_env_var_name = "TSP_EXTRACT_MEMBERS_FILE"
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
        tar_options.extend(
            [
                "--verbatim-files-from",
                "--files-from",
                _shell_env_file_ref(member_env_var_name),
            ]
        )
        input_files = (
            _member_input_file(
                env_var_name=member_env_var_name,
                members=operation.members,
            ),
        )

    tar_command = _render_shell_words(tar_options)
    lines = [
        "set -euo pipefail",
        f"mkdir -p -- {_shell_quote(operation.extraction_root)}",
    ]
    if operation.members:
        lines.append(f': "${{{member_env_var_name}:?missing extract member file}}"')
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
        input_files=input_files,
    )


def _render_archive_create(operation: ArchiveCreateOperation) -> RenderedShellCommand:
    if operation.compression_kind != "tar_zst":
        raise ValueError(f"Unsupported archive compression: {operation.compression_kind}")

    archive_path = operation.archive_path
    temporary_archive_path = _temporary_path(archive_path, ".creating")
    member_env_var_name = "TSP_ARCHIVE_MEMBERS_FILE"
    lines = [
        "set -euo pipefail",
        f"mkdir -p -- {_shell_quote(archive_path.parent)}",
        f"tmp_archive={_shell_quote(temporary_archive_path)}",
        "_cleanup() {",
        '  rm -f -- "$tmp_archive"',
        "}",
        "trap _cleanup EXIT",
        f': "${{{member_env_var_name}:?missing archive member file}}"',
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
    lines.append(
        "python -m tqdm "
        f"--total {_shell_quote(str(len(operation.members)))} "
        "--unit member "
        f"--desc {_shell_quote(operation.label)} "
        f'< "${member_env_var_name}" | '
        "tar --use-compress-program=zstd "
        f'-cf "$tmp_archive" -C {_shell_quote(operation.source_root)} '
        "--verbatim-files-from "
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
        input_files=(
            _member_input_file(
                env_var_name=member_env_var_name,
                members=operation.members,
            ),
        ),
    )


def _render_archive_verify(operation: ArchiveVerifyOperation) -> RenderedShellCommand:
    if operation.compression_kind != "tar_zst":
        raise ValueError(f"Unsupported archive compression: {operation.compression_kind}")

    expected_env_var_name = "TSP_EXPECTED_ARCHIVE_MEMBERS_FILE"
    lines = [
        "set -euo pipefail",
        f': "${{{expected_env_var_name}:?missing expected archive member file}}"',
        "expected_members_file=",
        "observed_members_file=",
        "_cleanup() {",
        '  if [ -n "${expected_members_file:-}" ]; then rm -f -- "$expected_members_file"; fi',
        '  if [ -n "${observed_members_file:-}" ]; then rm -f -- "$observed_members_file"; fi',
        "}",
        "trap _cleanup EXIT",
    ]
    lines.extend(
        [
            "expected_members_file=$(mktemp)",
            "observed_members_file=$(mktemp)",
            f'LC_ALL=C sort "${expected_env_var_name}" > "$expected_members_file"',
            "tar --use-compress-program=zstd "
            f"-tf {_shell_quote(operation.archive_path)} "
            "| sed 's#^\\./##' "
            "| python -m tqdm "
            f"--total {_shell_quote(str(len(operation.expected_members)))} "
            "--unit member "
            f"--desc {_shell_quote(operation.label)} "
            '| LC_ALL=C sort > "$observed_members_file"',
            'diff -u "$expected_members_file" "$observed_members_file"',
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
        input_files=(
            _member_input_file(
                env_var_name=expected_env_var_name,
                members=operation.expected_members,
            ),
        ),
    )


def _shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _shell_env_file_ref(env_var_name: str) -> str:
    return f'"${{{env_var_name}}}"'


def _render_multiline_script(lines: Iterable[str]) -> str:
    return "\n".join(line for line in lines if line).strip()


def _render_shell_words(words: Iterable[str | Path]) -> str:
    rendered_words = []
    for word in words:
        text = str(word)
        if text.startswith('"${') and text.endswith('}"'):
            rendered_words.append(text)
        else:
            rendered_words.append(_shell_quote(text))
    return " ".join(rendered_words)


def _temporary_path(path: Path, suffix: str) -> Path:
    return path.with_name(f".{path.name}{suffix}")


def _member_input_file(
    *,
    env_var_name: str,
    members: tuple[str, ...],
) -> RenderedShellInputFile:
    return RenderedShellInputFile(
        env_var_name=env_var_name,
        lines=members,
    )


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
