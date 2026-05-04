"""Generic file transfer specifications for notebook-driven workflows."""

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.workflows._shared.commands import (
    CommandSpec,
    command,
    shell_quote,
)


@dataclass(frozen=True, slots=True)
class FileTransferSpec:
    label: str
    source_path: Path
    target_path: Path
    expected_input_bytes: int
    command: CommandSpec


def build_byte_progress_copy_spec(
    *, label: str, source_path: Path, target_path: Path
) -> FileTransferSpec:
    if not source_path.is_file():
        raise ValueError(f"source_path must exist and be a file: {source_path}")

    expected_input_bytes = source_path.stat().st_size
    temporary_target_path = target_path.with_name(f".{target_path.name}.tmp")
    copy_command = command(
        label=label,
        bash=(
            f"mkdir -p {shell_quote(target_path.parent)} && "
            f"tqdm --bytes --total {expected_input_bytes} "
            f"< {shell_quote(source_path)} > {shell_quote(temporary_target_path)} && "
            f"mv -f {shell_quote(temporary_target_path)} {shell_quote(target_path)}"
        ),
        failure=f"Failed to copy file for {label}.",
    )
    return FileTransferSpec(
        label=copy_command.label,
        source_path=source_path,
        target_path=target_path,
        expected_input_bytes=expected_input_bytes,
        command=copy_command,
    )
