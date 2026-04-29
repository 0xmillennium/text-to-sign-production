"""Domain-free transfer command spec builders for notebook-visible workflows."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.workflows._shared.shell import shell_quote
from text_to_sign_production.workflows.commands import CommandSpec, command


def build_byte_progress_copy_command(
    *,
    label: str,
    source_path: Path,
    target_path: Path,
    expected_input_bytes: int,
    failure: str,
) -> CommandSpec:
    """Build a visible byte-progress file copy command without executing it."""

    bash = (
        f"mkdir -p {shell_quote(target_path.parent)} && "
        f"tqdm --bytes --total {expected_input_bytes} "
        f"< {shell_quote(source_path)} > {shell_quote(target_path)}"
    )
    return command(label=label, bash=bash, failure=failure)


__all__ = ["build_byte_progress_copy_command"]
