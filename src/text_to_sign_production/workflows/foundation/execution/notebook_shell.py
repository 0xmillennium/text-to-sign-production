from __future__ import annotations

import os
import re
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsInt

from text_to_sign_production.workflows.foundation.execution.results import (
    RenderedShellCommand,
    RenderedShellInputFile,
    ShellExecutionResult,
)

_ENV_VAR_NAME_PATTERN = re.compile(r"[A-Z_][A-Z0-9_]*")


@dataclass(slots=True)
class NotebookShellRunner:
    def run(self, rendered: RenderedShellCommand) -> ShellExecutionResult:
        _validate_input_file_env_vars(rendered.input_files)
        with tempfile.TemporaryDirectory(prefix="tts_workflow_shell_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_paths = _materialize_input_files(tmpdir_path, rendered.input_files)
            script_text = _materialized_script_text(rendered, input_paths)
            script_path = tmpdir_path / "operation.sh"
            script_path.write_text(script_text, encoding="utf-8")
            shell_command = f"bash -o pipefail {shlex.quote(str(script_path))}"

            ipython_shell = _ipython_shell()
            if ipython_shell is None:
                execution_mode = "system_script_file"
                returncode = _run_with_os_system(shell_command)
            else:
                execution_mode = "ipython_script_file"
                returncode = _run_in_ipython(shell_command)
        return ShellExecutionResult(
            shell_script=script_text,
            returncode=returncode,
            succeeded=returncode == 0,
            execution_mode=execution_mode,
        )


def run_rendered_shell_command(rendered: RenderedShellCommand) -> ShellExecutionResult:
    return NotebookShellRunner().run(rendered)


def _ipython_shell() -> Any | None:
    try:
        ipython_module = __import__("IPython")
    except ImportError:
        return None
    get_ipython = getattr(ipython_module, "get_ipython", None)
    if get_ipython is None:
        return None
    return get_ipython()


def _validate_input_file_env_vars(input_files: tuple[RenderedShellInputFile, ...]) -> None:
    seen_names: set[str] = set()
    for input_file in input_files:
        env_var_name = input_file.env_var_name
        if _ENV_VAR_NAME_PATTERN.fullmatch(env_var_name) is None:
            raise ValueError(
                f"Invalid shell input file environment variable name: {env_var_name!r}"
            )
        if env_var_name in seen_names:
            raise ValueError(
                f"Duplicate shell input file environment variable name: {env_var_name!r}"
            )
        seen_names.add(env_var_name)


def _materialize_input_files(
    tmpdir_path: Path,
    input_files: tuple[RenderedShellInputFile, ...],
) -> dict[str, Path]:
    input_paths: dict[str, Path] = {}
    for index, input_file in enumerate(input_files):
        input_path = tmpdir_path / f"input_{index}.txt"
        with input_path.open("w", encoding="utf-8") as file:
            for line in input_file.lines:
                file.write(f"{line}\n")
        input_paths[input_file.env_var_name] = input_path
    return input_paths


def _materialized_script_text(
    rendered: RenderedShellCommand,
    input_paths: dict[str, Path],
) -> str:
    env_exports = [f"export {name}={shlex.quote(str(path))}" for name, path in input_paths.items()]
    return "\n".join(
        [
            "set -euo pipefail",
            *env_exports,
            rendered.shell_script,
        ]
    ).strip()


def _run_in_ipython(shell_command: str) -> int:
    shell = _ipython_shell()
    if shell is None:
        raise RuntimeError("IPython shell is not available")
    raw_exit_code = shell.system(shell_command)
    if raw_exit_code is None:
        raw_exit_code = getattr(shell, "user_ns", {}).get("_exit_code", 0)
    return _normalize_returncode(raw_exit_code)


def _run_with_os_system(shell_command: str) -> int:
    return _normalize_returncode(os.system(shell_command))


def _normalize_returncode(raw_exit_code: object) -> int:
    if raw_exit_code is None:
        return 0
    if isinstance(raw_exit_code, bool):
        return int(raw_exit_code)
    if isinstance(raw_exit_code, (str, bytes, bytearray, SupportsInt)):
        returncode = int(raw_exit_code)
    else:
        return 1
    if returncode < 0:
        return 128 + abs(returncode)
    if returncode > 255:
        return _decode_wait_status(returncode)
    return returncode


def _decode_wait_status(status: int) -> int:
    if os.WIFEXITED(status):
        return int(os.WEXITSTATUS(status))
    if os.WIFSIGNALED(status):
        return 128 + int(os.WTERMSIG(status))
    return status
