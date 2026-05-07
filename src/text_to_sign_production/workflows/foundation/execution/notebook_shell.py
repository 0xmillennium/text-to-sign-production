from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, SupportsInt

from text_to_sign_production.workflows.foundation.execution.results import (
    RenderedShellCommand,
    ShellExecutionResult,
)


@dataclass(slots=True)
class NotebookShellRunner:
    def run(self, rendered: RenderedShellCommand) -> ShellExecutionResult:
        shell_command = _wrap_for_bash_pipefail(rendered.shell_script)
        ipython_shell = _ipython_shell()
        if ipython_shell is None:
            execution_mode = "system_shell"
            returncode = _run_with_os_system(shell_command)
        else:
            execution_mode = "ipython_shell"
            returncode = _run_in_ipython(shell_command)
        return ShellExecutionResult(
            shell_script=rendered.shell_script,
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


def _wrap_for_bash_pipefail(shell_script: str) -> str:
    quoted_script = "'" + shell_script.replace("'", "'\"'\"'") + "'"
    return f"bash -o pipefail -c {quoted_script}"


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
