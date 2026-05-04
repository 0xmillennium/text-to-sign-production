"""Shell command specifications for notebook-driven workflow steps."""

import shlex
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CommandSpec:
    label: str
    bash: str
    arg: str
    failure: str


def shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def command(*, label: str, bash: str, failure: str) -> CommandSpec:
    if not isinstance(label, str):
        raise ValueError("label must be a string")
    if not isinstance(bash, str):
        raise ValueError("bash must be a string")
    if not isinstance(failure, str):
        raise ValueError("failure must be a string")

    stripped_label = label.strip()
    stripped_bash = bash.strip()
    stripped_failure = failure.strip()
    if not stripped_label:
        raise ValueError("label must not be empty")
    if not stripped_bash:
        raise ValueError("bash must not be empty")
    if not stripped_failure:
        raise ValueError("failure must not be empty")
    return CommandSpec(
        label=stripped_label,
        bash=stripped_bash,
        arg=shell_quote(stripped_bash),
        failure=stripped_failure,
    )
