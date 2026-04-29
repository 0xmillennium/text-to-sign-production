"""Descriptive shell command specs for notebook-facing workflows."""

from __future__ import annotations

import shlex
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """A visible shell command contract; this object never executes it."""

    label: str
    bash: str
    arg: str
    failure: str


def command(*, label: str, bash: str, failure: str) -> CommandSpec:
    """Build a shell command spec for explicit notebook execution."""

    return CommandSpec(
        label=label,
        bash=bash,
        arg=shlex.quote(bash),
        failure=failure,
    )


__all__ = ["CommandSpec", "command"]
