"""Domain-free shell string helpers for workflow command specs."""

from __future__ import annotations

import shlex
from pathlib import Path


def shell_quote(value: Path | str) -> str:
    """Return a shell-quoted path or string without executing anything."""

    return shlex.quote(str(value))


__all__ = ["shell_quote"]
