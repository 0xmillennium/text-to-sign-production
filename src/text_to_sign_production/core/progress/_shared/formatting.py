"""Internal formatting helpers for progress sinks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def append_log_line(log_path: Path | None, line: str) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


def render_progress_line(prefix: str, message: str, fields: Mapping[str, object]) -> str:
    parts: list[str] = []
    if prefix:
        parts.append(prefix)
    parts.append(message)
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}: {_format_field_text(value)}")
    return " ".join(parts)


def _format_field_text(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


__all__ = ["append_log_line", "render_progress_line"]
