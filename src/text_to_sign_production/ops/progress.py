"""Shared progress reporting helpers for long-running operational tasks."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import TextIO


def format_bytes(num_bytes: int) -> str:
    """Format a byte count with binary units."""

    size = float(num_bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


@dataclass(slots=True)
class ProgressReporter:
    """Render a single-line progress indicator suitable for notebooks and terminals."""

    label: str
    total_bytes: int | None = None
    stream: TextIO = field(default_factory=lambda: sys.stdout, repr=False)
    min_interval_seconds: float = 0.2
    width: int = 24
    use_carriage_return: bool | None = None
    _last_render_time: float = field(default=0.0, init=False, repr=False)
    _last_completed: int = field(default=0, init=False, repr=False)
    _last_render_length: int = field(default=0, init=False, repr=False)
    _last_rendered_line: str = field(default="", init=False, repr=False)
    _use_carriage_return: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Choose terminal-style updates only for streams that advertise TTY behavior."""

        if self.use_carriage_return is not None:
            self._use_carriage_return = self.use_carriage_return
            return
        progress_mode = os.environ.get("T2SP_PROGRESS_MODE", "").strip().lower()
        if progress_mode in {"line", "lines", "newline", "newlines", "colab"}:
            self._use_carriage_return = False
            return
        if progress_mode in {"carriage_return", "cr", "tty"}:
            self._use_carriage_return = True
            return
        self._use_carriage_return = self.stream.isatty()

    def update(self, completed_bytes: int) -> None:
        """Render progress if enough time has passed or the operation finished."""

        now = time.monotonic()
        total = self.total_bytes
        should_render = (
            completed_bytes == 0
            or (total is not None and completed_bytes >= total)
            or now - self._last_render_time >= self.min_interval_seconds
            or completed_bytes < self._last_completed
        )
        if not should_render:
            return

        self._write_line(completed_bytes, end_line=False)

    def finish(self, completed_bytes: int | None = None) -> None:
        """Force a final render and terminate the progress line."""

        final_completed = self._last_completed if completed_bytes is None else completed_bytes
        final_line = self._render_line(final_completed)
        if self._use_carriage_return or final_line != self._last_rendered_line:
            self._write_line(final_completed, end_line=True)
            return
        self.stream.flush()

    def _write_line(self, completed_bytes: int, *, end_line: bool) -> None:
        line = self._render_line(completed_bytes)
        self._last_render_time = time.monotonic()
        self._last_completed = completed_bytes
        if self._use_carriage_return:
            padding = " " * max(0, self._last_render_length - len(line))
            self.stream.write(f"\r{line}{padding}")
            if end_line:
                self.stream.write("\n")
        else:
            self.stream.write(f"{line}\n")
        self._last_render_length = len(line)
        self._last_rendered_line = line
        self.stream.flush()

    def _render_line(self, completed_bytes: int) -> str:
        total = self.total_bytes
        if total is None or total <= 0:
            return f"{self.label}: {format_bytes(completed_bytes)} transferred"

        ratio = min(max(completed_bytes / total, 0.0), 1.0)
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        return (
            f"{self.label}: [{bar}] {ratio * 100:5.1f}% "
            f"({format_bytes(completed_bytes)} / {format_bytes(total)})"
        )
