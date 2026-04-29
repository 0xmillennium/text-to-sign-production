"""Shared progress helpers for long-running project operations.

The project uses live progress bars for long Python loops to provide compact,
readable feedback in notebooks and interactive sessions.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sized
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, TypeVar, cast

from tqdm.auto import tqdm as _tqdm  # type: ignore[import-untyped]

T = TypeVar("T")
_DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024
__all__ = [
    "BatchProgress",
    "FrameProgress",
    "ItemProgress",
    "NoOpProgressReporter",
    "ProgressReporter",
    "StdoutProgressReporter",
    "TqdmProgressReporter",
    "iter_with_progress",
    "progress_bar",
    "stream_file_with_progress",
]


class ProgressReporter(Protocol):
    """Minimal line-oriented progress reporter used across project layers."""

    def report(self, message: str, **fields: object) -> None:
        """Emit one progress message."""


class _ByteWriter(Protocol):
    """Minimal protocol for binary sinks used by streaming operations."""

    def write(self, data: bytes, /) -> object:
        """Write a binary chunk."""


class _ByteReader(Protocol):
    """Minimal protocol for binary sources used by streaming operations."""

    def read(self, size: int = -1, /) -> bytes:
        """Read a binary chunk."""


class _ProgressBar(Protocol):
    """Small context-manager protocol exposed by the project progress layer."""

    def __enter__(self) -> _ProgressBar:
        """Enter the progress-bar context."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """Exit the progress-bar context."""

    def update(self, n: int = 1) -> object:
        """Advance progress by ``n`` units."""

    def close(self) -> None:
        """Close the progress bar."""

    def set_postfix(self, **kwargs: object) -> None:
        """Add custom fields to the progress bar."""


_ByteSink = _ByteWriter | Callable[[bytes], object]


@dataclass(slots=True)
class NoOpProgressReporter:
    """Progress reporter that intentionally emits nothing."""

    def report(self, message: str, **fields: object) -> None:
        """Ignore a progress event."""


@dataclass(slots=True)
class StdoutProgressReporter:
    """Colab-safe stdout reporter with optional live-log persistence."""

    prefix: str = ""
    log_path: Path | None = None
    flush: bool = True

    def report(self, message: str, **fields: object) -> None:
        """Emit one progress line to stdout and the optional live log."""

        rendered = self._render(message, fields)
        print(rendered, flush=self.flush)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.write("\n")

    def _render(self, message: str, fields: dict[str, object]) -> str:
        parts: list[str] = []
        if self.prefix:
            parts.append(self.prefix)
        parts.append(message)
        for key, value in fields.items():
            if value is None:
                continue
            parts.append(f"{key}={_format_field_value(value)}")
        return " ".join(parts)


@dataclass(slots=True)
class TqdmProgressReporter:
    """Progress reporter that writes messages through tqdm-compatible stdout."""

    prefix: str = ""
    log_path: Path | None = None

    def report(self, message: str, **fields: object) -> None:
        """Emit one tqdm-safe message and mirror it to the optional live log."""

        rendered = StdoutProgressReporter(prefix=self.prefix)._render(message, fields)
        _tqdm.write(rendered, file=sys.stdout)
        sys.stdout.flush()
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.write("\n")


@dataclass(slots=True)
class ItemProgress:
    """Live progress helper with final summary field."""

    label: str
    total: int | None = None
    unit: str = "items"
    reporter: ProgressReporter = field(default_factory=StdoutProgressReporter)
    interval: int = 100
    start_time: float = field(default_factory=time.perf_counter)
    completed: int = 0
    _pbar: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._pbar = progress_bar(total=self.total, desc=self.label, unit=self.unit)

    def advance(self, count: int = 1, **fields: object) -> None:
        """Advance progress by ``count`` units."""

        self.completed += count
        if self._pbar is not None:
            self._pbar.update(count)
            if fields:
                if hasattr(self._pbar, "set_postfix"):
                    formatted = {k: _format_field_value(v) for k, v in fields.items()}
                    self._pbar.set_postfix(**formatted)

    def report(self, **fields: object) -> None:
        """Emit the current progress line (now skipped as handled by tqdm live bar)."""
        pass

    def finish(self, **fields: object) -> None:
        """Emit a final summary line."""

        if self._pbar is not None:
            if hasattr(self._pbar, "close"):
                self._pbar.close()
            self._pbar = None

        elapsed = time.perf_counter() - self.start_time
        throughput = self.completed / elapsed if elapsed > 0 else None
        total_display = "?" if self.total is None else str(self.total)
        self.reporter.report(
            f"{self.label} complete {self.completed}/{total_display}",
            elapsed=_format_seconds(elapsed),
            **{f"{self.unit}/s": _format_float(throughput)},
            **fields,
        )

    def _should_report(self) -> bool:
        return False


class BatchProgress(ItemProgress):
    """Progress helper for train/validation batch loops."""


class FrameProgress(ItemProgress):
    """Progress helper for video frame rendering loops."""


def iter_with_progress(
    iterable: Iterable[T],
    total: int | None = None,
    desc: str = "",
    unit: str = "items",
) -> Iterator[T]:
    """Yield an iterable with a live progress bar and final summary."""

    resolved_total = _known_total(iterable) if total is None else total
    progress = ItemProgress(
        label=desc or "progress",
        total=resolved_total,
        unit=unit,
        reporter=StdoutProgressReporter(),
        interval=100,
    )
    for item in iterable:
        yield item
        progress.advance()
    progress.finish()


def progress_bar(
    total: int | None = None,
    desc: str = "",
    unit: str = "items",
) -> _ProgressBar:
    """Return a project-standard progress bar for manual updates."""

    is_byte_progress = unit == "B"
    return cast(
        _ProgressBar,
        _tqdm(
            total=total,
            desc=desc,
            unit=unit,
            unit_scale=is_byte_progress,
            unit_divisor=1024 if is_byte_progress else 1000,
            file=sys.stdout,
            dynamic_ncols=True,
        ),
    )


def stream_file_with_progress(
    src: _ByteReader,
    writer: _ByteSink,
    total_bytes: int | None,
    desc: str = "",
) -> int:
    """Stream bytes from a source into a writer while updating byte progress."""

    completed = 0
    with progress_bar(total=total_bytes, desc=desc, unit="B") as bar:
        while chunk := src.read(_DEFAULT_CHUNK_SIZE):
            _write_chunk(writer, chunk)
            chunk_size = len(chunk)
            completed += chunk_size
            bar.update(chunk_size)
    return completed


def _write_chunk(writer: _ByteSink, chunk: bytes) -> None:
    if callable(writer):
        writer(chunk)
        return
    writer.write(chunk)


def _known_total(iterable: Iterable[object]) -> int | None:
    if isinstance(iterable, Sized):
        return len(iterable)
    return None


def _format_seconds(value: float | None) -> str | None:
    if value is None:
        return None
    if value < 60.0:
        return f"{value:.1f}s"
    minutes, seconds = divmod(value, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{int(seconds):02d}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h{int(minutes):02d}m"


def _format_float(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:.3g}"


def _format_field_value(value: Any) -> str:
    if isinstance(value, float):
        return _format_float(value) or "nan"
    return str(value)
