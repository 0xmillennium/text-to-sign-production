"""Generic progress helpers for long-running operations."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sized
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Protocol, TypeVar, cast

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
    """Line-oriented progress reporter."""

    def report(self, message: str, **fields: object) -> None:
        """Emit one progress message."""


class _ProgressBar(Protocol):
    """Small tqdm-compatible progress-bar protocol."""

    def __enter__(self) -> _ProgressBar: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...

    def update(self, n: int = 1) -> object: ...

    def close(self) -> None: ...

    def refresh(self) -> object: ...

    def set_postfix(self, ordered_dict: object | None = None, **kwargs: object) -> None: ...

    def set_postfix_str(self, s: str = "", refresh: bool = True) -> None: ...

    total: float | None


class _ByteReader(Protocol):
    """Binary source protocol."""

    def read(self, size: int = -1, /) -> bytes: ...


class _ByteWriter(Protocol):
    """Binary sink protocol."""

    def write(self, data: bytes, /) -> object: ...


_ByteSink = _ByteWriter | Callable[[bytes], object]


@dataclass(slots=True)
class NoOpProgressReporter:
    """Reporter that intentionally emits nothing."""

    def report(self, message: str, **fields: object) -> None:
        """Ignore a progress event."""
        return


@dataclass(slots=True)
class StdoutProgressReporter:
    """Stdout reporter with optional log mirroring."""

    prefix: str = ""
    log_path: Path | None = None
    flush: bool = True

    def report(self, message: str, **fields: object) -> None:
        line = _render_progress_line(self.prefix, message, fields)
        print(line, flush=self.flush)
        _append_log_line(self.log_path, line)


@dataclass(slots=True)
class TqdmProgressReporter:
    """tqdm-safe reporter with optional log mirroring."""

    prefix: str = ""
    log_path: Path | None = None

    def report(self, message: str, **fields: object) -> None:
        line = _render_progress_line(self.prefix, message, fields)
        _tqdm.write(line, file=sys.stdout)
        sys.stdout.flush()
        _append_log_line(self.log_path, line)


@dataclass(slots=True)
class ItemProgress:
    """Live progress helper."""

    label: str
    total: int | None = None
    unit: str = "items"
    reporter: ProgressReporter = field(default_factory=StdoutProgressReporter)
    emit_summary: bool = False

    completed: int = field(init=False, default=0)
    start_time: float = field(init=False, default_factory=time.perf_counter)
    _bar: _ProgressBar | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._bar = progress_bar(total=self.total, desc=self.label, unit=self.unit)

    def __enter__(self) -> ItemProgress:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            if self.emit_summary:
                self.finish()
            else:
                self.close()
        else:
            self.close()

    def advance(self, count: int = 1, **fields: object) -> None:
        """Advance the live progress bar."""

        self.completed += count
        if self._bar is None:
            return

        self._bar.update(count)
        postfix = _format_postfix_fields(fields)
        if postfix:
            self._bar.set_postfix(**postfix)

    def note(self, text: str) -> None:
        """Set a human-readable progress note without key=value formatting."""

        if self._bar is not None:
            self._bar.set_postfix_str(text)

    def add_total(self, count: int) -> None:
        """Increase the live progress total when later work is discovered."""

        if count <= 0 or self._bar is None:
            return
        self.total = (self.total or 0) + count
        self._bar.total = self.total
        self._bar.refresh()

    def finish(self, **fields: object) -> None:
        """Close the live bar and optionally emit a final summary line."""

        self.close()
        if not self.emit_summary:
            return

        elapsed = time.perf_counter() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else None
        total_display = "?" if self.total is None else str(self.total)

        self.reporter.report(
            f"{self.label} complete {self.completed}/{total_display}",
            elapsed=_format_seconds(elapsed),
            **{f"{self.unit}/s": _format_number(rate)},
            **fields,
        )

    def close(self) -> None:
        """Close the underlying progress bar without emitting a summary."""

        if self._bar is None:
            return
        self._bar.close()
        self._bar = None


@dataclass(slots=True)
class BatchProgress(ItemProgress):
    """Progress helper specialized for batch-oriented work."""

    unit: str = "batches"


@dataclass(slots=True)
class FrameProgress(ItemProgress):
    """Progress helper specialized for frame-oriented work."""

    unit: str = "frames"


def iter_with_progress(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str = "",
    unit: str = "items",
    reporter: ProgressReporter | None = None,
) -> Iterator[T]:
    """Yield items from an iterable while showing live progress."""

    resolved_total = total if total is not None else _known_total(iterable)
    progress = ItemProgress(
        label=desc or "progress",
        total=resolved_total,
        unit=unit,
        reporter=reporter or StdoutProgressReporter(),
    )
    completed = False
    try:
        for item in iterable:
            yield item
            progress.advance()
        completed = True
    finally:
        if completed:
            progress.close()
        else:
            progress.close()


def progress_bar(
    *,
    total: int | None = None,
    desc: str = "",
    unit: str = "items",
) -> _ProgressBar:
    """Return a standard tqdm progress bar for manual updates."""

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
    *,
    desc: str = "",
    reporter: ProgressReporter | None = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> int:
    """Stream bytes from a source to a writer with live byte progress."""

    progress = ItemProgress(
        label=desc or "stream",
        total=total_bytes,
        unit="B",
        reporter=reporter or StdoutProgressReporter(),
    )
    completed = False
    try:
        while chunk := src.read(chunk_size):
            _write_chunk(writer, chunk)
            progress.advance(len(chunk))
        completed = True
    finally:
        if completed:
            progress.close()
        else:
            progress.close()
    return progress.completed


def _write_chunk(writer: _ByteSink, chunk: bytes) -> None:
    if callable(writer):
        writer(chunk)
        return
    writer.write(chunk)


def _known_total(iterable: Iterable[object]) -> int | None:
    if isinstance(iterable, Sized):
        return len(iterable)
    return None


def _append_log_line(log_path: Path | None, line: str) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


def _render_progress_line(prefix: str, message: str, fields: dict[str, object]) -> str:
    parts: list[str] = []
    if prefix:
        parts.append(prefix)
    parts.append(message)
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={_format_field_text(value)}")
    return " ".join(parts)


def _format_postfix_fields(fields: dict[str, object]) -> dict[str, str]:
    formatted: dict[str, str] = {}
    for key, value in fields.items():
        if value is None:
            continue
        formatted[key] = _format_field_text(value)
    return formatted


def _format_seconds(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, rem_seconds = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{int(rem_seconds):02d}s"
    hours, rem_minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h{int(rem_minutes):02d}m"


def _format_number(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:.3g}"


def _format_field_text(value: object) -> str:
    if isinstance(value, float):
        return _format_number(value) or "nan"
    return str(value)
