"""Shared tqdm-based progress helpers for long-running project operations."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Iterator
from types import TracebackType
from typing import Protocol, TypeVar, cast

from tqdm.auto import tqdm as _tqdm  # type: ignore[import-untyped]

T = TypeVar("T")
_DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024
__all__ = ["iter_with_progress", "progress_bar", "stream_file_with_progress"]


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


_ByteSink = _ByteWriter | Callable[[bytes], object]


def iter_with_progress(
    iterable: Iterable[T],
    total: int | None = None,
    desc: str = "",
    unit: str = "items",
) -> Iterator[T]:
    """Yield an iterable through the project-standard progress bar."""

    yield from cast(
        Iterable[T],
        _tqdm(
            iterable,
            total=total,
            desc=desc,
            unit=unit,
            file=sys.stdout,
            dynamic_ncols=True,
        ),
    )


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
