"""Progress sinks and renderers."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Protocol, cast

from tqdm.auto import tqdm as _tqdm  # type: ignore[import-untyped]

from text_to_sign_production.core.progress._shared.formatting import (
    append_log_line,
    render_progress_line,
)
from text_to_sign_production.core.progress.session import ProgressTaskHandle
from text_to_sign_production.core.progress.specs import ProgressStageSpec


class _ProgressBar(Protocol):
    def update(self, n: int = 1) -> object: ...

    def close(self) -> None: ...

    def set_postfix(self, ordered_dict: object | None = None, **kwargs: object) -> None: ...


@dataclass(slots=True)
class TqdmProgressSink:
    """tqdm-backed sink for Python-owned workflow progress stages."""

    prefix: str = ""
    log_path: Path | None = None

    def open_task(
        self,
        spec: ProgressStageSpec,
        *,
        total: int | None = None,
    ) -> ProgressTaskHandle:
        return _TqdmProgressTaskHandle(spec=spec, total=total)

    def status(self, message: str, **fields: object) -> None:
        line = render_progress_line(self.prefix, message, fields)
        _tqdm.write(line, file=sys.stdout)
        sys.stdout.flush()
        append_log_line(self.log_path, line)


@dataclass(slots=True)
class _TqdmProgressTaskHandle:
    spec: ProgressStageSpec
    total: int | None
    completed: int = field(init=False, default=0)
    _bar: _ProgressBar | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._bar = _progress_bar(total=self.total, desc=self.spec.label, unit=self.spec.unit)

    def __enter__(self) -> _TqdmProgressTaskHandle:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def advance(
        self,
        count: int = 1,
        *,
        counters: Mapping[str, object] | None = None,
    ) -> None:
        if count < 0:
            raise ValueError("count must be >= 0")
        self.completed += count
        if self._bar is None:
            return
        if count:
            self._bar.update(count)
        postfix = _allowed_counter_postfix(self.spec, counters or {})
        if postfix:
            self._bar.set_postfix(postfix)

    def status(self, message: str, **fields: object) -> None:
        return

    def close(self) -> None:
        if self._bar is None:
            return
        self._bar.close()
        self._bar = None


def _progress_bar(*, total: int | None, desc: str, unit: str) -> _ProgressBar:
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


def _allowed_counter_postfix(
    spec: ProgressStageSpec,
    counters: Mapping[str, object],
) -> dict[str, str]:
    allowed = set(spec.allowed_counters)
    formatted: dict[str, str] = {}
    for key, value in counters.items():
        if key not in allowed or value is None:
            continue
        formatted[key] = str(value)
    return formatted


__all__ = ["TqdmProgressSink"]
