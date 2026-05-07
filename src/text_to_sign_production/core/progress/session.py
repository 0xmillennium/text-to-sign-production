"""Workflow-owned progress sessions and task handle contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import Protocol

from text_to_sign_production.core.progress.specs import ProgressStageSpec


class ProgressTaskHandle(Protocol):
    """A handle for one atomic progress stage."""

    @property
    def spec(self) -> ProgressStageSpec: ...

    def __enter__(self) -> ProgressTaskHandle: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def advance(
        self,
        count: int = 1,
        *,
        counters: Mapping[str, object] | None = None,
    ) -> None: ...

    def status(self, message: str, **fields: object) -> None: ...

    def close(self) -> None: ...


class ProgressSink(Protocol):
    """Renderer for workflow progress sessions."""

    def open_task(
        self,
        spec: ProgressStageSpec,
        *,
        total: int | None = None,
    ) -> ProgressTaskHandle: ...

    def status(self, message: str, **fields: object) -> None: ...


@dataclass(slots=True)
class ProgressSession:
    """Root workflow-owned progress session."""

    workflow_id: str
    sink: ProgressSink = field(default_factory=lambda: NoOpProgressSink())

    def task(
        self,
        spec: ProgressStageSpec,
        *,
        total: int | None = None,
    ) -> ProgressTaskHandle:
        if spec.workflow_id != self.workflow_id:
            raise ValueError(
                f"Stage {spec.stage_id!r} belongs to workflow {spec.workflow_id!r}, "
                f"not {self.workflow_id!r}."
            )
        if total is None:
            total = spec.default_total
        if not spec.bar_eligible:
            return NoOpProgressTaskHandle(spec=spec)
        return self.sink.open_task(spec, total=total)

    def status(self, message: str, **fields: object) -> None:
        self.sink.status(message, **fields)


@dataclass(slots=True)
class NoOpProgressTaskHandle:
    """Task handle that intentionally emits nothing."""

    spec: ProgressStageSpec

    def __enter__(self) -> NoOpProgressTaskHandle:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return

    def advance(
        self,
        count: int = 1,
        *,
        counters: Mapping[str, object] | None = None,
    ) -> None:
        return

    def status(self, message: str, **fields: object) -> None:
        return

    def close(self) -> None:
        return


@dataclass(slots=True)
class NoOpProgressSink:
    """Progress sink that intentionally emits nothing."""

    def open_task(
        self,
        spec: ProgressStageSpec,
        *,
        total: int | None = None,
    ) -> ProgressTaskHandle:
        return NoOpProgressTaskHandle(spec=spec)

    def status(self, message: str, **fields: object) -> None:
        return


__all__ = [
    "NoOpProgressSink",
    "ProgressSession",
    "ProgressSink",
    "ProgressTaskHandle",
]
