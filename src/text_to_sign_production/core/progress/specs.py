"""Typed progress stage specifications and renderer-neutral events."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

ProgressSplitBehavior = Literal["per_split", "global"]
ProgressEventKind = Literal["advance", "status", "finish"]


@dataclass(frozen=True, slots=True)
class ProgressStageSpec:
    """A typed description of one atomic workflow progress stage."""

    workflow_id: str
    stage_id: str
    label: str
    unit: str
    owner_module: str
    split_behavior: ProgressSplitBehavior
    operation_kind: str
    total_semantics: str
    bar_eligible: bool
    allowed_counters: tuple[str, ...] = ()
    artifact_role: str | None = None
    command_kind: str | None = None
    default_total: int | None = None

    def __post_init__(self) -> None:
        _require_non_empty("workflow_id", self.workflow_id)
        _require_non_empty("stage_id", self.stage_id)
        _require_non_empty("label", self.label)
        _require_non_empty("unit", self.unit)
        _require_non_empty("owner_module", self.owner_module)
        _require_non_empty("operation_kind", self.operation_kind)
        _require_non_empty("total_semantics", self.total_semantics)
        object.__setattr__(
            self,
            "allowed_counters",
            tuple(_validated_counter(counter) for counter in self.allowed_counters),
        )
        if self.default_total is not None and self.default_total < 0:
            raise ValueError("default_total must be >= 0")


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """A renderer-neutral progress update."""

    stage_id: str
    kind: ProgressEventKind = "advance"
    count: int = 1
    counters: Mapping[str, object] = field(default_factory=dict)
    message: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty("stage_id", self.stage_id)
        if self.count < 0:
            raise ValueError("count must be >= 0")
        object.__setattr__(self, "counters", dict(self.counters))


def _require_non_empty(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validated_counter(counter: str) -> str:
    _require_non_empty("counter", counter)
    return counter.strip()


__all__ = [
    "ProgressEvent",
    "ProgressSplitBehavior",
    "ProgressStageSpec",
]
