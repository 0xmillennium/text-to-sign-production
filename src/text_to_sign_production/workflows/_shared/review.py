"""Shared workflow review summary data transfer objects."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkflowReviewField:
    label: str
    value: object


@dataclass(frozen=True, slots=True)
class WorkflowReviewItem:
    label: str
    fields: tuple[WorkflowReviewField, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))


@dataclass(frozen=True, slots=True)
class WorkflowReviewSection:
    title: str
    items: tuple[WorkflowReviewItem, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))


__all__ = [
    "WorkflowReviewField",
    "WorkflowReviewItem",
    "WorkflowReviewSection",
]
