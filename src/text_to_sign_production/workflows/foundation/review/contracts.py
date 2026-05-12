from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

RenderableScalar: TypeAlias = str | int | float | bool | None | Path
RenderableValue: TypeAlias = RenderableScalar | tuple["RenderableValue", ...]


@dataclass(frozen=True, slots=True)
class WorkflowReviewField:
    label: str
    value: RenderableValue

    def __post_init__(self) -> None:
        _validate_non_empty_label("label", self.label)
        _validate_renderable_value(self.value)
        object.__setattr__(self, "label", self.label.strip())


@dataclass(frozen=True, slots=True)
class WorkflowReviewItem:
    label: str
    fields: tuple[WorkflowReviewField, ...]

    def __post_init__(self) -> None:
        _validate_non_empty_label("label", self.label)
        coerced_fields = _coerce_review_fields(self.fields)
        _ensure_unique_field_labels(coerced_fields)
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "fields", coerced_fields)


@dataclass(frozen=True, slots=True)
class WorkflowReviewSection:
    title: str
    items: tuple[WorkflowReviewItem, ...]

    def __post_init__(self) -> None:
        _validate_non_empty_label("title", self.title)
        object.__setattr__(self, "title", self.title.strip())
        object.__setattr__(self, "items", _coerce_review_items(self.items))


def review_field(label: str, value: RenderableValue) -> WorkflowReviewField:
    return WorkflowReviewField(label=_render_review_label(label), value=value)


def review_item(
    label: str,
    fields: Iterable[tuple[str, RenderableValue]] = (),
) -> WorkflowReviewItem:
    return WorkflowReviewItem(
        label=_render_review_label(label),
        fields=tuple(review_field(field_label, value) for field_label, value in fields),
    )


def review_section(
    title: str,
    items: Iterable[WorkflowReviewItem] = (),
) -> WorkflowReviewSection:
    return WorkflowReviewSection(
        title=_render_review_label(title),
        items=tuple(items),
    )


def review_lines_section(
    title: str,
    lines: Iterable[str],
) -> WorkflowReviewSection:
    return review_section(
        title,
        tuple(WorkflowReviewItem(label=_render_review_label(line), fields=()) for line in lines),
    )


def _render_review_label(value: str) -> str:
    label = value.strip()
    label = label.strip()
    if not label:
        raise ValueError("review labels must be non-empty")
    return label


def _validate_non_empty_label(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty label")


def _coerce_review_fields(
    fields: Iterable[WorkflowReviewField],
) -> tuple[WorkflowReviewField, ...]:
    coerced_fields = tuple(fields)
    for field in coerced_fields:
        if not isinstance(field, WorkflowReviewField):
            raise TypeError("fields must contain WorkflowReviewField instances")
    return coerced_fields


def _coerce_review_items(
    items: Iterable[WorkflowReviewItem],
) -> tuple[WorkflowReviewItem, ...]:
    coerced_items = tuple(items)
    for item in coerced_items:
        if not isinstance(item, WorkflowReviewItem):
            raise TypeError("items must contain WorkflowReviewItem instances")
    return coerced_items


def _ensure_unique_field_labels(fields: tuple[WorkflowReviewField, ...]) -> None:
    labels = [field.label for field in fields]
    if len(set(labels)) != len(labels):
        raise ValueError("review item field labels must be unique")


def _validate_renderable_value(value: RenderableValue) -> None:
    if isinstance(value, tuple):
        for item in value:
            _validate_renderable_value(item)
        return
    if value is None or isinstance(value, str | int | float | bool | Path):
        return
    raise TypeError(f"Unsupported review field value type: {type(value).__name__}")
