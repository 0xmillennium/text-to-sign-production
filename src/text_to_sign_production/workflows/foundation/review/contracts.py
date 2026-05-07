from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkflowReviewField:
    label: str
    value: object

    def __post_init__(self) -> None:
        _validate_non_empty_label("label", self.label)
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


def review_field(label: object, value: object) -> WorkflowReviewField:
    return WorkflowReviewField(label=_render_review_label(label), value=value)


def review_item(
    label: object,
    fields: Iterable[tuple[object, object]] = (),
) -> WorkflowReviewItem:
    return WorkflowReviewItem(
        label=_render_review_label(label),
        fields=tuple(review_field(field_label, value) for field_label, value in fields),
    )


def review_section(
    title: object,
    items: Iterable[WorkflowReviewItem] = (),
) -> WorkflowReviewSection:
    return WorkflowReviewSection(
        title=_render_review_label(title),
        items=tuple(items),
    )


def review_lines_section(
    title: object,
    lines: Iterable[object],
) -> WorkflowReviewSection:
    return review_section(
        title,
        tuple(WorkflowReviewItem(label=_render_review_label(line), fields=()) for line in lines),
    )


def review_mapping_section(
    title: object,
    mapping: Mapping[object, object],
    *,
    value_label: object = "value",
) -> WorkflowReviewSection:
    field_label = _render_review_label(value_label)
    return review_section(
        title,
        tuple(review_item(key, ((field_label, value),)) for key, value in mapping.items()),
    )


def review_nested_mapping_section(
    title: object,
    mapping: Mapping[object, Mapping[object, object]],
) -> WorkflowReviewSection:
    return review_section(
        title,
        tuple(review_item(key, nested_mapping.items()) for key, nested_mapping in mapping.items()),
    )


def _render_review_label(value: Any) -> str:
    if value is None:
        label = "none"
    elif hasattr(value, "value"):
        label = str(value.value)
    else:
        label = str(value)
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
