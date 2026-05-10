from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from text_to_sign_production.workflows.foundation.review.contracts import (
    WorkflowReviewField,
    WorkflowReviewItem,
    WorkflowReviewSection,
)


def jsonable(value: object) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): jsonable(mapping_value) for key, mapping_value in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "value"):
        return jsonable(value.value)
    return value


def markdown_value(value: object) -> str:
    return _escape_markdown_text(_markdown_value_text(value))


def markdown_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
) -> str:
    if not headers:
        raise ValueError("headers must not be empty")

    row_values = tuple(tuple(row) for row in rows)
    header_count = len(headers)
    for row in row_values:
        if len(row) != header_count:
            raise ValueError("row length must match header length")
    if not row_values:
        return "No records."

    rendered_headers = " | ".join(_escape_markdown_text(header) for header in headers)
    divider = " | ".join("---" for _ in headers)
    rendered_rows = tuple(" | ".join(markdown_value(value) for value in row) for row in row_values)
    return "\n".join(
        (
            f"| {rendered_headers} |",
            f"| {divider} |",
            *(f"| {row} |" for row in rendered_rows),
        )
    )


def render_review_field_lines(
    fields: Iterable[WorkflowReviewField],
) -> tuple[str, ...]:
    return tuple(
        f"{_escape_markdown_text(field.label)}: {markdown_value(field.value)}" for field in fields
    )


def render_review_item_markdown(
    item: WorkflowReviewItem,
) -> str:
    if not item.fields:
        return f"- {_escape_markdown_text(item.label)}"
    field_lines = tuple(f"  - {line}" for line in render_review_field_lines(item.fields))
    return _flatten_markdown_iterable((f"- {_escape_markdown_text(item.label)}", *field_lines))


def render_review_section_markdown(
    section: WorkflowReviewSection,
    *,
    heading_level: int = 2,
) -> str:
    if heading_level < 1:
        raise ValueError("heading_level must be >= 1")
    heading = _markdown_heading(section.title, heading_level)
    if not section.items:
        return _flatten_markdown_iterable((heading, "- none"))
    return _flatten_markdown_iterable(
        (heading, *(render_review_item_markdown(item) for item in section.items))
    )


def render_review_sections_markdown(
    sections: Iterable[WorkflowReviewSection],
    *,
    heading_level: int = 2,
) -> str:
    rendered_sections = tuple(
        render_review_section_markdown(section, heading_level=heading_level) for section in sections
    )
    if not rendered_sections:
        return f"{_markdown_heading('Review', heading_level)}\n\n- none"
    return "\n\n".join(rendered_sections)


def _markdown_heading(title: str, level: int) -> str:
    if level < 1:
        raise ValueError("heading level must be >= 1")
    return f"{'#' * level} {_escape_markdown_text(title)}"


def _escape_markdown_text(text: str) -> str:
    return text.replace("\r", " ").replace("\n", " ").replace("|", "\\|")


def _flatten_markdown_iterable(items: Iterable[str]) -> str:
    return "\n".join(items)


def _markdown_value_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, Mapping):
        return ", ".join(
            f"{_markdown_value_text(key)}: {_markdown_value_text(mapping_value)}"
            for key, mapping_value in value.items()
        )
    if isinstance(value, (list, tuple)):
        return "; ".join(_markdown_value_text(item) for item in value)
    try:
        return str(jsonable(value))
    except TypeError:
        return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True)
