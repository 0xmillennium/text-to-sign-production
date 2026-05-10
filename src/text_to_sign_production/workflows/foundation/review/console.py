from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from text_to_sign_production.workflows.foundation.review.contracts import (
    WorkflowReviewSection,
)


def display_review_sections(
    sections: Iterable[WorkflowReviewSection],
    *,
    leading_blank: bool = False,
) -> None:
    section_values = tuple(sections)
    if not section_values:
        _display_block_title("Review", leading_blank=leading_blank)
        print("- none")
        return

    for index, section in enumerate(section_values):
        _display_block_title(section.title, leading_blank=leading_blank or index > 0)
        if not section.items:
            print("- none")
            continue
        for item_index, item in enumerate(section.items, start=1):
            print(f"{item_index}. {_render_console_text(item.label)}")
            for field in item.fields:
                print(
                    f"   {_render_console_text(field.label)}: {_render_console_text(field.value)}"
                )


def _display_block_title(title: object, *, leading_blank: bool = False) -> None:
    rendered_title = _render_console_text(title)
    if leading_blank:
        print()
    print(rendered_title)
    print("-" * len(rendered_title))


def _render_console_text(value: Any) -> str:
    if value is None:
        return "none"
    return str(value)
