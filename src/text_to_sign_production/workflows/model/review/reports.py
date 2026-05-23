"""Rendering helper for model workflow review output."""

from __future__ import annotations

from collections.abc import Iterable

from text_to_sign_production.workflows.foundation.review import (
    WorkflowReviewSection,
    render_review_sections_markdown,
)


def render_model_review_markdown(
    sections: Iterable[WorkflowReviewSection],
) -> str:
    """Render operator-facing model sections without writing provider content."""

    return render_review_sections_markdown(sections)


__all__ = ["render_model_review_markdown"]
