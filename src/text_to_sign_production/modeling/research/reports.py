"""Required-report contracts for modeling research specifications."""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sign_production.modeling.research.validation import (
    ModelingResearchSpecError,
    require_non_empty,
    require_unique_strings,
)


@dataclass(frozen=True, slots=True)
class RequiredReport:
    """A report surface required by a modeling research artifact."""

    identifier: str
    title: str
    required_sections: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_non_empty(self.identifier, field_name="identifier")
        require_non_empty(self.title, field_name="title")
        if not isinstance(self.required_sections, tuple):
            raise ModelingResearchSpecError("required_sections must be a tuple.")
        require_unique_strings(self.required_sections, field_name="required_sections")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable required-report record."""

        return {
            "identifier": self.identifier,
            "title": self.title,
            "required_sections": list(self.required_sections),
        }


__all__ = [
    "RequiredReport",
]
