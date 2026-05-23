"""Risk-control contracts for modeling research specifications."""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sign_production.modeling.research.validation import (
    ModelingResearchSpecError,
    require_non_empty,
    require_unique_strings,
)


@dataclass(frozen=True, slots=True)
class RiskControl:
    """A risk control attached to a modeling research artifact."""

    identifier: str
    statement: str
    enforced_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_non_empty(self.identifier, field_name="identifier")
        require_non_empty(self.statement, field_name="statement")
        if not isinstance(self.enforced_by, tuple):
            raise ModelingResearchSpecError("enforced_by must be a tuple.")
        require_unique_strings(self.enforced_by, field_name="enforced_by")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable risk-control record."""

        return {
            "identifier": self.identifier,
            "statement": self.statement,
            "enforced_by": list(self.enforced_by),
        }


__all__ = [
    "RiskControl",
]
