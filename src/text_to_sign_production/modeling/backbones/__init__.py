"""Text backbone interfaces and lazy M0 backbone exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import TextBackbone, TextBackboneOutput

if TYPE_CHECKING:
    from .flan_t5 import DEFAULT_FLAN_T5_MODEL_NAME, FlanT5TextBackbone

__all__ = [
    "DEFAULT_FLAN_T5_MODEL_NAME",
    "FlanT5TextBackbone",
    "TextBackbone",
    "TextBackboneOutput",
]


def __getattr__(name: str) -> Any:
    if name == "DEFAULT_FLAN_T5_MODEL_NAME":
        from .flan_t5 import DEFAULT_FLAN_T5_MODEL_NAME

        return DEFAULT_FLAN_T5_MODEL_NAME
    if name == "FlanT5TextBackbone":
        from .flan_t5 import FlanT5TextBackbone

        return FlanT5TextBackbone
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
