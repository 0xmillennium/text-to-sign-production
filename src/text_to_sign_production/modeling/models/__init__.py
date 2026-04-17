"""Lazy exports for Sprint 3 baseline model components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .baseline import BaselinePoseOutput, BaselineTextToPoseModel
    from .decoder import SimplePoseDecoder
    from .heads import PoseChannelHead

__all__ = [
    "BaselinePoseOutput",
    "BaselineTextToPoseModel",
    "PoseChannelHead",
    "SimplePoseDecoder",
]


def __getattr__(name: str) -> Any:
    if name in {"BaselinePoseOutput", "BaselineTextToPoseModel"}:
        from .baseline import BaselinePoseOutput, BaselineTextToPoseModel

        return {
            "BaselinePoseOutput": BaselinePoseOutput,
            "BaselineTextToPoseModel": BaselineTextToPoseModel,
        }[name]
    if name == "PoseChannelHead":
        from .heads import PoseChannelHead

        return PoseChannelHead
    if name == "SimplePoseDecoder":
        from .decoder import SimplePoseDecoder

        return SimplePoseDecoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
