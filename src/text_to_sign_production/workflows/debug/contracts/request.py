from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from text_to_sign_production.core.ids import SampleSplit


class DebugWorkflowRequestError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DebugSampleRequest:
    debug_splits: tuple[SampleSplit, ...]
    target_sentence_name: str

    def __init__(
        self,
        debug_splits: Iterable[SampleSplit | str] | SampleSplit | str,
        target_sentence_name: str,
    ) -> None:
        object.__setattr__(self, "debug_splits", normalize_debug_splits(debug_splits))
        object.__setattr__(
            self,
            "target_sentence_name",
            _coerce_target_sentence_name(target_sentence_name),
        )


def normalize_debug_splits(
    value: Iterable[SampleSplit | str] | SampleSplit | str,
) -> tuple[SampleSplit, ...]:
    if isinstance(value, (SampleSplit, str)):
        raw_values = (value,)
    elif isinstance(value, Iterable):
        raw_values = tuple(value)
    else:
        raise DebugWorkflowRequestError("DEBUG_SPLITS must be an iterable of split values.")
    if not raw_values:
        raise DebugWorkflowRequestError("DEBUG_SPLITS must not be empty.")
    splits: list[SampleSplit] = []
    for raw in raw_values:
        try:
            split = raw if isinstance(raw, SampleSplit) else SampleSplit(str(raw).strip())
        except ValueError as exc:
            allowed = ", ".join(split.value for split in SampleSplit)
            raise DebugWorkflowRequestError(
                f"DEBUG_SPLITS values must be one of: {allowed}."
            ) from exc
        splits.append(split)
    if len(set(splits)) != len(splits):
        raise DebugWorkflowRequestError("DEBUG_SPLITS must not contain duplicate values.")
    return tuple(splits)


def _coerce_target_sentence_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DebugWorkflowRequestError("TARGET_SENTENCE_NAME must be non-empty.")
    return value.strip()


__all__ = ["DebugSampleRequest", "DebugWorkflowRequestError", "normalize_debug_splits"]
