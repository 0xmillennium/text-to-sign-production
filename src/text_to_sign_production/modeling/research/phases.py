"""Research phase identifiers for the modeling program."""

from __future__ import annotations

import enum


class ResearchPhase(enum.StrEnum):
    """Phase identifiers used by the Phase 4-9 modeling program."""

    PHASE_4 = "phase_4"
    PHASE_5 = "phase_5"
    PHASE_6 = "phase_6"
    PHASE_7 = "phase_7"
    PHASE_8 = "phase_8"
    PHASE_9 = "phase_9"


def phase_number(phase: ResearchPhase | str) -> int:
    """Return the integer phase number for a research phase."""

    try:
        resolved = ResearchPhase(phase)
    except ValueError as exc:
        raise ValueError(f"unsupported research phase: {phase!r}") from exc
    return int(resolved.value.removeprefix("phase_"))


__all__ = [
    "ResearchPhase",
    "phase_number",
]
