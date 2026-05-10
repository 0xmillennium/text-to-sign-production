"""Dropped-side debug materialization lifecycle support."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class DroppedDebugMaterializationOutcome(enum.StrEnum):
    """Attempt outcome for optional dropped-sample debug payload materialization."""

    NOT_ATTEMPTED = "not_attempted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DroppedMaterializationLifecycle:
    """Lifecycle facts for optional dropped-sample debug payload materialization."""

    debug_materialization_eligible: bool
    debug_materialization_attempted: bool
    debug_materialization_outcome: DroppedDebugMaterializationOutcome
    payload_path: str | None = None
    payload_exists: bool = False
    archive_publishable: bool = False
    failure_reason: str | None = None
    not_attempted_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "debug_materialization_outcome",
            DroppedDebugMaterializationOutcome(self.debug_materialization_outcome),
        )


def build_dropped_materialization_lifecycle(
    *,
    debug_materialization_eligible: bool,
    debug_materialization_attempted: bool,
    debug_materialization_outcome: DroppedDebugMaterializationOutcome | str,
    payload_path: str | None = None,
    payload_exists: bool = False,
    archive_publishable: bool = False,
    failure_reason: str | None = None,
    not_attempted_reason: str | None = None,
) -> DroppedMaterializationLifecycle:
    """Build explicit lifecycle facts for optional dropped debug payloads."""
    lifecycle = DroppedMaterializationLifecycle(
        debug_materialization_eligible=debug_materialization_eligible,
        debug_materialization_attempted=debug_materialization_attempted,
        debug_materialization_outcome=DroppedDebugMaterializationOutcome(
            debug_materialization_outcome
        ),
        payload_path=payload_path,
        payload_exists=payload_exists,
        archive_publishable=archive_publishable,
        failure_reason=failure_reason,
        not_attempted_reason=not_attempted_reason,
    )
    _validate_lifecycle(lifecycle)
    return lifecycle


def _validate_lifecycle(lifecycle: DroppedMaterializationLifecycle) -> None:
    outcome = lifecycle.debug_materialization_outcome
    if not lifecycle.debug_materialization_eligible:
        if lifecycle.debug_materialization_attempted:
            raise ValueError("Ineligible dropped materialization must not be attempted.")
        if outcome is not DroppedDebugMaterializationOutcome.NOT_ATTEMPTED:
            raise ValueError("Ineligible dropped materialization must be not_attempted.")
    if lifecycle.debug_materialization_attempted:
        if outcome is DroppedDebugMaterializationOutcome.NOT_ATTEMPTED:
            raise ValueError("Attempted dropped materialization must have a terminal outcome.")
    elif outcome is not DroppedDebugMaterializationOutcome.NOT_ATTEMPTED:
        raise ValueError("Unattempted dropped materialization must be not_attempted.")
    if lifecycle.payload_exists and not lifecycle.payload_path:
        raise ValueError("payload_exists requires payload_path.")
    if lifecycle.archive_publishable and not lifecycle.payload_exists:
        raise ValueError("archive_publishable requires payload_exists.")
    if outcome is DroppedDebugMaterializationOutcome.SUCCEEDED and (
        not lifecycle.payload_exists or not lifecycle.archive_publishable
    ):
        raise ValueError("Succeeded dropped materialization requires publishable payload.")
    if outcome is DroppedDebugMaterializationOutcome.FAILED and not lifecycle.failure_reason:
        raise ValueError("Failed dropped materialization requires failure_reason.")


__all__ = [
    "DroppedDebugMaterializationOutcome",
    "DroppedMaterializationLifecycle",
    "build_dropped_materialization_lifecycle",
]
