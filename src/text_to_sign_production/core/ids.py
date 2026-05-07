"""Repository-wide identity types."""

from __future__ import annotations

import enum
from typing import Final


class SampleSplit(enum.StrEnum):
    """Canonical split identity for sample-local and cross-split facts."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class SampleStatus(enum.StrEnum):
    """Canonical physical sample storage status."""

    PASSED = "passed"
    DROPPED = "dropped"


class TierName(enum.StrEnum):
    """Canonical physical tier manifest identity."""

    LOOSE = "loose"
    CLEAN = "clean"
    TIGHT = "tight"


class TierMembership(enum.StrEnum):
    """Canonical physical tier membership identity."""

    INCLUDED = "included"
    EXCLUDED = "excluded"


VALID_SAMPLE_SPLITS: Final[tuple[str, ...]] = tuple(split.value for split in SampleSplit)
VALID_SAMPLE_STATUSES: Final[tuple[str, ...]] = tuple(status.value for status in SampleStatus)
VALID_TIER_NAMES: Final[tuple[str, ...]] = tuple(tier.value for tier in TierName)
VALID_TIER_MEMBERSHIPS: Final[tuple[str, ...]] = tuple(
    membership.value for membership in TierMembership
)


__all__ = [
    "SampleSplit",
    "SampleStatus",
    "TierMembership",
    "TierName",
    "VALID_SAMPLE_SPLITS",
    "VALID_SAMPLE_STATUSES",
    "VALID_TIER_MEMBERSHIPS",
    "VALID_TIER_NAMES",
]
