"""Shared data-layer identity types."""

from __future__ import annotations

import enum
from typing import Final


class SampleSplit(enum.StrEnum):
    """Canonical split identity for sample-local and cross-split data facts."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


VALID_SAMPLE_SPLITS: Final[tuple[str, ...]] = tuple(split.value for split in SampleSplit)
