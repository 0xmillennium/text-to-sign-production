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


class CoordinateSpace(enum.StrEnum):
    """Canonical coordinate-space identity for prepared pose truth."""

    NORMALIZED_IMAGE = "normalized_image"


class SourceIssueCode(enum.StrEnum):
    """Canonical source-side issue codes carried by prepared samples."""

    MISSING_VIDEO_SOURCE = "missing_video_source"
    MISSING_KEYPOINT_SOURCE = "missing_keypoint_source"
    VIDEO_METADATA_NOT_PROVIDED = "video_metadata_not_provided"
    VIDEO_METADATA_UNREADABLE = "video_metadata_unreadable"
    MISSING_KEYPOINT_DIRECTORY = "missing_keypoint_directory"
    MISSING_FRAME_JSON_FILES = "missing_frame_json_files"


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
    "CoordinateSpace",
    "SourceIssueCode",
    "TierMembership",
    "TierName",
    "VALID_SAMPLE_SPLITS",
    "VALID_SAMPLE_STATUSES",
    "VALID_TIER_MEMBERSHIPS",
    "VALID_TIER_NAMES",
]
