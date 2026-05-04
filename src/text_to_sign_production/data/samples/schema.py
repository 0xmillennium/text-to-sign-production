"""Canonical schema constants for processed sample contracts."""

from typing import Final

# The definitive schema version for processed samples and manifests.
PROCESSED_SCHEMA_VERSION: Final[str] = "v1"

VALID_SAMPLE_SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")

# All four channels are canonical and equal members of a BFH sample.
# No channel is optional at the processed-sample contract level.
CANONICAL_POSE_CHANNELS: Final[tuple[str, ...]] = (
    "body",
    "left_hand",
    "right_hand",
    "face",
)

# Per-channel joint counts defined centrally for validation.
POSE_CHANNEL_JOINT_COUNTS: Final[dict[str, int]] = {
    "body": 25,
    "left_hand": 21,
    "right_hand": 21,
    "face": 70,
}

# Canonical coordinate dimensionality for all channels.
POSE_COORDINATE_DIMENSIONS: Final[int] = 2

# Required keys inside each channel payload record (data-only, no schema metadata).
REQUIRED_POSE_CHANNEL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "coordinates",
        "confidence",
    }
)
REQUIRED_SELECTED_PERSON_KEYS: Final[frozenset[str]] = frozenset(
    {
        "index",
        "multi_person_frame_count",
        "max_people_per_frame",
    }
)
REQUIRED_FRAME_QUALITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "valid_frame_count",
        "invalid_frame_count",
        "face_missing_frame_count",
        "out_of_bounds_coordinate_count",
        "frames_with_any_zeroed_required_joint",
        "frame_issue_counts",
        "channel_nonzero_frames",
    }
)

# Required top-level keys for any valid processed sample payload.
REQUIRED_PAYLOAD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "sample_id",
        "schema_version",
        "text",
        "split",
        "num_frames",
        "fps",
        "selected_person",
        "frame_quality",
        "pose",
    }
)

# Required keys for a passed manifest entry record.
REQUIRED_PASSED_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "sample_id",
        "schema_version",
        "status",
        "text",
        "split",
        "num_frames",
        "fps",
        "sample_path",
        "source_video_id",
        "source_sentence_id",
        "source_sentence_name",
        "selected_person",
        "frame_quality",
    }
)

# Required keys for a dropped manifest entry record.
REQUIRED_DROPPED_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "sample_id",
        "schema_version",
        "status",
        "split",
        "drop_stage",
        "drop_reasons",
        "debug_only",
    }
)

DROPPED_ONLY_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "drop_stage",
        "drop_reasons",
        "drop_details",
        "debug_only",
    }
)

PASSED_ONLY_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "source_video_id",
        "source_sentence_id",
        "source_sentence_name",
    }
)
