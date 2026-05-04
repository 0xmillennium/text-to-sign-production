"""Sample payload construction and record conversion."""

from collections.abc import Mapping
from typing import Any

from text_to_sign_production.data.samples.schema import PROCESSED_SCHEMA_VERSION
from text_to_sign_production.data.samples.types import (
    BfhPosePayload,
    FrameQualitySummary,
    PoseChannelPayload,
    ProcessedSamplePayload,
    SelectedPersonMetadata,
)


def build_payload(
    sample_id: str,
    text: str,
    split: str,
    num_frames: int,
    fps: float | None,
    selected_person: SelectedPersonMetadata,
    frame_quality: FrameQualitySummary,
    pose: BfhPosePayload,
) -> ProcessedSamplePayload:
    """Construct a canonical processed sample payload."""
    return ProcessedSamplePayload(
        sample_id=sample_id,
        schema_version=PROCESSED_SCHEMA_VERSION,
        text=text,
        split=split,
        num_frames=num_frames,
        fps=fps,
        selected_person=selected_person,
        frame_quality=frame_quality,
        pose=pose,
    )


def payload_from_record(record: Mapping[str, Any]) -> ProcessedSamplePayload:
    """Parse a payload record into its typed sample contract."""
    selected_person_record = _require_mapping(record["selected_person"], "selected_person")
    frame_quality_record = _require_mapping(record["frame_quality"], "frame_quality")
    pose_record = _require_mapping(record["pose"], "pose")

    return ProcessedSamplePayload(
        sample_id=str(record["sample_id"]),
        schema_version=str(record["schema_version"]),
        text=str(record["text"]),
        split=str(record["split"]),
        num_frames=int(record["num_frames"]),
        fps=float(record["fps"]) if record.get("fps") is not None else None,
        selected_person=_selected_person_from_record(selected_person_record),
        frame_quality=_frame_quality_from_record(frame_quality_record),
        pose=_pose_from_record(pose_record),
    )


def _require_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Payload field {field_name!r} must be a mapping.")
    return value


def _selected_person_from_record(record: Mapping[str, Any]) -> SelectedPersonMetadata:
    return SelectedPersonMetadata(
        index=int(record["index"]),
        multi_person_frame_count=int(record["multi_person_frame_count"]),
        max_people_per_frame=int(record["max_people_per_frame"]),
    )


def _frame_quality_from_record(record: Mapping[str, Any]) -> FrameQualitySummary:
    return FrameQualitySummary(
        valid_frame_count=int(record["valid_frame_count"]),
        invalid_frame_count=int(record["invalid_frame_count"]),
        face_missing_frame_count=int(record["face_missing_frame_count"]),
        out_of_bounds_coordinate_count=int(record["out_of_bounds_coordinate_count"]),
        frames_with_any_zeroed_required_joint=int(record["frames_with_any_zeroed_required_joint"]),
        frame_issue_counts=_int_mapping(record.get("frame_issue_counts", {})),
        channel_nonzero_frames=_int_mapping(record.get("channel_nonzero_frames", {})),
    )


def _pose_from_record(record: Mapping[str, Any]) -> BfhPosePayload:
    return BfhPosePayload(
        body=_channel_from_record(_require_mapping(record["body"], "pose.body")),
        left_hand=_channel_from_record(_require_mapping(record["left_hand"], "pose.left_hand")),
        right_hand=_channel_from_record(_require_mapping(record["right_hand"], "pose.right_hand")),
        face=_channel_from_record(_require_mapping(record["face"], "pose.face")),
    )


def _channel_from_record(record: Mapping[str, Any]) -> PoseChannelPayload:
    return PoseChannelPayload(
        coordinates=record["coordinates"],
        confidence=record["confidence"],
    )


def _int_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError("Expected a mapping of string keys to integer counts.")
    return {str(key): int(item) for key, item in value.items()}
