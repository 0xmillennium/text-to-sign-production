"""Sample payload construction, record conversion, and artifact IO."""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from text_to_sign_production.data._shared.identities import SampleSplit
from text_to_sign_production.data.samples._shared.parsing import (
    frame_quality_from_record,
    require_mapping,
    sample_split_from_record,
    selected_person_from_record,
)
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
    split: SampleSplit | str,
    num_frames: int,
    fps: float | None,
    selected_person: SelectedPersonMetadata,
    frame_quality: FrameQualitySummary,
    pose: BfhPosePayload,
    people_per_frame: Any | None = None,
    frame_valid_mask: Any | None = None,
) -> ProcessedSamplePayload:
    """Construct a canonical processed sample payload."""
    return ProcessedSamplePayload(
        sample_id=sample_id,
        schema_version=PROCESSED_SCHEMA_VERSION,
        text=text,
        split=SampleSplit(split),
        num_frames=num_frames,
        fps=fps,
        selected_person=selected_person,
        frame_quality=frame_quality,
        pose=pose,
        people_per_frame=people_per_frame,
        frame_valid_mask=frame_valid_mask,
    )


def payload_from_record(record: Mapping[str, Any]) -> ProcessedSamplePayload:
    """Parse a payload record into its typed sample contract."""
    selected_person_record = require_mapping(
        record["selected_person"], "selected_person", surface="Payload"
    )
    frame_quality_record = require_mapping(
        record["frame_quality"], "frame_quality", surface="Payload"
    )
    pose_record = require_mapping(record["pose"], "pose", surface="Payload")

    return ProcessedSamplePayload(
        sample_id=str(record["sample_id"]),
        schema_version=str(record["schema_version"]),
        text=str(record["text"]),
        split=sample_split_from_record(record["split"], "Payload split"),
        num_frames=int(record["num_frames"]),
        fps=float(record["fps"]) if record.get("fps") is not None else None,
        selected_person=selected_person_from_record(selected_person_record),
        frame_quality=frame_quality_from_record(frame_quality_record),
        pose=_pose_from_record(pose_record),
    )


def _pose_from_record(record: Mapping[str, Any]) -> BfhPosePayload:
    return BfhPosePayload(
        body=_channel_from_record(require_mapping(record["body"], "pose.body", surface="Payload")),
        left_hand=_channel_from_record(
            require_mapping(record["left_hand"], "pose.left_hand", surface="Payload")
        ),
        right_hand=_channel_from_record(
            require_mapping(record["right_hand"], "pose.right_hand", surface="Payload")
        ),
        face=_channel_from_record(require_mapping(record["face"], "pose.face", surface="Payload")),
    )


def _channel_from_record(record: Mapping[str, Any]) -> PoseChannelPayload:
    return PoseChannelPayload(
        coordinates=record["coordinates"],
        confidence=record["confidence"],
    )


def write_processed_sample_payload(path: Path, payload: ProcessedSamplePayload) -> None:
    """Write one canonical processed sample payload as a compressed NPZ artifact."""
    from text_to_sign_production.data.samples.validate import validate_payload

    issues = validate_payload(payload)
    if issues:
        codes = ", ".join(issue.code for issue in issues)
        raise ValueError(f"Invalid processed sample payload: {codes}")
    people_per_frame = _required_people_per_frame(payload)
    frame_valid_mask = _required_frame_valid_mask(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        processed_schema_version=np.asarray(payload.schema_version),
        sample_id=np.asarray(payload.sample_id),
        split=np.asarray(payload.split.value),
        text=np.asarray(payload.text),
        fps=np.asarray(np.nan if payload.fps is None else payload.fps, dtype=np.float32),
        body=np.asarray(payload.pose.body.coordinates),
        body_confidence=np.asarray(payload.pose.body.confidence),
        left_hand=np.asarray(payload.pose.left_hand.coordinates),
        left_hand_confidence=np.asarray(payload.pose.left_hand.confidence),
        right_hand=np.asarray(payload.pose.right_hand.coordinates),
        right_hand_confidence=np.asarray(payload.pose.right_hand.confidence),
        face=np.asarray(payload.pose.face.coordinates),
        face_confidence=np.asarray(payload.pose.face.confidence),
        selected_person_index=np.asarray(payload.selected_person.index, dtype=np.int16),
        people_per_frame=people_per_frame,
        frame_valid_mask=frame_valid_mask,
        frame_quality_json=np.asarray(
            json.dumps(payload.frame_quality.to_record(), sort_keys=True)
        ),
        selected_person_json=np.asarray(
            json.dumps(payload.selected_person.to_record(), sort_keys=True)
        ),
    )


def load_processed_sample_payload(path: Path) -> ProcessedSamplePayload:
    """Load one canonical processed sample payload from a compressed NPZ artifact."""
    try:
        with np.load(path, allow_pickle=False) as sample:
            body = np.asarray(sample["body"])
            record = {
                "sample_id": str(np.asarray(sample["sample_id"]).item()),
                "schema_version": str(np.asarray(sample["processed_schema_version"]).item()),
                "text": str(np.asarray(sample["text"]).item()),
                "split": str(np.asarray(sample["split"]).item()),
                "num_frames": int(body.shape[0]),
                "fps": _optional_fps(np.asarray(sample["fps"]).item()),
                "selected_person": json.loads(
                    str(np.asarray(sample["selected_person_json"]).item())
                ),
                "frame_quality": json.loads(str(np.asarray(sample["frame_quality_json"]).item())),
                "pose": {
                    "body": {
                        "coordinates": body,
                        "confidence": np.asarray(sample["body_confidence"]),
                    },
                    "left_hand": {
                        "coordinates": np.asarray(sample["left_hand"]),
                        "confidence": np.asarray(sample["left_hand_confidence"]),
                    },
                    "right_hand": {
                        "coordinates": np.asarray(sample["right_hand"]),
                        "confidence": np.asarray(sample["right_hand_confidence"]),
                    },
                    "face": {
                        "coordinates": np.asarray(sample["face"]),
                        "confidence": np.asarray(sample["face_confidence"]),
                    },
                },
            }
            payload = payload_from_record(record)
            payload.people_per_frame = np.asarray(sample["people_per_frame"])
            payload.frame_valid_mask = np.asarray(sample["frame_valid_mask"])
            from text_to_sign_production.data.samples.validate import validate_payload

            issues = validate_payload(payload)
            if issues:
                codes = ", ".join(issue.code for issue in issues)
                raise ValueError(f"Invalid processed sample payload: {codes}")
            _require_valid_artifact_arrays(path, payload)
            return payload
    except (BadZipFile, EOFError, OSError, KeyError, ValueError, TypeError) as exc:
        raise ValueError(
            f"Processed sample payload could not be loaded from {path}: {exc}"
        ) from exc


def _required_people_per_frame(payload: ProcessedSamplePayload) -> np.ndarray:
    if payload.people_per_frame is None:
        raise ValueError(f"Payload {payload.sample_id!r} is missing people_per_frame.")
    people_per_frame = np.asarray(payload.people_per_frame)
    expected_shape = (payload.num_frames,)
    if tuple(people_per_frame.shape) != expected_shape:
        raise ValueError(
            f"people_per_frame shape for {payload.sample_id!r} must be "
            f"{expected_shape}, got {tuple(people_per_frame.shape)}."
        )
    if not np.issubdtype(people_per_frame.dtype, np.integer):
        raise ValueError(f"people_per_frame for {payload.sample_id!r} must be integer.")
    return people_per_frame


def _required_frame_valid_mask(payload: ProcessedSamplePayload) -> np.ndarray:
    if payload.frame_valid_mask is None:
        raise ValueError(f"Payload {payload.sample_id!r} is missing frame_valid_mask.")
    frame_valid_mask = np.asarray(payload.frame_valid_mask)
    expected_shape = (payload.num_frames,)
    if tuple(frame_valid_mask.shape) != expected_shape:
        raise ValueError(
            f"frame_valid_mask shape for {payload.sample_id!r} must be "
            f"{expected_shape}, got {tuple(frame_valid_mask.shape)}."
        )
    if frame_valid_mask.dtype != np.dtype(np.bool_):
        raise ValueError(f"frame_valid_mask for {payload.sample_id!r} must be bool.")
    valid_count = int(np.count_nonzero(frame_valid_mask))
    if valid_count != payload.frame_quality.valid_frame_count:
        raise ValueError(
            f"frame_valid_mask valid count for {payload.sample_id!r} must match "
            "frame_quality.valid_frame_count."
        )
    return frame_valid_mask


def _require_valid_artifact_arrays(path: Path, payload: ProcessedSamplePayload) -> None:
    _required_people_per_frame(payload)
    _required_frame_valid_mask(payload)


def _optional_fps(value: Any) -> float | None:
    fps = float(value)
    if math.isnan(fps):
        return None
    return fps
