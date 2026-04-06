"""OpenPose JSON inspection and normalization helpers."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .constants import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    EXPECTED_OPENPOSE_3D_KEYS,
    EXPECTED_OPENPOSE_PERSON_KEYS,
    EXPECTED_OPENPOSE_TOP_LEVEL_KEYS,
    OPENPOSE_CHANNEL_SPECS,
)

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int16]


@dataclass(slots=True)
class FirstFrameInspection:
    """Schema facts gathered from the first frame of a matched clip."""

    top_level_keys: tuple[str, ...]
    person_keys: tuple[str, ...]
    channel_lengths: dict[str, int]
    people_count: int
    has_face: bool
    issue_codes: list[str]


@dataclass(slots=True)
class ParsedFrame:
    """Normalized frame data plus audit metadata."""

    coords: dict[str, FloatArray]
    confidences: dict[str, FloatArray]
    people_count: int
    frame_valid: bool
    face_missing: bool
    out_of_bounds_coordinate_count: int
    has_any_zeroed_required_joint: bool
    issue_codes: list[str]
    top_level_keys: tuple[str, ...]
    person_keys: tuple[str, ...]
    channel_lengths: dict[str, int]


def zero_coords(channel: str) -> FloatArray:
    """Return a zero-filled coordinate tensor for a single frame."""

    _, num_points, _ = OPENPOSE_CHANNEL_SPECS[channel]
    return np.zeros((num_points, 2), dtype=np.float32)


def zero_confidence(channel: str) -> FloatArray:
    """Return a zero-filled confidence vector for a single frame."""

    _, num_points, _ = OPENPOSE_CHANNEL_SPECS[channel]
    return np.zeros((num_points,), dtype=np.float32)


def _reshape_flat_keypoints(
    flat_values: list[float], expected_points: int
) -> tuple[FloatArray, FloatArray]:
    if len(flat_values) != expected_points * 3:
        raise ValueError(f"Expected {expected_points * 3} values, got {len(flat_values)}.")

    raw = np.asarray(flat_values, dtype=np.float32).reshape(expected_points, 3)
    coords = raw[:, :2].copy()
    confidence = raw[:, 2].copy()
    return coords, confidence


def inspect_first_frame(path: Path) -> FirstFrameInspection:
    """Inspect a first-frame OpenPose payload for schema reporting."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return FirstFrameInspection(
            top_level_keys=(),
            person_keys=(),
            channel_lengths={},
            people_count=0,
            has_face=False,
            issue_codes=["json_decode_error"],
        )

    issue_codes: list[str] = []
    if not isinstance(payload, dict):
        return FirstFrameInspection(
            top_level_keys=(),
            person_keys=(),
            channel_lengths={},
            people_count=0,
            has_face=False,
            issue_codes=["top_level_not_object"],
        )

    top_level_keys = tuple(sorted(str(key) for key in payload.keys()))
    missing_top_level = sorted(EXPECTED_OPENPOSE_TOP_LEVEL_KEYS.difference(payload.keys()))
    for key in missing_top_level:
        issue_codes.append(f"missing_top_level_key:{key}")

    people = payload.get("people")
    if not isinstance(people, list):
        return FirstFrameInspection(
            top_level_keys=top_level_keys,
            person_keys=(),
            channel_lengths={},
            people_count=0,
            has_face=False,
            issue_codes=issue_codes + ["people_not_list"],
        )

    if not people:
        return FirstFrameInspection(
            top_level_keys=top_level_keys,
            person_keys=(),
            channel_lengths={},
            people_count=0,
            has_face=False,
            issue_codes=issue_codes + ["people_empty"],
        )

    person = people[0]
    if not isinstance(person, dict):
        return FirstFrameInspection(
            top_level_keys=top_level_keys,
            person_keys=(),
            channel_lengths={},
            people_count=len(people),
            has_face=False,
            issue_codes=issue_codes + ["person_not_object"],
        )

    person_keys = tuple(sorted(str(key) for key in person.keys()))
    missing_person_keys = sorted(EXPECTED_OPENPOSE_PERSON_KEYS.difference(person.keys()))
    for key in missing_person_keys:
        issue_codes.append(f"missing_person_key:{key}")

    channel_lengths: dict[str, int] = {}
    has_face = False
    for channel_name, (raw_key, expected_points, _) in OPENPOSE_CHANNEL_SPECS.items():
        raw_values = person.get(raw_key, [])
        channel_lengths[raw_key] = len(raw_values) if isinstance(raw_values, list) else -1
        if (
            channel_name == "face"
            and isinstance(raw_values, list)
            and len(raw_values) == expected_points * 3
        ):
            has_face = True
        if not isinstance(raw_values, list):
            issue_codes.append(f"channel_not_list:{raw_key}")
        elif len(raw_values) != expected_points * 3:
            issue_codes.append(f"unexpected_channel_length:{raw_key}:{len(raw_values)}")

    for key in EXPECTED_OPENPOSE_3D_KEYS:
        raw_values = person.get(key, [])
        if not isinstance(raw_values, list):
            issue_codes.append(f"channel_not_list:{key}")

    return FirstFrameInspection(
        top_level_keys=top_level_keys,
        person_keys=person_keys,
        channel_lengths=channel_lengths,
        people_count=len(people),
        has_face=has_face,
        issue_codes=issue_codes,
    )


def parse_frame(
    path: Path, *, canvas_width: int = CANVAS_WIDTH, canvas_height: int = CANVAS_HEIGHT
) -> ParsedFrame:
    """Parse and normalize one OpenPose frame JSON file."""

    coords = {channel: zero_coords(channel) for channel in OPENPOSE_CHANNEL_SPECS}
    confidences = {channel: zero_confidence(channel) for channel in OPENPOSE_CHANNEL_SPECS}
    issue_codes: list[str] = []
    top_level_keys: tuple[str, ...] = ()
    person_keys: tuple[str, ...] = ()
    channel_lengths: dict[str, int] = {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return ParsedFrame(
            coords=coords,
            confidences=confidences,
            people_count=0,
            frame_valid=False,
            face_missing=True,
            out_of_bounds_coordinate_count=0,
            has_any_zeroed_required_joint=False,
            issue_codes=["json_decode_error"],
            top_level_keys=top_level_keys,
            person_keys=person_keys,
            channel_lengths=channel_lengths,
        )

    if not isinstance(payload, dict):
        return ParsedFrame(
            coords=coords,
            confidences=confidences,
            people_count=0,
            frame_valid=False,
            face_missing=True,
            out_of_bounds_coordinate_count=0,
            has_any_zeroed_required_joint=False,
            issue_codes=["top_level_not_object"],
            top_level_keys=top_level_keys,
            person_keys=person_keys,
            channel_lengths=channel_lengths,
        )

    top_level_keys = tuple(sorted(str(key) for key in payload.keys()))
    for key in sorted(EXPECTED_OPENPOSE_TOP_LEVEL_KEYS.difference(payload.keys())):
        issue_codes.append(f"missing_top_level_key:{key}")

    people = payload.get("people")
    if not isinstance(people, list):
        return ParsedFrame(
            coords=coords,
            confidences=confidences,
            people_count=0,
            frame_valid=False,
            face_missing=True,
            out_of_bounds_coordinate_count=0,
            has_any_zeroed_required_joint=False,
            issue_codes=issue_codes + ["people_not_list"],
            top_level_keys=top_level_keys,
            person_keys=person_keys,
            channel_lengths=channel_lengths,
        )

    if not people:
        return ParsedFrame(
            coords=coords,
            confidences=confidences,
            people_count=0,
            frame_valid=False,
            face_missing=True,
            out_of_bounds_coordinate_count=0,
            has_any_zeroed_required_joint=False,
            issue_codes=issue_codes + ["people_empty"],
            top_level_keys=top_level_keys,
            person_keys=person_keys,
            channel_lengths=channel_lengths,
        )

    person = people[0]
    if not isinstance(person, dict):
        return ParsedFrame(
            coords=coords,
            confidences=confidences,
            people_count=len(people),
            frame_valid=False,
            face_missing=True,
            out_of_bounds_coordinate_count=0,
            has_any_zeroed_required_joint=False,
            issue_codes=issue_codes + ["person_not_object"],
            top_level_keys=top_level_keys,
            person_keys=person_keys,
            channel_lengths=channel_lengths,
        )

    person_keys = tuple(sorted(str(key) for key in person.keys()))
    for key in sorted(EXPECTED_OPENPOSE_PERSON_KEYS.difference(person.keys())):
        issue_codes.append(f"missing_person_key:{key}")

    frame_valid = True
    face_missing = False
    out_of_bounds_coordinate_count = 0
    has_any_zeroed_required_joint = False

    for channel, (raw_key, expected_points, required) in OPENPOSE_CHANNEL_SPECS.items():
        raw_values = person.get(raw_key, [])
        channel_lengths[raw_key] = len(raw_values) if isinstance(raw_values, list) else -1
        if not isinstance(raw_values, list):
            issue_codes.append(f"channel_not_list:{raw_key}")
            if required:
                frame_valid = False
            else:
                face_missing = True
            continue
        if len(raw_values) != expected_points * 3:
            issue_codes.append(f"unexpected_channel_length:{raw_key}:{len(raw_values)}")
            if required:
                frame_valid = False
            else:
                face_missing = True
            continue

        parsed_coords, parsed_confidence = _reshape_flat_keypoints(raw_values, expected_points)
        parsed_coords[:, 0] = parsed_coords[:, 0] / np.float32(canvas_width)
        parsed_coords[:, 1] = parsed_coords[:, 1] / np.float32(canvas_height)

        raw_array = np.asarray(raw_values, dtype=np.float32).reshape(expected_points, 3)
        x_values = raw_array[:, 0]
        y_values = raw_array[:, 1]
        out_of_bounds_coordinate_count += int(
            np.count_nonzero((x_values < 0) | (x_values > canvas_width))
        )
        out_of_bounds_coordinate_count += int(
            np.count_nonzero((y_values < 0) | (y_values > canvas_height))
        )

        coords[channel] = parsed_coords
        confidences[channel] = parsed_confidence

        # This is an audit/debug signal only. A zeroed raw joint row does not
        # by itself make the frame invalid if the channel shape is otherwise usable.
        if required and np.any(np.all(raw_array == 0.0, axis=1)):
            has_any_zeroed_required_joint = True

        if channel == "face" and float(np.sum(parsed_confidence)) <= 0.0:
            face_missing = True

    return ParsedFrame(
        coords=coords,
        confidences=confidences,
        people_count=len(people),
        frame_valid=frame_valid,
        face_missing=face_missing,
        out_of_bounds_coordinate_count=out_of_bounds_coordinate_count,
        has_any_zeroed_required_joint=has_any_zeroed_required_joint,
        issue_codes=issue_codes,
        top_level_keys=top_level_keys,
        person_keys=person_keys,
        channel_lengths=channel_lengths,
    )


def counter_to_sorted_mapping(counter: Counter[str]) -> dict[str, int]:
    """Convert a counter into a deterministic mapping."""

    return {key: counter[key] for key in sorted(counter)}
