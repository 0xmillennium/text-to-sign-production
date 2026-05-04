"""Frame JSON parsing into domain channels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from text_to_sign_production.data.pose.schema import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    EXPECTED_OPENPOSE_PERSON_KEYS,
    EXPECTED_OPENPOSE_TOP_LEVEL_KEYS,
    OPENPOSE_CHANNEL_SPECS,
)
from text_to_sign_production.data.pose.types import (
    FloatArray,
    ParsedFrameResult,
    ParsedPerson,
)


def zero_coords(channel: str) -> FloatArray:
    """Return a zero-filled coordinate tensor for a single frame."""
    _, num_points = OPENPOSE_CHANNEL_SPECS[channel]
    return np.zeros((num_points, 2), dtype=np.float32)


def zero_confidence(channel: str) -> FloatArray:
    """Return a zero-filled confidence vector for a single frame."""
    _, num_points = OPENPOSE_CHANNEL_SPECS[channel]
    return np.zeros((num_points,), dtype=np.float32)


def _reshape_flat_keypoints(
    flat_values: list[float], expected_points: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if len(flat_values) != expected_points * 3:
        raise ValueError(f"Expected {expected_points * 3} values, got {len(flat_values)}.")

    raw = np.asarray(flat_values, dtype=np.float32).reshape(expected_points, 3)
    coords = raw[:, :2].copy()
    confidence = raw[:, 2].copy()
    return raw, coords, confidence


def parse_frame(
    path: Path,
    *,
    canvas_width: int = CANVAS_WIDTH,
    canvas_height: int = CANVAS_HEIGHT,
) -> ParsedFrameResult:
    """Parse and normalize one OpenPose frame JSON file."""

    issue_codes: list[str] = []

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return ParsedFrameResult(
            people=[],
            frame_valid=False,
            issue_codes=["json_decode_error"],
        )
    except UnicodeDecodeError:
        return ParsedFrameResult(
            people=[],
            frame_valid=False,
            issue_codes=["unicode_decode_error"],
        )

    if not isinstance(payload, dict):
        return ParsedFrameResult(
            people=[],
            frame_valid=False,
            issue_codes=["top_level_not_object"],
        )

    for key in sorted(EXPECTED_OPENPOSE_TOP_LEVEL_KEYS.difference(payload.keys())):
        issue_codes.append(f"missing_top_level_key:{key}")

    raw_people = payload.get("people")
    if not isinstance(raw_people, list):
        return ParsedFrameResult(
            people=[],
            frame_valid=False,
            issue_codes=issue_codes + ["people_not_list"],
        )

    if not raw_people:
        return ParsedFrameResult(
            people=[],
            frame_valid=False,
            issue_codes=issue_codes + ["people_empty"],
        )

    parsed_people: list[ParsedPerson] = []

    for person in raw_people:
        if not isinstance(person, dict):
            parsed_people.append(
                ParsedPerson(
                    coords={ch: zero_coords(ch) for ch in OPENPOSE_CHANNEL_SPECS},
                    confidences={ch: zero_confidence(ch) for ch in OPENPOSE_CHANNEL_SPECS},
                    person_valid=False,
                    face_missing=True,
                    out_of_bounds_coordinate_count=0,
                    has_any_zeroed_canonical_joint=False,
                    issue_codes=["person_not_object"],
                )
            )
            continue

        person_issues: list[str] = []
        for key in sorted(EXPECTED_OPENPOSE_PERSON_KEYS.difference(person.keys())):
            person_issues.append(f"missing_person_key:{key}")

        person_valid = True
        face_missing = False
        out_of_bounds = 0
        has_zeroed = False

        coords = {}
        confidences = {}

        for channel, (raw_key, expected_points) in OPENPOSE_CHANNEL_SPECS.items():
            raw_values = person.get(raw_key, [])
            if not isinstance(raw_values, list):
                person_issues.append(f"channel_not_list:{raw_key}")
                person_valid = False
                coords[channel] = zero_coords(channel)
                confidences[channel] = zero_confidence(channel)
                continue
            if len(raw_values) != expected_points * 3:
                person_issues.append(f"unexpected_channel_length:{raw_key}:{len(raw_values)}")
                person_valid = False
                coords[channel] = zero_coords(channel)
                confidences[channel] = zero_confidence(channel)
                continue

            try:
                raw_array, parsed_coords, parsed_confidence = _reshape_flat_keypoints(
                    raw_values, expected_points
                )
            except (TypeError, ValueError):
                person_issues.append(f"channel_non_numeric:{raw_key}")
                person_valid = False
                coords[channel] = zero_coords(channel)
                confidences[channel] = zero_confidence(channel)
                continue
            parsed_coords[:, 0] = parsed_coords[:, 0] / np.float32(canvas_width)
            parsed_coords[:, 1] = parsed_coords[:, 1] / np.float32(canvas_height)

            x_values = raw_array[:, 0]
            y_values = raw_array[:, 1]
            out_of_bounds += int(np.count_nonzero((x_values < 0) | (x_values >= canvas_width)))
            out_of_bounds += int(np.count_nonzero((y_values < 0) | (y_values >= canvas_height)))

            coords[channel] = parsed_coords
            confidences[channel] = parsed_confidence

            if np.any(np.all(raw_array == 0.0, axis=1)):
                has_zeroed = True

            if channel == "face" and float(np.sum(parsed_confidence)) <= 0.0:
                face_missing = True

        parsed_people.append(
            ParsedPerson(
                coords=coords,
                confidences=confidences,
                person_valid=person_valid,
                face_missing=face_missing,
                out_of_bounds_coordinate_count=out_of_bounds,
                has_any_zeroed_canonical_joint=has_zeroed,
                issue_codes=person_issues,
            )
        )

    # Frame is valid if at least one person is structurally valid.
    frame_valid = any(p.person_valid for p in parsed_people)

    return ParsedFrameResult(
        people=parsed_people,
        frame_valid=frame_valid,
        issue_codes=issue_codes,
    )
