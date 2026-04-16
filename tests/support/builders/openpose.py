"""OpenPose fake payload builders used by unit, integration, and e2e tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_channel(
    num_points: int,
    *,
    x_offset: float = 100.0,
    y_offset: float = 200.0,
    zero_last: bool = False,
    confidence: float = 0.9,
) -> list[float]:
    values: list[float] = []
    for index in range(num_points):
        if zero_last and index == num_points - 1:
            values.extend([0.0, 0.0, 0.0])
        else:
            values.extend([x_offset + float(index), y_offset + float(index), confidence])
    return values


def person_payload(
    *,
    zero_body_joint: bool = False,
    face_confidence: float = 0.9,
) -> dict[str, Any]:
    return {
        "person_id": [-1],
        "pose_keypoints_2d": build_channel(25, zero_last=zero_body_joint),
        "face_keypoints_2d": build_channel(
            70,
            x_offset=300.0,
            y_offset=150.0,
            confidence=face_confidence,
        ),
        "hand_left_keypoints_2d": build_channel(21, x_offset=400.0, y_offset=350.0),
        "hand_right_keypoints_2d": build_channel(21, x_offset=500.0, y_offset=320.0),
        "pose_keypoints_3d": [],
        "face_keypoints_3d": [],
        "hand_left_keypoints_3d": [],
        "hand_right_keypoints_3d": [],
    }


def frame_payload(*, people: list[dict[str, Any]]) -> dict[str, Any]:
    return {"version": 1.3, "people": people}


def write_openpose_frame(path: Path, *, people: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(frame_payload(people=people)), encoding="utf-8")
