"""Low-level OpenPose frame parsing tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.support.builders.openpose import person_payload, write_openpose_frame
from text_to_sign_production.data.openpose import parse_frame

pytestmark = pytest.mark.unit


def test_parse_frame_reads_static_tiny_fixture(fixtures_dir: Path) -> None:
    parsed = parse_frame(fixtures_dir / "openpose/tiny_frame.json")

    assert parsed.people_count == 1
    assert parsed.frame_valid is True
    assert parsed.coords["body"].shape == (25, 2)
    assert np.isclose(parsed.coords["body"][0, 0], 100.0 / 1280.0)
    assert parsed.confidences["left_hand"].shape == (21,)


def test_parse_frame_tracks_multi_person_and_any_zeroed_required_joint(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.json"
    write_openpose_frame(
        frame_path,
        people=[person_payload(zero_body_joint=True), person_payload()],
    )

    parsed = parse_frame(frame_path)

    assert parsed.people_count == 2
    assert parsed.frame_valid is True
    assert parsed.has_any_zeroed_required_joint is True
    assert parsed.coords["body"].shape == (25, 2)
    assert np.isclose(parsed.coords["body"][0, 0], 100.0 / 1280.0)
    assert parsed.confidences["left_hand"].shape == (21,)


def test_parse_frame_counts_canvas_edge_coordinates_as_out_of_bounds(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.json"
    payload = person_payload()
    payload["pose_keypoints_2d"][:3] = [1280.0, 720.0, 0.9]
    write_openpose_frame(frame_path, people=[payload])

    parsed = parse_frame(frame_path)

    assert parsed.out_of_bounds_coordinate_count == 2


def test_parse_frame_marks_face_missing_when_face_channel_has_no_confidence(
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "frame.json"
    payload = person_payload()
    payload["face_keypoints_2d"] = [0.0] * (70 * 3)
    write_openpose_frame(frame_path, people=[payload])

    parsed = parse_frame(frame_path)

    assert parsed.face_missing is True
    assert parsed.frame_valid is True
