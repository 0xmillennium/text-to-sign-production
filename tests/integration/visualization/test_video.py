"""Integration tests for skeleton-only and side-by-side MP4 rendering."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from tests.support.builders.media import write_tiny_decodable_mp4
from tests.support.builders.samples import write_processed_sample_npz
from tests.support.builders.visual_debug import renderable_pose_overrides
from text_to_sign_production.visualization import (
    load_pose_sample,
    render_side_by_side_video,
    render_skeleton_video,
)

pytestmark = pytest.mark.integration


def test_skeleton_only_rendering_writes_readable_mp4(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.npz"
    write_processed_sample_npz(
        sample_path,
        num_frames=3,
        overrides=renderable_pose_overrides(num_frames=3),
    )
    pose_sample = load_pose_sample(sample_path)
    output_path = tmp_path / "skeleton.mp4"

    metadata = render_skeleton_video(
        pose_sample=pose_sample,
        output_path=output_path,
        fps=24.0,
    )

    assert metadata["output_path"] == output_path.resolve().as_posix()
    capture = cv2.VideoCapture(str(output_path))
    try:
        ok, frame = capture.read()
        assert ok
        assert frame.shape[:2] == (720, 1280)
    finally:
        capture.release()


def test_side_by_side_rendering_writes_readable_mp4_with_expected_dimensions(
    tmp_path: Path,
) -> None:
    sample_path = tmp_path / "sample.npz"
    write_processed_sample_npz(
        sample_path,
        num_frames=3,
        overrides=renderable_pose_overrides(num_frames=3),
    )
    pose_sample = load_pose_sample(sample_path)
    source_video = tmp_path / "source.mp4"
    write_tiny_decodable_mp4(source_video, frame_count=3)
    output_path = tmp_path / "debug.mp4"

    metadata = render_side_by_side_video(
        source_video_path=source_video,
        pose_sample=pose_sample,
        output_path=output_path,
        fps=24.0,
    )

    assert metadata["output_path"] == output_path.resolve().as_posix()
    capture = cv2.VideoCapture(str(output_path))
    try:
        ok, frame = capture.read()
        assert ok
        assert frame.shape[:2] == (720, 2560)
    finally:
        capture.release()
