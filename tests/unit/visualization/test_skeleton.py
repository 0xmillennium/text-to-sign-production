"""Unit tests for 2D skeleton frame rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from tests.support.builders.samples import write_processed_sample_npz
from tests.support.builders.visual_debug import renderable_pose_overrides
from text_to_sign_production.visualization import (
    PoseSample,
    SkeletonRenderConfig,
    load_pose_sample,
    render_pose_frame,
)

pytestmark = pytest.mark.unit


def test_skeleton_frame_rendering_produces_expected_dimensions(tmp_path: Path) -> None:
    sample = _write_and_load_renderable_sample(tmp_path)

    frame = render_pose_frame(sample, 0, config=SkeletonRenderConfig(draw_labels=False))

    assert frame.shape == (720, 1280, 3)
    assert frame.dtype == np.dtype(np.uint8)


def test_body_25_topology_rendering_changes_expected_pixels(tmp_path: Path) -> None:
    sample = _write_and_load_renderable_sample(tmp_path)
    baseline = _baseline_frame(sample)

    frame = render_pose_frame(sample, 0, config=SkeletonRenderConfig(draw_labels=False))

    assert not np.array_equal(frame[252, 640], baseline[252, 640])


def test_hand_21_topology_rendering_changes_expected_pixels(tmp_path: Path) -> None:
    sample = _write_and_load_renderable_sample(tmp_path)
    baseline = _baseline_frame(sample)

    frame = render_pose_frame(sample, 0, config=SkeletonRenderConfig(draw_labels=False))

    assert not np.array_equal(frame[360, 350], baseline[360, 350])
    assert not np.array_equal(frame[360, 930], baseline[360, 930])


def test_face_scatter_changes_expected_pixels(tmp_path: Path) -> None:
    sample = _write_and_load_renderable_sample(tmp_path)
    baseline = _baseline_frame(sample)

    frame = render_pose_frame(sample, 0, config=SkeletonRenderConfig(draw_labels=False))

    assert not np.array_equal(frame[72, 128], baseline[72, 128])


def test_confidence_threshold_filters_low_confidence_joints(tmp_path: Path) -> None:
    overrides = renderable_pose_overrides(num_frames=3)
    overrides["body_confidence"][:, [1, 8]] = 0.4
    sample_path = tmp_path / "sample.npz"
    write_processed_sample_npz(sample_path, num_frames=3, overrides=overrides)
    sample = load_pose_sample(sample_path)
    baseline = _baseline_frame(sample)

    frame = render_pose_frame(
        sample,
        0,
        config=SkeletonRenderConfig(confidence_threshold=0.5, draw_labels=False),
    )

    assert np.array_equal(frame[252, 640], baseline[252, 640])


def test_invalid_frame_rendering_has_visible_status_label(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.npz"
    write_processed_sample_npz(
        sample_path,
        num_frames=3,
        overrides=renderable_pose_overrides(
            num_frames=3,
            frame_valid_mask=np.asarray([False, True, True], dtype=np.bool_),
        ),
    )
    sample = load_pose_sample(sample_path)

    frame = render_pose_frame(sample, 0)

    assert frame.shape == (720, 1280, 3)
    assert int(np.count_nonzero(frame[12:52, 12:720])) > 0


def test_missing_and_non_finite_joints_are_skipped(tmp_path: Path) -> None:
    overrides = renderable_pose_overrides(num_frames=3)
    overrides["body"][0, 1, :] = [np.nan, np.inf]
    sample_path = tmp_path / "sample.npz"
    write_processed_sample_npz(sample_path, num_frames=3, overrides=overrides)
    sample = load_pose_sample(sample_path)

    frame = render_pose_frame(sample, 0, config=SkeletonRenderConfig(draw_labels=False))

    assert frame.shape == (720, 1280, 3)


def _write_and_load_renderable_sample(tmp_path: Path) -> PoseSample:
    sample_path = tmp_path / "sample.npz"
    write_processed_sample_npz(
        sample_path,
        num_frames=3,
        overrides=renderable_pose_overrides(num_frames=3),
    )
    return load_pose_sample(sample_path)


def _baseline_frame(sample: PoseSample) -> npt.NDArray[np.uint8]:
    return render_pose_frame(
        sample,
        0,
        config=SkeletonRenderConfig(confidence_threshold=2.0, draw_labels=False),
    )
