"""Unit tests for processed pose sample loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.builders.samples import write_processed_sample_npz
from tests.support.builders.visual_debug import renderable_pose_overrides
from text_to_sign_production.visualization import PoseSampleError, load_pose_sample

pytestmark = pytest.mark.unit


def test_load_pose_sample_reads_required_processed_v1_arrays(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.npz"
    write_processed_sample_npz(
        sample_path,
        num_frames=3,
        overrides=renderable_pose_overrides(num_frames=3),
    )

    sample = load_pose_sample(sample_path)

    assert sample.path == sample_path.resolve()
    assert sample.num_frames == 3
    assert sample.body.shape == (3, 25, 2)
    assert sample.left_hand_confidence.shape == (3, 21)
    assert sample.frame_valid_mask.tolist() == [True, True, True]


def test_load_pose_sample_rejects_missing_required_arrays(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.npz"
    write_processed_sample_npz(sample_path, drop_keys=("body",))

    with pytest.raises(PoseSampleError, match="missing required arrays"):
        load_pose_sample(sample_path)
