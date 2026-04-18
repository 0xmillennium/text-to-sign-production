"""Sprint 3 processed modeling dataset item contract tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.support.builders.manifests import processed_record
from tests.support.modeling import (
    patch_modeling_repo_root,
    write_processed_modeling_manifest,
    write_processed_modeling_sample,
)
from text_to_sign_production.modeling.data import (
    ProcessedPoseDataset,
    read_processed_modeling_manifest,
)

pytestmark = pytest.mark.unit


def test_dataset_loads_manifest_referenced_npz_and_preserves_raw_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_modeling_repo_root(monkeypatch, tmp_path)
    frame_valid_mask = np.asarray([True, False, True], dtype=np.bool_)
    body = np.full((3, 25, 2), 2.0, dtype=np.float32)
    left_hand = np.full((3, 21, 2), 3.0, dtype=np.float32)
    right_hand = np.full((3, 21, 2), 4.0, dtype=np.float32)
    sample_path = write_processed_modeling_sample(
        tmp_path,
        sample_id="sample",
        num_frames=3,
        frame_valid_mask=frame_valid_mask,
        overrides={
            "body": body,
            "left_hand": left_hand,
            "right_hand": right_hand,
        },
    )
    manifest_path = write_processed_modeling_manifest(
        tmp_path,
        [
            processed_record(
                sample_path,
                sample_id="sample",
                text="A raw English sentence.",
                num_frames=3,
                frame_valid_count=2,
                frame_invalid_count=1,
            )
        ],
    )

    records = read_processed_modeling_manifest(manifest_path, split="train")
    dataset = ProcessedPoseDataset(manifest_path, split="train")
    item = dataset[0]

    assert len(records) == 1
    assert len(dataset) == 1
    assert item.sample_id == "sample"
    assert item.split == "train"
    assert item.text == "A raw English sentence."
    assert item.fps == 24.0
    assert item.num_frames == 3
    assert item.sample_path_value == sample_path
    assert item.body.shape == (3, 25, 2)
    assert item.left_hand.shape == (3, 21, 2)
    assert item.right_hand.shape == (3, 21, 2)
    assert item.body.dtype == np.float32
    assert np.all(item.body == 2.0)
    assert np.all(item.left_hand == 3.0)
    assert np.all(item.right_hand == 4.0)
    assert item.frame_valid_mask.tolist() == [True, False, True]
