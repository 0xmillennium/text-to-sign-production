"""Sprint 3 processed modeling dataset contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import text_to_sign_production.data.utils as utils_mod
from tests.support.builders.manifests import processed_record, write_jsonl_records
from tests.support.builders.samples import write_processed_sample_npz
from text_to_sign_production.data.constants import PROCESSED_SCHEMA_VERSION
from text_to_sign_production.modeling.data import (
    SPRINT3_TARGET_CHANNEL_SHAPES,
    SPRINT3_TARGET_CHANNELS,
    ProcessedModelingDataError,
    ProcessedPoseDataset,
    load_processed_pose_sample,
    read_processed_modeling_manifest,
)

pytestmark = pytest.mark.unit


def _sample_relative_path(split: str = "train", sample_id: str = "sample") -> str:
    return f"data/processed/v1/samples/{split}/{sample_id}.npz"


def _write_manifest(root: Path, records: list[dict[str, Any]], *, split: str = "train") -> Path:
    manifest_path = root / f"data/processed/v1/manifests/{split}.jsonl"
    write_jsonl_records(manifest_path, records)
    return manifest_path


def _write_sample(
    root: Path,
    *,
    split: str = "train",
    sample_id: str = "sample",
    num_frames: int = 2,
    frame_valid_mask: npt.NDArray[np.bool_] | None = None,
    overrides: dict[str, Any] | None = None,
    drop_keys: tuple[str, ...] = (),
) -> str:
    sample_relative_path = _sample_relative_path(split=split, sample_id=sample_id)
    resolved_overrides = {} if overrides is None else dict(overrides)
    if frame_valid_mask is not None:
        resolved_overrides["frame_valid_mask"] = frame_valid_mask
    write_processed_sample_npz(
        root / sample_relative_path,
        num_frames=num_frames,
        overrides=resolved_overrides,
        drop_keys=drop_keys,
    )
    return sample_relative_path


def test_sprint3_target_channel_contract_uses_expected_processed_shapes() -> None:
    assert SPRINT3_TARGET_CHANNELS == ("body", "left_hand", "right_hand")
    assert SPRINT3_TARGET_CHANNEL_SHAPES == {
        "body": (25, 2),
        "left_hand": (21, 2),
        "right_hand": (21, 2),
    }


def test_dataset_loads_manifest_referenced_npz_and_preserves_raw_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    frame_valid_mask = np.asarray([True, False, True], dtype=np.bool_)
    body = np.full((3, 25, 2), 2.0, dtype=np.float32)
    left_hand = np.full((3, 21, 2), 3.0, dtype=np.float32)
    right_hand = np.full((3, 21, 2), 4.0, dtype=np.float32)
    sample_path = _write_sample(
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
    manifest_path = _write_manifest(
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


def test_manifest_reader_rejects_unknown_requested_split(tmp_path: Path) -> None:
    with pytest.raises(ProcessedModelingDataError, match="unknown split"):
        read_processed_modeling_manifest(tmp_path / "missing.jsonl", split="dev")


def test_manifest_reader_wraps_malformed_jsonl_with_line_context(tmp_path: Path) -> None:
    manifest_path = tmp_path / "data/processed/v1/manifests/train.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{broken\n", encoding="utf-8")

    with pytest.raises(ProcessedModelingDataError, match="line 1.*not valid JSON"):
        read_processed_modeling_manifest(manifest_path, split="train")


def test_manifest_reader_wraps_non_object_jsonl_with_line_context(tmp_path: Path) -> None:
    manifest_path = tmp_path / "data/processed/v1/manifests/train.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ProcessedModelingDataError, match="line 1.*JSON object"):
        read_processed_modeling_manifest(manifest_path, split="train")


def test_manifest_reader_rejects_unknown_record_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        [
            processed_record(
                "data/processed/v1/samples/dev/sample.npz",
                split="dev",
            )
        ],
    )

    with pytest.raises(ProcessedModelingDataError, match="unknown split"):
        read_processed_modeling_manifest(manifest_path)


def test_manifest_reader_rejects_sample_id_with_outer_whitespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    sample_path = _write_sample(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        [
            processed_record(
                sample_path,
                sample_id=" sample ",
            )
        ],
    )

    with pytest.raises(ProcessedModelingDataError, match="whitespace in sample_id"):
        read_processed_modeling_manifest(manifest_path, split="train")


def test_manifest_reader_rejects_duplicate_sample_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    sample_path = _write_sample(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        [
            processed_record(sample_path),
            processed_record(sample_path),
        ],
    )

    with pytest.raises(ProcessedModelingDataError, match="Duplicate processed sample_id"):
        read_processed_modeling_manifest(manifest_path, split="train")


def test_manifest_reader_rejects_missing_manifest_sample_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    sample_path = _sample_relative_path()
    manifest_path = _write_manifest(tmp_path, [processed_record(sample_path)])

    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_processed_modeling_manifest(manifest_path, split="train")


def test_manifest_reader_rejects_invalid_manifest_sample_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    manifest_path = _write_manifest(tmp_path, [processed_record("/tmp/sample.npz")])

    with pytest.raises(ProcessedModelingDataError, match="invalid sample_path"):
        read_processed_modeling_manifest(manifest_path, split="train")


def test_manifest_reader_rejects_record_split_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    sample_path = _write_sample(tmp_path)
    manifest_path = _write_manifest(tmp_path, [processed_record(sample_path)])

    with pytest.raises(ProcessedModelingDataError, match="expected 'val'"):
        read_processed_modeling_manifest(manifest_path, split="val")


def test_load_processed_pose_sample_rejects_unexpected_payload_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    sample_path = _write_sample(
        tmp_path,
        overrides={"processed_schema_version": np.asarray("t2sp-processed-v999")},
    )
    manifest_path = _write_manifest(tmp_path, [processed_record(sample_path)])
    record = read_processed_modeling_manifest(manifest_path, split="train")[0]

    with pytest.raises(ProcessedModelingDataError, match=PROCESSED_SCHEMA_VERSION):
        load_processed_pose_sample(record)


def test_load_processed_pose_sample_rejects_selected_person_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    sample_path = _write_sample(
        tmp_path,
        overrides={"selected_person_index": np.asarray(1, dtype=np.int16)},
    )
    manifest_path = _write_manifest(tmp_path, [processed_record(sample_path)])
    record = read_processed_modeling_manifest(manifest_path, split="train")[0]

    with pytest.raises(ProcessedModelingDataError, match="selected_person_index"):
        load_processed_pose_sample(record)


def test_load_processed_pose_sample_rejects_missing_target_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    sample_path = _write_sample(tmp_path, drop_keys=("left_hand",))
    manifest_path = _write_manifest(tmp_path, [processed_record(sample_path)])
    record = read_processed_modeling_manifest(manifest_path, split="train")[0]

    with pytest.raises(ProcessedModelingDataError, match="left_hand"):
        load_processed_pose_sample(record)


def test_load_processed_pose_sample_rejects_target_channel_shape_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    sample_path = _write_sample(
        tmp_path,
        overrides={"body": np.zeros((3, 25, 2), dtype=np.float32)},
    )
    manifest_path = _write_manifest(tmp_path, [processed_record(sample_path)])
    record = read_processed_modeling_manifest(manifest_path, split="train")[0]

    with pytest.raises(ProcessedModelingDataError, match=r"expected \(2, 25, 2\)"):
        load_processed_pose_sample(record)


def test_load_processed_pose_sample_rejects_frame_valid_mask_shape_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    sample_path = _write_sample(
        tmp_path,
        frame_valid_mask=np.asarray([True, False, True], dtype=np.bool_),
    )
    manifest_path = _write_manifest(tmp_path, [processed_record(sample_path)])
    record = read_processed_modeling_manifest(manifest_path, split="train")[0]

    with pytest.raises(ProcessedModelingDataError, match="frame_valid_mask"):
        load_processed_pose_sample(record)


def test_load_processed_pose_sample_rejects_frame_valid_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    sample_path = _write_sample(
        tmp_path,
        frame_valid_mask=np.asarray([True, False], dtype=np.bool_),
    )
    manifest_path = _write_manifest(tmp_path, [processed_record(sample_path)])
    record = read_processed_modeling_manifest(manifest_path, split="train")[0]

    with pytest.raises(ProcessedModelingDataError, match="valid count"):
        load_processed_pose_sample(record)
