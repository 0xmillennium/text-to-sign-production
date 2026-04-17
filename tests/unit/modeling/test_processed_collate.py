"""Sprint 3 processed pose collation contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

import text_to_sign_production.modeling.data.collate as collate_module
from text_to_sign_production.modeling.data import ProcessedPoseItem

pytestmark = pytest.mark.unit


class _FakeTensor:
    def __init__(self, array: npt.NDArray[Any]) -> None:
        self._array = array

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._array.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._array.shape)

    def tolist(self) -> list[Any]:
        return cast(list[Any], self._array.tolist())

    def __getitem__(self, index: Any) -> _FakeTensor:
        return _FakeTensor(np.asarray(self._array[index]))

    def __setitem__(self, index: Any, value: object) -> None:
        if isinstance(value, _FakeTensor):
            self._array[index] = value._array
        else:
            self._array[index] = value

    def mean(self) -> _FakeTensor:
        return _FakeTensor(np.asarray(self._array.mean()))

    def abs(self) -> _FakeTensor:
        return _FakeTensor(np.abs(self._array))

    def sum(self) -> _FakeTensor:
        return _FakeTensor(np.asarray(self._array.sum()))

    def item(self) -> Any:
        return self._array.item()


class _FakeTorch:
    float32 = np.dtype(np.float32)
    bool = np.dtype(np.bool_)
    long = np.dtype(np.int64)

    def zeros(self, shape: tuple[int, ...], *, dtype: np.dtype[Any]) -> _FakeTensor:
        return _FakeTensor(np.zeros(shape, dtype=dtype))

    def ones(self, shape: tuple[int, ...], *, dtype: np.dtype[Any]) -> _FakeTensor:
        return _FakeTensor(np.ones(shape, dtype=dtype))

    def as_tensor(self, data: object, *, dtype: np.dtype[Any]) -> _FakeTensor:
        return _FakeTensor(np.asarray(data, dtype=dtype))


@pytest.fixture
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> _FakeTorch:
    torch = _FakeTorch()

    def fake_import_module(module_name: str) -> _FakeTorch:
        assert module_name == "torch"
        return torch

    monkeypatch.setattr(collate_module, "import_module", fake_import_module)
    return torch


def _item(
    sample_id: str,
    *,
    length: int,
    fill_value: float,
    frame_valid_mask: list[bool],
) -> ProcessedPoseItem:
    return ProcessedPoseItem(
        sample_id=sample_id,
        split="train",
        text=f"text for {sample_id}",
        fps=24.0,
        num_frames=length,
        sample_path=Path(f"/tmp/{sample_id}.npz"),
        sample_path_value=f"data/processed/v1/samples/train/{sample_id}.npz",
        body=np.full((length, 25, 2), fill_value, dtype=np.float32),
        left_hand=np.full((length, 21, 2), fill_value + 1.0, dtype=np.float32),
        right_hand=np.full((length, 21, 2), fill_value + 2.0, dtype=np.float32),
        frame_valid_mask=np.asarray(frame_valid_mask, dtype=np.bool_),
    )


def test_collate_pads_variable_length_samples_and_preserves_metadata(
    fake_torch: _FakeTorch,
) -> None:
    first = _item(
        "short",
        length=2,
        fill_value=1.0,
        frame_valid_mask=[True, False],
    )
    second = _item(
        "long",
        length=4,
        fill_value=5.0,
        frame_valid_mask=[False, True, True, False],
    )

    batch = collate_module.collate_processed_pose_samples([first, second])

    assert batch.texts == ["text for short", "text for long"]
    assert batch.sample_ids == ["short", "long"]
    assert batch.splits == ["train", "train"]
    assert batch.fps == [24.0, 24.0]
    assert batch.num_frames == [2, 4]
    assert batch.lengths.tolist() == [2, 4]
    assert batch.body.shape == (2, 4, 25, 2)
    assert batch.left_hand.shape == (2, 4, 21, 2)
    assert batch.right_hand.shape == (2, 4, 21, 2)
    assert batch.body.dtype == fake_torch.float32
    assert batch.padding_mask.dtype == fake_torch.bool
    assert batch.frame_valid_mask.dtype == fake_torch.bool
    assert batch.padding_mask.tolist() == [
        [False, False, True, True],
        [False, False, False, False],
    ]
    assert batch.frame_valid_mask.tolist() == [
        [True, False, False, False],
        [False, True, True, False],
    ]
    assert batch.body[0, 0].mean().item() == pytest.approx(1.0)
    assert batch.body[0, 2].abs().sum().item() == pytest.approx(0.0)
    assert batch.left_hand[1, 0].mean().item() == pytest.approx(6.0)
    assert batch.right_hand[1, 0].mean().item() == pytest.approx(7.0)


def test_collate_rejects_empty_batches() -> None:
    with pytest.raises(ValueError, match="empty"):
        collate_module.collate_processed_pose_samples([])


def test_collate_rejects_item_length_metadata_mismatch(fake_torch: _FakeTorch) -> None:
    item = _item("bad", length=2, fill_value=1.0, frame_valid_mask=[True, True])
    broken_item = ProcessedPoseItem(
        sample_id=item.sample_id,
        split=item.split,
        text=item.text,
        fps=item.fps,
        num_frames=3,
        sample_path=item.sample_path,
        sample_path_value=item.sample_path_value,
        body=item.body,
        left_hand=item.left_hand,
        right_hand=item.right_hand,
        frame_valid_mask=item.frame_valid_mask,
    )

    with pytest.raises(ValueError, match="num_frames"):
        collate_module.collate_processed_pose_samples([broken_item])
