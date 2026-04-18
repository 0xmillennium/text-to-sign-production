"""Sprint 3 baseline training and validation loop tests."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest

torch: Any = pytest.importorskip("torch")

from tests.support.modeling_torch import (  # noqa: E402
    build_dummy_baseline_model,
    pose_surface,
    processed_batch,
)
from text_to_sign_production.modeling.data import ProcessedPoseBatch  # noqa: E402
from text_to_sign_production.modeling.training.evaluate import (  # noqa: E402
    move_batch_to_device,
    run_validation_epoch,
    validation_step,
)
from text_to_sign_production.modeling.training.train import (  # noqa: E402
    run_training_epoch,
    set_reproducibility_seed,
    train_step,
)

pytestmark = pytest.mark.unit


def _batch() -> ProcessedPoseBatch:
    return processed_batch()


def _batch_with_frame_valid_mask(frame_valid_mask: list[bool]) -> ProcessedPoseBatch:
    return ProcessedPoseBatch(
        texts=["hello world"],
        sample_ids=["sample"],
        splits=["train"],
        lengths=torch.tensor([2], dtype=torch.long),
        body=pose_surface("body", fill_value=0.5),
        left_hand=pose_surface("left_hand", fill_value=-0.25),
        right_hand=pose_surface("right_hand", fill_value=0.75),
        padding_mask=torch.tensor([[False, False]], dtype=torch.bool),
        frame_valid_mask=torch.tensor([frame_valid_mask], dtype=torch.bool),
        fps=[24.0],
        num_frames=[2],
    )


def test_train_step_computes_masked_loss_and_updates_parameters() -> None:
    set_reproducibility_seed(7)
    model = build_dummy_baseline_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    result = train_step(model, _batch(), optimizer)

    assert result.loss > 0.0
    assert result.valid_frame_count == 2
    assert any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, model.parameters(), strict=True)
    )


def test_validation_step_uses_phase4_loss_and_metric_surface() -> None:
    set_reproducibility_seed(11)
    model = build_dummy_baseline_model()

    result = validation_step(model, _batch())

    assert result.valid_frame_count == 2
    assert result.loss >= 0.0
    assert result.metric >= 0.0


def test_training_epoch_skips_zero_valid_batches_and_trains_on_valid_batches() -> None:
    set_reproducibility_seed(17)
    model = build_dummy_baseline_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    result = run_training_epoch(
        model,
        [_batch_with_frame_valid_mask([False, False]), _batch()],
        optimizer,
        device=torch.device("cpu"),
    )

    assert result.valid_frame_count == 2
    assert result.loss > 0.0


def test_training_epoch_reports_zero_valid_epoch_after_skipping_batches() -> None:
    model = build_dummy_baseline_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="Training epoch has zero valid"):
        run_training_epoch(
            model,
            [_batch_with_frame_valid_mask([False, False])],
            optimizer,
            device=torch.device("cpu"),
        )


def test_validation_epoch_skips_zero_valid_batches_and_validates_remaining_batches() -> None:
    set_reproducibility_seed(19)
    model = build_dummy_baseline_model()

    result = run_validation_epoch(
        model,
        [_batch_with_frame_valid_mask([False, False]), _batch()],
        device=torch.device("cpu"),
    )

    assert result.valid_frame_count == 2
    assert result.loss >= 0.0
    assert result.metric >= 0.0


def test_validation_epoch_reports_zero_valid_epoch_after_skipping_batches() -> None:
    model = build_dummy_baseline_model()

    with pytest.raises(ValueError, match="Validation epoch has zero valid"):
        run_validation_epoch(
            model,
            [_batch_with_frame_valid_mask([False, False])],
            device=torch.device("cpu"),
        )


def test_move_batch_to_device_preserves_metadata_and_moves_tensors() -> None:
    batch = _batch()

    moved = move_batch_to_device(batch, torch.device("cpu"))

    assert moved.texts is batch.texts
    assert moved.sample_ids is batch.sample_ids
    assert moved.fps is batch.fps
    assert moved.body.device == torch.device("cpu")
    assert moved.left_hand.device == torch.device("cpu")
    assert moved.right_hand.device == torch.device("cpu")
    assert moved.padding_mask.device == torch.device("cpu")


def test_set_reproducibility_seed_resets_local_generators() -> None:
    set_reproducibility_seed(123)
    first_values = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(1).item()),
    )

    set_reproducibility_seed(123)
    second_values = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(1).item()),
    )

    assert second_values == pytest.approx(first_values)
