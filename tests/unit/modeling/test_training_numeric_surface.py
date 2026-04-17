"""Sprint 3 mask-aware training numeric surface tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")

from text_to_sign_production.modeling.data import (  # noqa: E402
    SPRINT3_TARGET_CHANNEL_SHAPES,
    SPRINT3_TARGET_CHANNELS,
)
from text_to_sign_production.modeling.models.baseline import BaselinePoseOutput  # noqa: E402
from text_to_sign_production.modeling.training.losses import masked_pose_mse_loss  # noqa: E402
from text_to_sign_production.modeling.training.masking import (  # noqa: E402
    build_effective_frame_mask,
)
from text_to_sign_production.modeling.training.metrics import (  # noqa: E402
    masked_average_keypoint_l2_error,
)

pytestmark = pytest.mark.unit


def _surface(
    *,
    batch_size: int = 1,
    frames: int = 3,
    fill_value: float = 0.0,
    dtype: Any | None = None,
) -> dict[str, Any]:
    resolved_dtype = torch.float32 if dtype is None else dtype
    return {
        channel: torch.full(
            (batch_size, frames, *SPRINT3_TARGET_CHANNEL_SHAPES[channel]),
            fill_value,
            dtype=resolved_dtype,
        )
        for channel in SPRINT3_TARGET_CHANNELS
    }


def _default_masks() -> tuple[Any, Any]:
    padding_mask = torch.tensor([[False, False, True]], dtype=torch.bool)
    frame_valid_mask = torch.tensor([[True, False, False]], dtype=torch.bool)
    return padding_mask, frame_valid_mask


def test_build_effective_frame_mask_combines_padding_and_frame_validity() -> None:
    padding_mask = torch.tensor(
        [
            [False, False, True, True],
            [False, False, False, True],
        ],
        dtype=torch.bool,
    )
    frame_valid_mask = torch.tensor(
        [
            [True, False, False, False],
            [False, True, True, False],
        ],
        dtype=torch.bool,
    )

    effective_mask = build_effective_frame_mask(
        padding_mask=padding_mask,
        frame_valid_mask=frame_valid_mask,
    )

    assert torch.equal(
        effective_mask,
        torch.tensor(
            [
                [True, False, False, False],
                [False, True, True, False],
            ],
            dtype=torch.bool,
        ),
    )


def test_build_effective_frame_mask_rejects_malformed_masks() -> None:
    padding_mask = torch.zeros((1, 2), dtype=torch.float32)
    frame_valid_mask = torch.zeros((1, 2), dtype=torch.bool)

    with pytest.raises(ValueError, match="dtype"):
        build_effective_frame_mask(
            padding_mask=padding_mask,
            frame_valid_mask=frame_valid_mask,
        )

    with pytest.raises(ValueError, match="shape"):
        build_effective_frame_mask(
            padding_mask=torch.zeros((1, 2), dtype=torch.bool),
            frame_valid_mask=torch.zeros((1, 3), dtype=torch.bool),
        )


def test_build_effective_frame_mask_rejects_padding_marked_valid() -> None:
    padding_mask = torch.tensor([[False, True]], dtype=torch.bool)
    frame_valid_mask = torch.tensor([[True, True]], dtype=torch.bool)

    with pytest.raises(ValueError, match="padding frames"):
        build_effective_frame_mask(
            padding_mask=padding_mask,
            frame_valid_mask=frame_valid_mask,
        )


def test_masked_pose_mse_loss_ignores_invalid_and_padded_frames_and_aggregates_channels() -> None:
    padding_mask, frame_valid_mask = _default_masks()
    predictions = _surface()
    targets = _surface()
    targets["body"][0, 0].fill_(1.0)
    targets["left_hand"][0, 0].fill_(2.0)
    targets["right_hand"][0, 0].fill_(3.0)
    for channel in SPRINT3_TARGET_CHANNELS:
        targets[channel][0, 1].fill_(1000.0)
        targets[channel][0, 2].fill_(1000.0)
    predictions["face"] = torch.full((1, 3, 70, 2), 999.0)
    targets["face"] = torch.zeros((1, 3, 70, 2))

    loss = masked_pose_mse_loss(
        predictions=predictions,
        targets=targets,
        padding_mask=padding_mask,
        frame_valid_mask=frame_valid_mask,
    )

    expected_numerator = (25 * 2 * 1.0) + (21 * 2 * 4.0) + (21 * 2 * 9.0)
    expected_denominator = (25 * 2) + (21 * 2) + (21 * 2)
    expected = expected_numerator / expected_denominator
    assert loss.item() == pytest.approx(expected)


def test_masked_average_keypoint_l2_error_averages_keypoint_distances_across_channels() -> None:
    padding_mask, frame_valid_mask = _default_masks()
    predictions = _surface()
    targets = _surface()
    targets["body"][0, 0, :, 0].fill_(3.0)
    targets["body"][0, 0, :, 1].fill_(4.0)
    targets["left_hand"][0, 0, :, 1].fill_(2.0)
    targets["right_hand"][0, 0, :, 0].fill_(1.0)
    for channel in SPRINT3_TARGET_CHANNELS:
        targets[channel][0, 1].fill_(1000.0)
        targets[channel][0, 2].fill_(1000.0)

    metric = masked_average_keypoint_l2_error(
        predictions=predictions,
        targets=targets,
        padding_mask=padding_mask,
        frame_valid_mask=frame_valid_mask,
    )

    expected = ((25 * 5.0) + (21 * 2.0) + (21 * 1.0)) / (25 + 21 + 21)
    assert metric.item() == pytest.approx(expected)


def test_numeric_surface_accepts_phase3_baseline_output_and_object_targets() -> None:
    prediction_channels = _surface(frames=1)
    predictions = BaselinePoseOutput(
        body=prediction_channels["body"],
        left_hand=prediction_channels["left_hand"],
        right_hand=prediction_channels["right_hand"],
    )
    targets = SimpleNamespace(**_surface(frames=1))
    padding_mask = torch.tensor([[False]], dtype=torch.bool)
    frame_valid_mask = torch.tensor([[True]], dtype=torch.bool)

    loss = masked_pose_mse_loss(
        predictions=predictions,
        targets=targets,
        padding_mask=padding_mask,
        frame_valid_mask=frame_valid_mask,
    )
    metric = masked_average_keypoint_l2_error(
        predictions=predictions,
        targets=targets,
        padding_mask=padding_mask,
        frame_valid_mask=frame_valid_mask,
    )

    assert loss.item() == pytest.approx(0.0)
    assert metric.item() == pytest.approx(0.0)


def test_numeric_surface_rejects_missing_required_channel() -> None:
    padding_mask, frame_valid_mask = _default_masks()
    predictions = _surface()
    targets = _surface()
    del predictions["left_hand"]

    with pytest.raises(ValueError, match="left_hand"):
        masked_pose_mse_loss(
            predictions=predictions,
            targets=targets,
            padding_mask=padding_mask,
            frame_valid_mask=frame_valid_mask,
        )


def test_numeric_surface_rejects_channel_batch_time_shape_mismatch() -> None:
    padding_mask, frame_valid_mask = _default_masks()
    predictions = _surface()
    targets = _surface()
    targets["body"] = torch.zeros((1, 2, 25, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="batch/time"):
        masked_pose_mse_loss(
            predictions=predictions,
            targets=targets,
            padding_mask=padding_mask,
            frame_valid_mask=frame_valid_mask,
        )


def test_numeric_surface_rejects_wrong_channel_pose_shape() -> None:
    padding_mask, frame_valid_mask = _default_masks()
    predictions = _surface()
    targets = _surface()
    predictions["body"] = torch.zeros((1, 3, 24, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="expected"):
        masked_average_keypoint_l2_error(
            predictions=predictions,
            targets=targets,
            padding_mask=padding_mask,
            frame_valid_mask=frame_valid_mask,
        )


def test_numeric_surface_rejects_non_floating_pose_tensor() -> None:
    padding_mask, frame_valid_mask = _default_masks()
    predictions = _surface()
    targets = _surface()
    predictions["body"] = torch.zeros((1, 3, 25, 2), dtype=torch.long)

    with pytest.raises(ValueError, match="floating"):
        masked_pose_mse_loss(
            predictions=predictions,
            targets=targets,
            padding_mask=padding_mask,
            frame_valid_mask=frame_valid_mask,
        )


def test_masked_loss_and_metric_reject_zero_valid_contributing_frames() -> None:
    predictions = _surface(frames=2)
    targets = _surface(frames=2)
    padding_mask = torch.tensor([[False, True]], dtype=torch.bool)
    frame_valid_mask = torch.tensor([[False, False]], dtype=torch.bool)

    with pytest.raises(ValueError, match="zero valid"):
        masked_pose_mse_loss(
            predictions=predictions,
            targets=targets,
            padding_mask=padding_mask,
            frame_valid_mask=frame_valid_mask,
        )
    with pytest.raises(ValueError, match="zero valid"):
        masked_average_keypoint_l2_error(
            predictions=predictions,
            targets=targets,
            padding_mask=padding_mask,
            frame_valid_mask=frame_valid_mask,
        )
