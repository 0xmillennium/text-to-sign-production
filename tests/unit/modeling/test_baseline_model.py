"""Sprint 3 baseline model surface tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")

from text_to_sign_production.modeling.backbones import TextBackboneOutput  # noqa: E402
from text_to_sign_production.modeling.data import SPRINT3_TARGET_CHANNEL_SHAPES  # noqa: E402
from text_to_sign_production.modeling.models.baseline import (  # noqa: E402
    BaselineTextToPoseModel,
)

pytestmark = pytest.mark.unit


class _DeterministicBackbone:
    output_dim = 4

    def __init__(self) -> None:
        self.seen_devices: list[Any] = []

    def __call__(
        self,
        texts: Sequence[str],
        *,
        device: Any | None = None,
    ) -> TextBackboneOutput:
        resolved_device = torch.device("cpu") if device is None else torch.device(device)
        self.seen_devices.append(resolved_device)
        batch_size = len(texts)
        pooled_embedding = torch.arange(
            batch_size * self.output_dim,
            dtype=torch.float32,
            device=resolved_device,
        ).view(batch_size, self.output_dim)
        token_embeddings = pooled_embedding.unsqueeze(1).expand(batch_size, 2, self.output_dim)
        attention_mask = torch.ones((batch_size, 2), dtype=torch.long, device=resolved_device)
        return TextBackboneOutput(
            token_embeddings=token_embeddings,
            pooled_embedding=pooled_embedding,
            attention_mask=attention_mask,
        )


def _batch() -> dict[str, object]:
    return {
        "texts": ["short sample", "long sample"],
        "lengths": torch.tensor([2, 4], dtype=torch.long),
        "padding_mask": torch.tensor(
            [
                [False, False, True, True],
                [False, False, False, False],
            ],
            dtype=torch.bool,
        ),
        "frame_valid_mask": torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, True],
            ],
            dtype=torch.bool,
        ),
    }


def test_baseline_model_outputs_channel_separated_shapes() -> None:
    model = BaselineTextToPoseModel(
        _DeterministicBackbone(),
        decoder_hidden_dim=8,
        latent_dim=6,
    )

    output = model(_batch())
    output_dict = output.as_dict()

    assert set(output_dict) == {"body", "left_hand", "right_hand"}
    assert not hasattr(output, "face")
    assert not hasattr(output, "frame_valid_mask")
    assert output.body.shape == (2, 4, 25, 2)
    assert output.left_hand.shape == (2, 4, 21, 2)
    assert output.right_hand.shape == (2, 4, 21, 2)


def test_baseline_model_heads_follow_target_channel_shapes() -> None:
    model = BaselineTextToPoseModel(
        _DeterministicBackbone(),
        decoder_hidden_dim=8,
        latent_dim=6,
    )

    assert (model.body_head.keypoints, model.body_head.coordinate_dim) == (
        SPRINT3_TARGET_CHANNEL_SHAPES["body"]
    )
    assert (model.left_hand_head.keypoints, model.left_hand_head.coordinate_dim) == (
        SPRINT3_TARGET_CHANNEL_SHAPES["left_hand"]
    )
    assert (model.right_hand_head.keypoints, model.right_hand_head.coordinate_dim) == (
        SPRINT3_TARGET_CHANNEL_SHAPES["right_hand"]
    )


def test_baseline_model_uses_model_device_for_backbone_and_masks() -> None:
    backbone = _DeterministicBackbone()
    model = BaselineTextToPoseModel(
        backbone,
        decoder_hidden_dim=8,
        latent_dim=6,
    ).to(torch.device("meta"))

    output = model(_batch())

    assert backbone.seen_devices[-1] == torch.device("meta")
    assert output.body.device.type == "meta"
    assert output.left_hand.device.type == "meta"
    assert output.right_hand.device.type == "meta"


def test_baseline_model_zeroes_padding_frames_with_existing_mask_polarity() -> None:
    model = BaselineTextToPoseModel(
        _DeterministicBackbone(),
        decoder_hidden_dim=8,
        latent_dim=6,
    )

    output = model(_batch())

    assert torch.count_nonzero(output.body[0, 2:]).item() == 0
    assert torch.count_nonzero(output.left_hand[0, 2:]).item() == 0
    assert torch.count_nonzero(output.right_hand[0, 2:]).item() == 0


def test_baseline_model_rejects_missing_texts() -> None:
    model = BaselineTextToPoseModel(
        _DeterministicBackbone(),
        decoder_hidden_dim=8,
        latent_dim=6,
    )
    batch = _batch()
    del batch["texts"]

    with pytest.raises(ValueError, match="texts"):
        model(batch)


def test_baseline_model_rejects_lengths_that_do_not_match_padding_mask() -> None:
    model = BaselineTextToPoseModel(
        _DeterministicBackbone(),
        decoder_hidden_dim=8,
        latent_dim=6,
    )
    batch = _batch()
    batch["lengths"] = torch.tensor([3, 4], dtype=torch.long)

    with pytest.raises(ValueError, match="padding_mask"):
        model(batch)


def test_baseline_model_rejects_frame_valid_mask_marking_padding_valid() -> None:
    model = BaselineTextToPoseModel(
        _DeterministicBackbone(),
        decoder_hidden_dim=8,
        latent_dim=6,
    )
    batch = _batch()
    batch["frame_valid_mask"] = torch.tensor(
        [
            [True, False, True, False],
            [True, True, False, True],
        ],
        dtype=torch.bool,
    )

    with pytest.raises(ValueError, match="padding frames"):
        model(batch)
