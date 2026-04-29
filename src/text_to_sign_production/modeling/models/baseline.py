"""Conservative M0 direct text-to-full-BFH pose model surface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from text_to_sign_production.modeling.backbones.base import TextBackbone
from text_to_sign_production.modeling.data import M0_TARGET_CHANNEL_SHAPES

from .decoder import SimplePoseDecoder
from .heads import PoseChannelHead


@dataclass(frozen=True, slots=True)
class BaselinePoseOutput:
    """Channel-separated continuous pose predictions from the baseline model.

    Masks remain part of the input batch contract. The model validates mask compatibility but does
    not return masks as predictions.
    """

    body: torch.Tensor
    left_hand: torch.Tensor
    right_hand: torch.Tensor
    face: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return the public channel-separated output contract."""

        return {
            "body": self.body,
            "left_hand": self.left_hand,
            "right_hand": self.right_hand,
            "face": self.face,
        }


class BaselineTextToPoseModel(nn.Module):
    """M0 baseline model from transcript text directly to continuous full-BFH channels."""

    def __init__(
        self,
        backbone: TextBackbone,
        *,
        decoder_hidden_dim: int = 256,
        decoder_layers: int = 2,
        decoder_dropout: float = 0.0,
        frame_position_encoding_dim: int = 0,
    ) -> None:
        super().__init__()
        if backbone.output_dim <= 0:
            raise ValueError("backbone.output_dim must be positive.")
        if decoder_hidden_dim <= 0:
            raise ValueError("decoder_hidden_dim must be positive.")

        self.backbone: TextBackbone = backbone
        self.decoder = SimplePoseDecoder(
            text_embedding_dim=backbone.output_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_layers,
            dropout=decoder_dropout,
            frame_position_encoding_dim=frame_position_encoding_dim,
        )
        body_keypoints, body_coordinate_dim = M0_TARGET_CHANNEL_SHAPES["body"]
        left_hand_keypoints, left_hand_coordinate_dim = M0_TARGET_CHANNEL_SHAPES["left_hand"]
        right_hand_keypoints, right_hand_coordinate_dim = M0_TARGET_CHANNEL_SHAPES["right_hand"]
        face_keypoints, face_coordinate_dim = M0_TARGET_CHANNEL_SHAPES["face"]
        self.body_head = PoseChannelHead(
            input_dim=decoder_hidden_dim,
            keypoints=body_keypoints,
            coordinate_dim=body_coordinate_dim,
        )
        self.left_hand_head = PoseChannelHead(
            input_dim=decoder_hidden_dim,
            keypoints=left_hand_keypoints,
            coordinate_dim=left_hand_coordinate_dim,
        )
        self.right_hand_head = PoseChannelHead(
            input_dim=decoder_hidden_dim,
            keypoints=right_hand_keypoints,
            coordinate_dim=right_hand_coordinate_dim,
        )
        self.face_head = PoseChannelHead(
            input_dim=decoder_hidden_dim,
            keypoints=face_keypoints,
            coordinate_dim=face_coordinate_dim,
        )

    def forward(self, batch: Any) -> BaselinePoseOutput:
        """Predict channel-separated full-BFH pose tensors from a processed batch surface.

        ``frame_valid_mask`` is validated when present so later masked training can trust the
        batch contract, but it remains on the input batch rather than the prediction output.
        """

        texts = _extract_texts(batch)
        padding_mask = _extract_bool_tensor(batch, "padding_mask")
        lengths = _extract_lengths(
            batch,
            batch_size=len(texts),
            target_frames=padding_mask.shape[1],
        )
        _validate_padding_mask(padding_mask, batch_size=len(texts), lengths=lengths)
        _validate_optional_frame_valid_mask(batch, padding_mask=padding_mask)

        model_device = _module_parameter_device(self)
        encoded = self.backbone(texts, device=model_device)
        pooled_embedding = encoded.pooled_embedding
        if not isinstance(pooled_embedding, torch.Tensor):
            raise ValueError("backbone pooled_embedding must be a torch.Tensor.")
        if pooled_embedding.ndim != 2:
            raise ValueError("backbone pooled_embedding must have shape [B, D].")
        if pooled_embedding.shape[0] != len(texts):
            raise ValueError("backbone pooled_embedding batch dimension must match texts.")
        if pooled_embedding.shape[1] != self.backbone.output_dim:
            raise ValueError("backbone pooled_embedding width must match backbone.output_dim.")
        if pooled_embedding.device != model_device:
            raise ValueError("backbone pooled_embedding device must match the model device.")

        model_lengths = lengths.to(device=model_device)
        model_padding_mask = padding_mask.to(device=model_device)
        decoder_features = self.decoder(
            pooled_embedding,
            lengths=model_lengths,
            target_frames=padding_mask.shape[1],
        )

        body = _zero_padded_frames(self.body_head(decoder_features), model_padding_mask)
        left_hand = _zero_padded_frames(self.left_hand_head(decoder_features), model_padding_mask)
        right_hand = _zero_padded_frames(
            self.right_hand_head(decoder_features),
            model_padding_mask,
        )
        face = _zero_padded_frames(self.face_head(decoder_features), model_padding_mask)

        return BaselinePoseOutput(
            body=body,
            left_hand=left_hand,
            right_hand=right_hand,
            face=face,
        )


def _get_batch_field(batch: Any, field_name: str) -> Any:
    if isinstance(batch, Mapping):
        if field_name not in batch:
            raise ValueError(f"batch is missing required field {field_name!r}.")
        return batch[field_name]
    if not hasattr(batch, field_name):
        raise ValueError(f"batch is missing required field {field_name!r}.")
    return getattr(batch, field_name)


def _has_batch_field(batch: Any, field_name: str) -> bool:
    if isinstance(batch, Mapping):
        return field_name in batch
    return hasattr(batch, field_name)


def _extract_texts(batch: Any) -> list[str]:
    raw_texts = _get_batch_field(batch, "texts")
    if isinstance(raw_texts, str) or not isinstance(raw_texts, Sequence):
        raise ValueError("batch field 'texts' must be a sequence of strings.")

    texts = list(raw_texts)
    if not texts:
        raise ValueError("batch field 'texts' must not be empty.")
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("batch field 'texts' must contain only strings.")
    return texts


def _extract_bool_tensor(batch: Any, field_name: str) -> torch.Tensor:
    value = _get_batch_field(batch, field_name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"batch field {field_name!r} must be a torch.Tensor.")
    if value.dtype != torch.bool:
        raise ValueError(f"batch field {field_name!r} must have dtype torch.bool.")
    if value.ndim != 2:
        raise ValueError(f"batch field {field_name!r} must have shape [B, T].")
    return value


def _extract_lengths(batch: Any, *, batch_size: int, target_frames: int) -> torch.Tensor:
    raw_lengths = _get_batch_field(batch, "lengths")
    if isinstance(raw_lengths, torch.Tensor):
        lengths = raw_lengths.to(dtype=torch.long)
    else:
        lengths = torch.as_tensor(raw_lengths, dtype=torch.long)

    if lengths.ndim != 1:
        raise ValueError("batch field 'lengths' must have shape [B].")
    if lengths.shape[0] != batch_size:
        raise ValueError("batch field 'lengths' batch dimension must match texts.")
    if bool(torch.any(lengths < 0).item()):
        raise ValueError("batch field 'lengths' must not contain negative values.")
    if bool(torch.any(lengths > target_frames).item()):
        raise ValueError("batch field 'lengths' cannot exceed padding_mask width.")
    return lengths


def _validate_padding_mask(
    padding_mask: torch.Tensor,
    *,
    batch_size: int,
    lengths: torch.Tensor,
) -> None:
    if padding_mask.shape[0] != batch_size:
        raise ValueError("padding_mask batch dimension must match texts.")
    non_padding_counts = (~padding_mask).sum(dim=1).to(dtype=torch.long)
    expected_lengths = lengths.to(device=padding_mask.device)
    if not torch.equal(non_padding_counts, expected_lengths):
        raise ValueError(
            "padding_mask must use True for padding and match the explicit lengths field."
        )


def _validate_optional_frame_valid_mask(batch: Any, *, padding_mask: torch.Tensor) -> None:
    if not _has_batch_field(batch, "frame_valid_mask"):
        return
    frame_valid_mask = _extract_bool_tensor(batch, "frame_valid_mask")
    if frame_valid_mask.shape != padding_mask.shape:
        raise ValueError("frame_valid_mask shape must match padding_mask.")
    if bool(torch.any(frame_valid_mask & padding_mask.to(device=frame_valid_mask.device)).item()):
        raise ValueError("frame_valid_mask must not mark padding frames as valid.")


def _zero_padded_frames(prediction: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    return prediction.masked_fill(padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)


def _module_parameter_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")
