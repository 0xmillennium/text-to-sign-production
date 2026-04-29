"""Torch-backed Sprint 3 test doubles and builders."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from text_to_sign_production.data.constants import PROCESSED_SCHEMA_VERSION
from text_to_sign_production.modeling.backbones import TextBackboneOutput
from text_to_sign_production.modeling.data import (
    M0_TARGET_CHANNEL_SHAPES,
    ProcessedPoseBatch,
    ProcessedPoseItem,
)
from text_to_sign_production.modeling.models import BaselineTextToPoseModel
from text_to_sign_production.modeling.training.checkpointing import (
    CheckpointMetrics,
    save_training_checkpoint,
)


class DummyTextBackbone:
    """Small backbone fake that avoids Hugging Face downloads."""

    output_dim = 4

    def __call__(
        self,
        texts: Sequence[str],
        *,
        device: Any | None = None,
    ) -> TextBackboneOutput:
        resolved_device = torch.device("cpu") if device is None else torch.device(device)
        batch_size = len(texts)
        pooled_embedding = torch.ones((batch_size, self.output_dim), device=resolved_device)
        return TextBackboneOutput(
            token_embeddings=pooled_embedding.unsqueeze(1),
            pooled_embedding=pooled_embedding,
            attention_mask=torch.ones((batch_size, 1), dtype=torch.long, device=resolved_device),
        )


class DeterministicTextBackbone:
    """Backbone fake that records devices and returns deterministic embeddings."""

    output_dim = 4

    def __init__(self) -> None:
        self.seen_devices: list[torch.device] = []

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
        return TextBackboneOutput(
            token_embeddings=token_embeddings,
            pooled_embedding=pooled_embedding,
            attention_mask=torch.ones((batch_size, 2), dtype=torch.long, device=resolved_device),
        )


def build_dummy_baseline_model(
    *,
    decoder_hidden_dim: int = 8,
    latent_dim: int = 6,
) -> BaselineTextToPoseModel:
    """Build the tiny baseline model used by torch-dependent tests."""

    _ = latent_dim
    return BaselineTextToPoseModel(
        DummyTextBackbone(),
        decoder_hidden_dim=decoder_hidden_dim,
    )


def processed_batch(
    *,
    frame_valid_mask: list[bool] | None = None,
) -> ProcessedPoseBatch:
    """Return a one-sample processed batch with Sprint 3 target channels."""

    resolved_frame_valid_mask = [True, True] if frame_valid_mask is None else frame_valid_mask
    return ProcessedPoseBatch(
        texts=["hello world"],
        sample_ids=["sample"],
        splits=["train"],
        lengths=torch.tensor([2], dtype=torch.long),
        body=pose_surface("body", fill_value=0.5),
        body_confidence=confidence_surface("body"),
        left_hand=pose_surface("left_hand", fill_value=-0.25),
        left_hand_confidence=confidence_surface("left_hand"),
        right_hand=pose_surface("right_hand", fill_value=0.75),
        right_hand_confidence=confidence_surface("right_hand"),
        face=pose_surface("face", fill_value=0.125),
        face_confidence=confidence_surface("face"),
        padding_mask=torch.tensor([[False, False]], dtype=torch.bool),
        frame_valid_mask=torch.tensor([resolved_frame_valid_mask], dtype=torch.bool),
        people_per_frame=torch.ones((1, 2), dtype=torch.int16),
        selected_person_indices=[0],
        processed_schema_versions=[PROCESSED_SCHEMA_VERSION],
        fps=[24.0],
        num_frames=[2],
    )


def pose_surface(
    channel: str,
    *,
    batch_size: int = 1,
    frames: int = 2,
    fill_value: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a tensor matching a Sprint 3 target channel."""

    return torch.full(
        (batch_size, frames, *M0_TARGET_CHANNEL_SHAPES[channel]),
        fill_value,
        dtype=dtype,
    )


def confidence_surface(
    channel: str,
    *,
    batch_size: int = 1,
    frames: int = 2,
    fill_value: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a tensor matching a full-BFH confidence channel."""

    keypoint_count = M0_TARGET_CHANNEL_SHAPES[channel][0]
    return torch.full(
        (batch_size, frames, keypoint_count),
        fill_value,
        dtype=dtype,
    )


def processed_pose_item(
    *,
    sample_id: str = "sample one",
    split: str = "val",
    text: str = "A raw English sentence.",
) -> ProcessedPoseItem:
    """Return a tiny processed item for qualitative artifact tests."""

    return ProcessedPoseItem(
        sample_id=sample_id,
        split=split,
        text=text,
        fps=24.0,
        num_frames=2,
        sample_path=Path(f"/tmp/{sample_id}.npz"),
        sample_path_value=f"data/processed/v1/samples/{split}/{sample_id}.npz",
        processed_schema_version=PROCESSED_SCHEMA_VERSION,
        selected_person_index=0,
        body=np.full((2, 25, 2), 1.0, dtype=np.float32),
        body_confidence=np.ones((2, 25), dtype=np.float32),
        left_hand=np.full((2, 21, 2), 2.0, dtype=np.float32),
        left_hand_confidence=np.ones((2, 21), dtype=np.float32),
        right_hand=np.full((2, 21, 2), 3.0, dtype=np.float32),
        right_hand_confidence=np.ones((2, 21), dtype=np.float32),
        face=np.full((2, 70, 2), 4.0, dtype=np.float32),
        face_confidence=np.ones((2, 70), dtype=np.float32),
        frame_valid_mask=np.asarray([True, False], dtype=np.bool_),
        people_per_frame=np.ones((2,), dtype=np.int16),
    )


def write_dummy_training_checkpoint(path: Path) -> None:
    """Write a strict checkpoint for the dummy baseline model."""

    model = build_dummy_baseline_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        epoch=1,
        role="best",
        config_summary={"backbone": {"name": "dummy"}},
        config_hash="dummy",
        backbone_name="dummy",
        model_revision="dummy",
        seed=5,
        metrics=CheckpointMetrics(
            train_loss=1.0,
            validation_loss=2.0,
            validation_metric=3.0,
        ),
        run_mode="train",
        best_metric=0.0,
        best_epoch=1,
        target_standardization=None,
    )
