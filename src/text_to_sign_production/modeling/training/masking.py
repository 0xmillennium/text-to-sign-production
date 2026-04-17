"""Mask construction utilities for Sprint 3 baseline numeric surfaces."""

from __future__ import annotations

import torch


def build_effective_frame_mask(
    padding_mask: torch.Tensor,
    frame_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the frame-level contribution mask for losses and metrics.

    ``padding_mask`` keeps the Phase 2 polarity where ``True`` means padding and must be
    excluded. ``frame_valid_mask`` keeps the processed-data polarity where ``True`` means a real
    frame is valid and may contribute. A timestep contributes only when it is not padding and the
    real frame is valid. If the two masks are already on different devices, ``padding_mask`` is
    aligned to ``frame_valid_mask.device`` after dtype, rank, and shape validation and before
    padded-frame validation and combination, matching the baseline model's optional mask validation
    behavior.
    """

    _validate_bool_frame_mask(padding_mask, field_name="padding_mask")
    _validate_bool_frame_mask(frame_valid_mask, field_name="frame_valid_mask")
    if frame_valid_mask.shape != padding_mask.shape:
        raise ValueError("frame_valid_mask shape must match padding_mask.")
    aligned_padding_mask = padding_mask.to(device=frame_valid_mask.device)
    if bool(torch.any(frame_valid_mask & aligned_padding_mask).item()):
        raise ValueError("frame_valid_mask must not mark padding frames as valid.")
    return (~aligned_padding_mask) & frame_valid_mask


def _validate_bool_frame_mask(mask: torch.Tensor, *, field_name: str) -> None:
    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"{field_name} must be a torch.Tensor.")
    if mask.dtype != torch.bool:
        raise ValueError(f"{field_name} must have dtype torch.bool.")
    if mask.ndim != 2:
        raise ValueError(f"{field_name} must have shape [B, T].")
