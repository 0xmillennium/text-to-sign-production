"""Lazy exports for Sprint 3 baseline training utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .losses import masked_pose_mse_loss
    from .masking import build_effective_frame_mask
    from .metrics import masked_average_keypoint_l2_error

__all__ = [
    "build_effective_frame_mask",
    "masked_average_keypoint_l2_error",
    "masked_pose_mse_loss",
]


def __getattr__(name: str) -> Any:
    if name == "build_effective_frame_mask":
        from .masking import build_effective_frame_mask

        return build_effective_frame_mask
    if name == "masked_pose_mse_loss":
        from .losses import masked_pose_mse_loss

        return masked_pose_mse_loss
    if name == "masked_average_keypoint_l2_error":
        from .metrics import masked_average_keypoint_l2_error

        return masked_average_keypoint_l2_error
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
