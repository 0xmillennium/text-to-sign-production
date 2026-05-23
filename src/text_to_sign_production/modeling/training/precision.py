"""Shared mixed-precision policy helpers for model providers."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import ContextManager

import torch


@dataclass(frozen=True, slots=True)
class ResolvedPrecisionPolicy:
    requested_policy: str
    resolved_policy: str
    autocast_enabled: bool
    autocast_dtype: torch.dtype | None
    device_type: str
    precision_applied: bool = False

    @property
    def precision_policy(self) -> str:
        return self.resolved_policy

    def with_precision_applied(self, precision_applied: bool) -> "ResolvedPrecisionPolicy":
        return replace(self, precision_applied=bool(precision_applied))

    def to_metadata(self, *, precision_applied: bool | None = None) -> dict[str, object]:
        requested_applied = (
            self.precision_applied if precision_applied is None else bool(precision_applied)
        )
        applied = requested_applied and self.autocast_enabled
        return {
            "precision_policy": self.resolved_policy,
            "requested_precision_policy": self.requested_policy,
            "resolved_precision_policy": self.resolved_policy,
            "autocast_enabled": self.autocast_enabled,
            "autocast_dtype": (
                None if self.autocast_dtype is None else str(self.autocast_dtype)
            ),
            "precision_applied": applied,
            "precision_device_type": self.device_type,
        }


def resolve_precision_policy(policy: str, device: torch.device) -> ResolvedPrecisionPolicy:
    """Resolve a requested precision policy for one torch device."""

    requested_policy = policy
    if policy not in {"auto", "fp32", "bf16", "fp16"}:
        raise ValueError("precision policy must be one of {'auto', 'fp32', 'bf16', 'fp16'}")
    device = torch.device(device)
    if device.type != "cuda":
        return ResolvedPrecisionPolicy(
            requested_policy=policy,
            resolved_policy="fp32",
            autocast_enabled=False,
            autocast_dtype=None,
            device_type=device.type,
        )
    if policy == "auto":
        policy = "bf16" if torch.cuda.is_bf16_supported() else "fp32"
    if policy == "fp32":
        return ResolvedPrecisionPolicy(
            requested_policy=requested_policy,
            resolved_policy="fp32",
            autocast_enabled=False,
            autocast_dtype=None,
            device_type=device.type,
        )
    dtype = torch.bfloat16 if policy == "bf16" else torch.float16
    if policy == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 precision policy requires CUDA bf16 support")
    return ResolvedPrecisionPolicy(
        requested_policy=requested_policy,
        resolved_policy=policy,
        autocast_enabled=True,
        autocast_dtype=dtype,
        device_type=device.type,
    )


def autocast_context(policy: ResolvedPrecisionPolicy) -> ContextManager[object]:
    """Return an autocast context or no-op for a resolved policy."""

    if not policy.autocast_enabled or policy.autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=policy.device_type, dtype=policy.autocast_dtype)


__all__ = [
    "ResolvedPrecisionPolicy",
    "autocast_context",
    "resolve_precision_policy",
]
