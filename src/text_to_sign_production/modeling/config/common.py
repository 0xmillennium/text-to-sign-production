"""Small reusable configuration fragments for future modeling providers."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float
    extra: Mapping[str, object]

    def __post_init__(self) -> None:
        _require_text(self.name, "optimizer name")
        if not _finite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite.")
        if not _finite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative and finite.")
        object.__setattr__(self, "extra", _json_mapping(self.extra, "optimizer extra"))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "extra": dict(self.extra),
        }


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    name: str
    warmup_steps: int | None
    total_steps: int | None
    extra: Mapping[str, object]

    def __post_init__(self) -> None:
        _require_text(self.name, "scheduler name")
        _optional_non_negative_int(self.warmup_steps, "warmup_steps")
        _optional_non_negative_int(self.total_steps, "total_steps")
        if (
            self.warmup_steps is not None
            and self.total_steps is not None
            and self.warmup_steps > self.total_steps
        ):
            raise ValueError("warmup_steps cannot exceed total_steps.")
        object.__setattr__(self, "extra", _json_mapping(self.extra, "scheduler extra"))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True, slots=True)
class TrainingLoopConfig:
    max_epochs: int | None
    batch_size: int
    gradient_clip_norm: float | None
    seed: int | None
    num_workers: int
    device: str

    def __post_init__(self) -> None:
        if self.max_epochs is not None and (
            not isinstance(self.max_epochs, int)
            or isinstance(self.max_epochs, bool)
            or self.max_epochs <= 0
        ):
            raise ValueError("max_epochs must be positive when provided.")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.gradient_clip_norm is not None and (
            not _finite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0
        ):
            raise ValueError("gradient_clip_norm must be positive and finite when provided.")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise ValueError("seed must be an integer when provided.")
        if not isinstance(self.num_workers, int) or isinstance(self.num_workers, bool) or self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        _require_text(self.device, "device")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "gradient_clip_norm": self.gradient_clip_norm,
            "seed": self.seed,
            "num_workers": self.num_workers,
            "device": self.device,
        }


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty.")


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def _optional_non_negative_int(value: int | None, name: str) -> None:
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value < 0
    ):
        raise ValueError(f"{name} must be non-negative when provided.")


def _json_mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys.")
    resolved = dict(value)
    try:
        json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain JSON-serializable values.") from exc
    return MappingProxyType(resolved)


__all__ = ["OptimizerConfig", "SchedulerConfig", "TrainingLoopConfig"]
