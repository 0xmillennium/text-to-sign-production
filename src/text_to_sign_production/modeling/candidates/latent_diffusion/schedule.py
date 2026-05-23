"""Pure diffusion schedule and q-sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from text_to_sign_production.modeling.candidates.latent_diffusion.config import (
    DiffusionScheduleConfig,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.errors import (
    LatentDiffusionError,
)

DIFFUSION_SCHEDULE_SCHEMA_VERSION = "t2sp-diffusion-schedule-v1"


@dataclass(frozen=True, slots=True)
class DiffusionSchedule:
    schema_version: str
    schedule: str
    timesteps: int
    beta_start: float
    beta_end: float
    betas: np.ndarray
    alphas: np.ndarray
    alpha_bars: np.ndarray

    def __post_init__(self) -> None:
        if self.schema_version != DIFFUSION_SCHEDULE_SCHEMA_VERSION:
            raise LatentDiffusionError("diffusion schedule schema_version is unsupported.")
        if self.schedule != "linear_beta":
            raise LatentDiffusionError("diffusion schedule must be 'linear_beta'.")
        if not isinstance(self.timesteps, int) or isinstance(self.timesteps, bool) or self.timesteps <= 1:
            raise LatentDiffusionError("diffusion timesteps must be greater than 1.")
        if not 0.0 < float(self.beta_start) < float(self.beta_end) < 1.0:
            raise LatentDiffusionError("beta range must satisfy 0 < beta_start < beta_end < 1.")
        betas = _array(self.betas, "betas", self.timesteps)
        alphas = _array(self.alphas, "alphas", self.timesteps)
        alpha_bars = _array(self.alpha_bars, "alpha_bars", self.timesteps)
        if not np.allclose(alphas, 1.0 - betas):
            raise LatentDiffusionError("alphas must equal 1 - betas.")
        if not np.all((betas > 0.0) & (betas < 1.0)):
            raise LatentDiffusionError("betas must be in (0, 1).")
        if not np.all(np.diff(alpha_bars) < 0.0):
            raise LatentDiffusionError("alpha_bars must be strictly monotonically decreasing.")
        for array in (betas, alphas, alpha_bars):
            array.setflags(write=False)
        object.__setattr__(self, "betas", betas)
        object.__setattr__(self, "alphas", alphas)
        object.__setattr__(self, "alpha_bars", alpha_bars)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "schedule": self.schedule,
            "timesteps": self.timesteps,
            "beta_start": float(self.beta_start),
            "beta_end": float(self.beta_end),
            "betas": self.betas.tolist(),
            "alphas": self.alphas.tolist(),
            "alpha_bars": self.alpha_bars.tolist(),
        }


def build_linear_beta_schedule(config: DiffusionScheduleConfig) -> DiffusionSchedule:
    """Build a validated linear beta schedule."""

    if not isinstance(config, DiffusionScheduleConfig):
        raise LatentDiffusionError("config must be a DiffusionScheduleConfig.")
    betas = np.linspace(
        float(config.beta_start),
        float(config.beta_end),
        int(config.timesteps),
        dtype=np.float64,
    )
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas, dtype=np.float64)
    return DiffusionSchedule(
        schema_version=DIFFUSION_SCHEDULE_SCHEMA_VERSION,
        schedule=config.schedule,
        timesteps=config.timesteps,
        beta_start=float(config.beta_start),
        beta_end=float(config.beta_end),
        betas=betas.astype(np.float32),
        alphas=alphas.astype(np.float32),
        alpha_bars=alpha_bars.astype(np.float32),
    )


def q_sample(*, x0, t, noise, schedule: DiffusionSchedule):
    """Sample q(x_t | x_0) while preserving numpy or torch input type."""

    if not isinstance(schedule, DiffusionSchedule):
        raise LatentDiffusionError("schedule must be a DiffusionSchedule.")
    if _is_torch_tensor(x0):
        return _q_sample_torch(x0=x0, t=t, noise=noise, schedule=schedule)
    return _q_sample_numpy(x0=np.asarray(x0), t=t, noise=np.asarray(noise), schedule=schedule)


def _q_sample_numpy(*, x0: np.ndarray, t, noise: np.ndarray, schedule: DiffusionSchedule) -> np.ndarray:
    _require_floating_numpy(x0, "x0")
    _require_floating_numpy(noise, "noise")
    if x0.shape != noise.shape:
        raise LatentDiffusionError("q_sample noise shape must match x0 shape.")
    indices = np.asarray(t, dtype=np.int64)
    _validate_timestep_indices(indices, schedule.timesteps)
    alpha_bar = schedule.alpha_bars[indices]
    if alpha_bar.ndim > 0:
        alpha_bar = alpha_bar.reshape(alpha_bar.shape + (1,) * (x0.ndim - alpha_bar.ndim))
    return (
        np.sqrt(alpha_bar).astype(np.float32) * x0
        + np.sqrt(1.0 - alpha_bar).astype(np.float32) * noise
    ).astype(x0.dtype, copy=False)


def _q_sample_torch(*, x0, t, noise, schedule: DiffusionSchedule):
    import torch

    if not isinstance(noise, torch.Tensor):
        raise LatentDiffusionError("q_sample torch inputs require torch noise.")
    if not x0.dtype.is_floating_point:
        raise LatentDiffusionError("q_sample x0 must be a floating torch tensor.")
    if not noise.dtype.is_floating_point:
        raise LatentDiffusionError("q_sample noise must be a floating torch tensor.")
    if tuple(x0.shape) != tuple(noise.shape):
        raise LatentDiffusionError("q_sample noise shape must match x0 shape.")
    indices = torch.as_tensor(t, dtype=torch.long, device=x0.device)
    if torch.any(indices < 0) or torch.any(indices >= schedule.timesteps):
        raise LatentDiffusionError(
            f"q_sample timestep index must be in [0, {schedule.timesteps - 1}]."
        )
    alpha_bars = torch.as_tensor(
        np.array(schedule.alpha_bars, copy=True),
        dtype=x0.dtype,
        device=x0.device,
    )
    alpha_bar = alpha_bars[indices]
    if alpha_bar.ndim > 0:
        alpha_bar = alpha_bar.reshape(tuple(alpha_bar.shape) + (1,) * (x0.ndim - alpha_bar.ndim))
    return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise


def _validate_timestep_indices(indices: np.ndarray, timesteps: int) -> None:
    if np.any(indices < 0) or np.any(indices >= timesteps):
        raise LatentDiffusionError(
            f"q_sample timestep index must be in [0, {timesteps - 1}]."
        )


def _require_floating_numpy(value: np.ndarray, name: str) -> None:
    if value.dtype.kind != "f":
        raise LatentDiffusionError(f"q_sample {name} must be a floating numpy array.")


def _is_torch_tensor(value: object) -> bool:
    return value.__class__.__module__.startswith("torch")


def _array(value: np.ndarray, name: str, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).copy()
    if array.shape != (length,):
        raise LatentDiffusionError(f"{name} must have shape ({length},).")
    if not np.all(np.isfinite(array)):
        raise LatentDiffusionError(f"{name} must contain finite values.")
    return array


__all__ = [
    "DIFFUSION_SCHEDULE_SCHEMA_VERSION",
    "DiffusionSchedule",
    "build_linear_beta_schedule",
    "q_sample",
]
