"""Seeded DDPM latent sampling helpers and trace contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from text_to_sign_production.modeling.candidates.latent_diffusion.errors import (
    LatentDiffusionError,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.model import LatentDenoiser
from text_to_sign_production.modeling.candidates.latent_diffusion.schedule import (
    DiffusionSchedule,
)

LATENT_SAMPLING_TRACE_SCHEMA_VERSION = "t2sp-latent-sampling-trace-v1"


@dataclass(frozen=True, slots=True)
class LatentSamplingConfig:
    sampler: str
    sampling_steps: int
    seed: int
    candidate_index: int

    def __post_init__(self) -> None:
        if self.sampler != "ddpm":
            raise LatentDiffusionError("sampler must be 'ddpm'.")
        _require_positive_int(self.sampling_steps, "sampling_steps")
        _require_int(self.seed, "seed")
        if not isinstance(self.candidate_index, int) or isinstance(self.candidate_index, bool) or self.candidate_index < 0:
            raise LatentDiffusionError("candidate_index must be a non-negative integer.")

    def to_dict(self) -> dict[str, object]:
        return {
            "sampler": self.sampler,
            "sampling_steps": self.sampling_steps,
            "seed": self.seed,
            "candidate_index": self.candidate_index,
        }


@dataclass(frozen=True, slots=True)
class LatentSamplingTrace:
    schema_version: str
    sampler: str
    timesteps: int
    sampling_steps: int
    seed: int
    candidate_index: int
    latent_shape: tuple[int, ...]
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != LATENT_SAMPLING_TRACE_SCHEMA_VERSION:
            raise LatentDiffusionError("latent sampling trace schema_version is unsupported.")
        if self.sampler != "ddpm":
            raise LatentDiffusionError("sampling trace sampler must be 'ddpm'.")
        _require_positive_int(self.timesteps, "timesteps")
        _require_positive_int(self.sampling_steps, "sampling_steps")
        _require_int(self.seed, "seed")
        if not isinstance(self.candidate_index, int) or isinstance(self.candidate_index, bool) or self.candidate_index < 0:
            raise LatentDiffusionError("candidate_index must be a non-negative integer.")
        shape = tuple(self.latent_shape)
        if not shape or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise LatentDiffusionError("latent_shape must contain positive integers.")
        object.__setattr__(self, "latent_shape", shape)
        notes = tuple(self.notes)
        for note in notes:
            if not isinstance(note, str) or not note.strip():
                raise LatentDiffusionError("sampling trace notes must be non-empty strings.")
        object.__setattr__(self, "notes", notes)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sampler": self.sampler,
            "timesteps": self.timesteps,
            "sampling_steps": self.sampling_steps,
            "seed": self.seed,
            "candidate_index": self.candidate_index,
            "latent_shape": list(self.latent_shape),
            "notes": list(self.notes),
        }


def sample_latent_sequence(
    *,
    denoiser: LatentDenoiser,
    schedule: DiffusionSchedule,
    text_embedding: torch.Tensor,
    length: int,
    latent_dim: int,
    seed: int,
    candidate_index: int,
    device: torch.device,
    sampling_steps: int | None = None,
) -> tuple[np.ndarray, LatentSamplingTrace]:
    """Generate one latent sequence with explicit seeded DDPM sampling."""

    if not isinstance(denoiser, LatentDenoiser):
        raise LatentDiffusionError("denoiser must be a LatentDenoiser.")
    if not isinstance(schedule, DiffusionSchedule):
        raise LatentDiffusionError("schedule must be a DiffusionSchedule.")
    _require_positive_int(length, "length")
    _require_positive_int(latent_dim, "latent_dim")
    _require_int(seed, "seed")
    if not isinstance(candidate_index, int) or isinstance(candidate_index, bool) or candidate_index < 0:
        raise LatentDiffusionError("candidate_index must be a non-negative integer.")
    if schedule.timesteps < 1:
        raise LatentDiffusionError("schedule must contain at least one timestep.")
    effective_sampling_steps = schedule.timesteps if sampling_steps is None else sampling_steps
    _require_positive_int(effective_sampling_steps, "sampling_steps")
    if effective_sampling_steps > schedule.timesteps:
        raise LatentDiffusionError("sampling_steps must be <= schedule.timesteps.")
    if text_embedding.ndim == 1:
        text_embedding = text_embedding.unsqueeze(0)
    if text_embedding.ndim != 2 or text_embedding.shape[0] != 1:
        raise LatentDiffusionError("text_embedding must have shape (embedding_dim,) or (1, embedding_dim).")
    if latent_dim != denoiser.latent_dim:
        raise LatentDiffusionError("latent_dim must match denoiser.latent_dim.")
    if length > denoiser.max_positions:
        raise LatentDiffusionError(
            f"generated length {length} exceeds denoiser.max_positions "
            f"{denoiser.max_positions}; adjust length generation bounds before sampling."
        )
    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device)
    generator.manual_seed(int(seed) + int(candidate_index) * 1_000_003)
    x = torch.randn((length, latent_dim), generator=generator, device=resolved_device)
    text = text_embedding.to(device=resolved_device, dtype=x.dtype).repeat(length, 1)
    positions = torch.arange(length, device=resolved_device, dtype=torch.long)
    betas = torch.tensor(schedule.betas, dtype=x.dtype, device=resolved_device)
    alphas = torch.tensor(schedule.alphas, dtype=x.dtype, device=resolved_device)
    alpha_bars = torch.tensor(schedule.alpha_bars, dtype=x.dtype, device=resolved_device)
    denoiser = denoiser.to(resolved_device)
    denoiser.eval()
    timestep_indices = np.linspace(
        0,
        schedule.timesteps - 1,
        num=effective_sampling_steps,
        dtype=np.int64,
    )
    timestep_indices = tuple(int(index) for index in np.unique(timestep_indices)[::-1])
    with torch.no_grad():
        for index in timestep_indices:
            t = torch.full((length,), index, device=resolved_device, dtype=torch.long)
            predicted = denoiser(x, t, text, positions)
            beta = betas[index]
            alpha = alphas[index]
            alpha_bar = alpha_bars[index]
            mean = (x - beta / torch.sqrt(1.0 - alpha_bar) * predicted) / torch.sqrt(alpha)
            if index > 0:
                noise = torch.randn(
                    x.shape,
                    generator=generator,
                    device=resolved_device,
                    dtype=x.dtype,
                )
                x = mean + torch.sqrt(beta) * noise
            else:
                x = mean
    sample = x.detach().cpu().numpy().astype(np.float32)
    if not np.all(np.isfinite(sample)):
        raise LatentDiffusionError("sampled latent sequence contains non-finite values.")
    trace = LatentSamplingTrace(
        schema_version=LATENT_SAMPLING_TRACE_SCHEMA_VERSION,
        sampler="ddpm",
        timesteps=schedule.timesteps,
        sampling_steps=len(timestep_indices),
        seed=seed,
        candidate_index=candidate_index,
        latent_shape=tuple(sample.shape),
        notes=(
            "DDPM foundation sampler only; generated latents are seed-sensitive.",
            "Reduced sampling steps are for smoke/debug wiring and performance only.",
        ),
    )
    return sample, trace


def _require_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise LatentDiffusionError(f"{name} must be an integer.")


def _require_positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LatentDiffusionError(f"{name} must be a positive integer.")


__all__ = [
    "LATENT_SAMPLING_TRACE_SCHEMA_VERSION",
    "LatentSamplingConfig",
    "LatentSamplingTrace",
    "sample_latent_sequence",
]
