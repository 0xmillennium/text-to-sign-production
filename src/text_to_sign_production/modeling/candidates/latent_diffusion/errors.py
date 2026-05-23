"""Errors for latent_diffusion foundation contracts."""

from __future__ import annotations


class LatentDiffusionError(ValueError):
    """Raised when latent_diffusion foundation inputs or artifacts are invalid."""


__all__ = ["LatentDiffusionError"]
