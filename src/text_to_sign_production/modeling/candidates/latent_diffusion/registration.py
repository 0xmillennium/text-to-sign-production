"""Explicit registration entry points for the latent_diffusion provider."""

from __future__ import annotations

from text_to_sign_production.modeling.candidates import (
    DEFAULT_MODEL_PROVIDER_REGISTRY,
    ModelProviderRegistrationError,
    ModelProviderRegistry,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.provider import (
    LatentDiffusionProvider,
)
from text_to_sign_production.modeling.research import ModelKey


def register_latent_diffusion_provider(
    registry: ModelProviderRegistry | None = None,
    *,
    replace: bool = False,
) -> LatentDiffusionProvider:
    """Register one concrete latent_diffusion provider explicitly."""

    resolved_registry = DEFAULT_MODEL_PROVIDER_REGISTRY if registry is None else registry
    provider = LatentDiffusionProvider()
    resolved_registry.register(provider, replace=replace)
    return provider


def ensure_latent_diffusion_provider_registered(
    registry: ModelProviderRegistry | None = None,
) -> LatentDiffusionProvider:
    """Return an existing latent_diffusion provider or register one once."""

    resolved_registry = DEFAULT_MODEL_PROVIDER_REGISTRY if registry is None else registry
    existing = resolved_registry.get(ModelKey.LATENT_DIFFUSION)
    if existing is None:
        return register_latent_diffusion_provider(resolved_registry)
    if not isinstance(existing, LatentDiffusionProvider):
        raise ModelProviderRegistrationError(
            "latent_diffusion is already registered with an incompatible provider instance."
        )
    return existing


__all__ = [
    "ensure_latent_diffusion_provider_registered",
    "register_latent_diffusion_provider",
]
