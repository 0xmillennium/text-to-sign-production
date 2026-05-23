"""Explicit registration entry points for the real base_direct provider."""

from __future__ import annotations

from text_to_sign_production.modeling.candidates import (
    DEFAULT_MODEL_PROVIDER_REGISTRY,
    ModelProviderRegistrationError,
    ModelProviderRegistry,
)
from text_to_sign_production.modeling.candidates.base_direct.provider import BaseDirectProvider
from text_to_sign_production.modeling.research import ModelKey


def register_base_direct_provider(
    registry: ModelProviderRegistry | None = None,
    *,
    replace: bool = False,
) -> BaseDirectProvider:
    """Register one concrete base_direct provider explicitly."""

    resolved_registry = DEFAULT_MODEL_PROVIDER_REGISTRY if registry is None else registry
    provider = BaseDirectProvider()
    resolved_registry.register(provider, replace=replace)
    return provider


def ensure_base_direct_provider_registered(
    registry: ModelProviderRegistry | None = None,
) -> BaseDirectProvider:
    """Return an existing base_direct provider or register one once."""

    resolved_registry = DEFAULT_MODEL_PROVIDER_REGISTRY if registry is None else registry
    existing = resolved_registry.get(ModelKey.BASE_DIRECT)
    if existing is None:
        return register_base_direct_provider(resolved_registry)
    if not isinstance(existing, BaseDirectProvider):
        raise ModelProviderRegistrationError(
            "base_direct is already registered with an incompatible provider instance."
        )
    return existing


__all__ = [
    "ensure_base_direct_provider_registered",
    "register_base_direct_provider",
]
