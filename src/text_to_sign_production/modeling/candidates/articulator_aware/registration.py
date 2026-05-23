"""Explicit registration entry points for the articulator-aware provider."""

from __future__ import annotations

from text_to_sign_production.modeling.candidates import (
    DEFAULT_MODEL_PROVIDER_REGISTRY,
    ModelProviderRegistrationError,
    ModelProviderRegistry,
)
from text_to_sign_production.modeling.candidates.articulator_aware.provider import (
    ArticulatorAwareProvider,
)
from text_to_sign_production.modeling.research import ModelKey


def register_articulator_aware_provider(
    registry: ModelProviderRegistry | None = None,
    *,
    replace: bool = False,
) -> ArticulatorAwareProvider:
    resolved_registry = DEFAULT_MODEL_PROVIDER_REGISTRY if registry is None else registry
    provider = ArticulatorAwareProvider()
    resolved_registry.register(provider, replace=replace)
    return provider


def ensure_articulator_aware_provider_registered(
    registry: ModelProviderRegistry | None = None,
) -> ArticulatorAwareProvider:
    resolved_registry = DEFAULT_MODEL_PROVIDER_REGISTRY if registry is None else registry
    existing = resolved_registry.get(ModelKey.ARTICULATOR_AWARE)
    if existing is None:
        return register_articulator_aware_provider(resolved_registry)
    if not isinstance(existing, ArticulatorAwareProvider):
        raise ModelProviderRegistrationError(
            "articulator_aware is already registered with an incompatible provider instance."
        )
    return existing


__all__ = [
    "ensure_articulator_aware_provider_registered",
    "register_articulator_aware_provider",
]
