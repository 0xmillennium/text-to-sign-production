"""Registration and lookup mechanics for future concrete model providers."""

from __future__ import annotations

from dataclasses import dataclass, field

from text_to_sign_production.modeling.candidates.errors import (
    ModelProviderLookupError,
    ModelProviderRegistrationError,
)
from text_to_sign_production.modeling.candidates.provider import (
    ModelProvider,
    validate_model_provider,
)
from text_to_sign_production.modeling.registry import list_model_specs
from text_to_sign_production.modeling.research import ModelKey


@dataclass(slots=True)
class ModelProviderRegistry:
    """Mutable provider registry ordered by the research model registry."""

    _providers: dict[ModelKey, ModelProvider] = field(default_factory=dict)

    def register(self, provider: ModelProvider, *, replace: bool = False) -> None:
        """Register one concrete provider after contract validation."""

        validate_model_provider(provider)
        key = provider.spec.key
        if key in self._providers and not replace:
            raise ModelProviderRegistrationError(
                f"model provider is already registered for {key.value!r}."
            )
        self._providers[key] = provider

    def get(self, key: ModelKey | str) -> ModelProvider | None:
        """Return a provider, or None when the key is unknown or unregistered."""

        try:
            resolved = ModelKey(key)
        except (TypeError, ValueError):
            return None
        return self._providers.get(resolved)

    def require(self, key: ModelKey | str) -> ModelProvider:
        """Return a registered provider or raise a provider lookup error."""

        try:
            resolved = ModelKey(key)
        except (TypeError, ValueError) as exc:
            raise ModelProviderLookupError(f"unknown model provider key: {key!r}.") from exc
        provider = self._providers.get(resolved)
        if provider is None:
            raise ModelProviderLookupError(
                f"no model provider is registered for {resolved.value!r}."
            )
        return provider

    def list_keys(self) -> tuple[ModelKey, ...]:
        """Return registered provider keys in research model order."""

        return tuple(
            spec.key for spec in list_model_specs() if spec.key in self._providers
        )

    def list_providers(self) -> tuple[ModelProvider, ...]:
        """Return registered providers in research model order."""

        return tuple(self._providers[key] for key in self.list_keys())


DEFAULT_MODEL_PROVIDER_REGISTRY = ModelProviderRegistry()


def register_model_provider(provider: ModelProvider, *, replace: bool = False) -> None:
    """Register a provider in the process-local default registry."""

    DEFAULT_MODEL_PROVIDER_REGISTRY.register(provider, replace=replace)


def get_model_provider(key: ModelKey | str) -> ModelProvider | None:
    """Return a provider from the default registry, if registered."""

    return DEFAULT_MODEL_PROVIDER_REGISTRY.get(key)


def require_model_provider(key: ModelKey | str) -> ModelProvider:
    """Return a provider from the default registry or raise."""

    return DEFAULT_MODEL_PROVIDER_REGISTRY.require(key)


def list_model_providers() -> tuple[ModelProvider, ...]:
    """Return all providers registered in the default registry."""

    return DEFAULT_MODEL_PROVIDER_REGISTRY.list_providers()


__all__ = [
    "DEFAULT_MODEL_PROVIDER_REGISTRY",
    "ModelProviderRegistry",
    "get_model_provider",
    "list_model_providers",
    "register_model_provider",
    "require_model_provider",
]
