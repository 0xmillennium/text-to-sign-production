"""Resolve and load a concrete model provider when one is registered."""

from __future__ import annotations

from text_to_sign_production.modeling.candidates import (
    DEFAULT_MODEL_PROVIDER_REGISTRY,
    ModelProvider,
    ModelProviderLoadedConfig,
    ModelProviderRegistry,
    ModelRunRequest,
    validate_model_provider,
)
from text_to_sign_production.workflows.model.contracts import (
    ModelProviderConfigResult,
    ModelProviderResolution,
    ModelWorkflowInvariantError,
)

_PROVIDER_UNAVAILABLE_MESSAGE = (
    "No provider is registered for this model key yet. "
    "This is expected before the model candidate implementation stage."
)


def resolve_model_provider(
    request: ModelRunRequest,
    *,
    provider_registry: ModelProviderRegistry | None = None,
) -> ModelProviderResolution:
    """Return provider availability without treating missing Stage 6 work as failure."""

    registry = (
        DEFAULT_MODEL_PROVIDER_REGISTRY
        if provider_registry is None
        else provider_registry
    )
    provider = registry.get(request.model_key)
    return ModelProviderResolution(
        request=request,
        provider_available=provider is not None,
        provider_key=request.model_key,
        provider=provider,
        message=None if provider is not None else _PROVIDER_UNAVAILABLE_MESSAGE,
    )


def load_model_provider_config(
    provider: ModelProvider,
    request: ModelRunRequest,
) -> ModelProviderConfigResult:
    """Ask an available provider to load its run configuration."""

    try:
        validate_model_provider(provider)
        loaded_config = provider.load_config(request)
    except Exception as exc:
        if request.config_path is not None and not request.config_path.is_file():
            raise ModelWorkflowInvariantError(
                "model provider configuration snapshot is missing. "
                "Run restore_runtime(...) before provider config loading, "
                "or provide an existing config snapshot."
            ) from exc
        raise ModelWorkflowInvariantError("model provider configuration load failed") from exc
    if not isinstance(loaded_config, ModelProviderLoadedConfig):
        raise ModelWorkflowInvariantError(
            "model provider did not return ModelProviderLoadedConfig"
        )
    if loaded_config.model_key is not request.model_key:
        raise ModelWorkflowInvariantError(
            "loaded model provider config does not match request model key"
        )
    return ModelProviderConfigResult(
        request=request,
        provider=provider,
        loaded_config=loaded_config,
    )


__all__ = [
    "load_model_provider_config",
    "resolve_model_provider",
]
