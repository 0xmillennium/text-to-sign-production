"""Explicit registration entry points for the learned_pose_token provider."""

from __future__ import annotations

from text_to_sign_production.modeling.candidates import (
    DEFAULT_MODEL_PROVIDER_REGISTRY,
    ModelProviderRegistrationError,
    ModelProviderRegistry,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.provider import (
    LearnedPoseTokenProvider,
)
from text_to_sign_production.modeling.research import ModelKey


def register_learned_pose_token_provider(
    registry: ModelProviderRegistry | None = None,
    *,
    replace: bool = False,
) -> LearnedPoseTokenProvider:
    """Register one concrete learned_pose_token provider explicitly."""

    resolved_registry = DEFAULT_MODEL_PROVIDER_REGISTRY if registry is None else registry
    provider = LearnedPoseTokenProvider()
    resolved_registry.register(provider, replace=replace)
    return provider


def ensure_learned_pose_token_provider_registered(
    registry: ModelProviderRegistry | None = None,
) -> LearnedPoseTokenProvider:
    """Return an existing learned_pose_token provider or register one once."""

    resolved_registry = DEFAULT_MODEL_PROVIDER_REGISTRY if registry is None else registry
    existing = resolved_registry.get(ModelKey.LEARNED_POSE_TOKEN)
    if existing is None:
        return register_learned_pose_token_provider(resolved_registry)
    if not isinstance(existing, LearnedPoseTokenProvider):
        raise ModelProviderRegistrationError(
            "learned_pose_token is already registered with an incompatible provider instance."
        )
    return existing


__all__ = [
    "ensure_learned_pose_token_provider_registered",
    "register_learned_pose_token_provider",
]
