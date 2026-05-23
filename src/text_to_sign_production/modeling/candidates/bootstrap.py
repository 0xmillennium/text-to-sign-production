"""Explicit bootstrap for concrete model candidate providers."""

from __future__ import annotations

from text_to_sign_production.modeling.candidates.errors import ModelProviderLookupError
from text_to_sign_production.modeling.research import ModelKey


def ensure_model_provider_registered(model_key: ModelKey | str) -> None:
    """Register the selected implemented provider or fail with implementation guidance."""

    try:
        resolved_key = ModelKey(model_key)
    except (TypeError, ValueError) as exc:
        supported = ", ".join(key.value for key in ModelKey)
        raise ModelProviderLookupError(
            f"Unknown model key {model_key!r}. Select one of: {supported}."
        ) from exc
    if resolved_key is ModelKey.BASE_DIRECT:
        from text_to_sign_production.modeling.candidates.base_direct.registration import (
            ensure_base_direct_provider_registered,
        )

        ensure_base_direct_provider_registered()
        return
    if resolved_key is ModelKey.LEARNED_POSE_TOKEN:
        from text_to_sign_production.modeling.candidates.learned_pose_token.registration import (
            ensure_learned_pose_token_provider_registered,
        )

        ensure_learned_pose_token_provider_registered()
        return
    if resolved_key is ModelKey.LATENT_DIFFUSION:
        from text_to_sign_production.modeling.candidates.latent_diffusion.registration import (
            ensure_latent_diffusion_provider_registered,
        )

        ensure_latent_diffusion_provider_registered()
        return
    if resolved_key is ModelKey.ARTICULATOR_AWARE:
        from text_to_sign_production.modeling.candidates.articulator_aware.registration import (
            ensure_articulator_aware_provider_registered,
        )

        ensure_articulator_aware_provider_registered()
        return
    raise ModelProviderLookupError(
        f"Provider implementation for model key {resolved_key.value!r} is not available yet. "
        "Implement package "
        f"text_to_sign_production.modeling.candidates.{resolved_key.value}.registration "
        "before selecting this model key."
    )


__all__ = ["ensure_model_provider_registered"]
