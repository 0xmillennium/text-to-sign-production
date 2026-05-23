"""Public surface for the explicit M0 direct baseline provider."""

from __future__ import annotations

from text_to_sign_production.modeling.candidates.base_direct.config import (
    BaseDirectRunConfig,
    base_direct_config_to_dict,
    load_base_direct_config,
)
from text_to_sign_production.modeling.candidates.base_direct.provider import BaseDirectProvider
from text_to_sign_production.modeling.candidates.base_direct.registration import (
    ensure_base_direct_provider_registered,
    register_base_direct_provider,
)

__all__ = [
    "BaseDirectProvider",
    "BaseDirectRunConfig",
    "load_base_direct_config",
    "base_direct_config_to_dict",
    "register_base_direct_provider",
    "ensure_base_direct_provider_registered",
]
