"""Compatibility import for model compute profile helpers."""

from text_to_sign_production.workflows.model.contracts.compute_profiles import (
    DEFAULT_MODEL_COMPUTE_PROFILE,
    MODEL_COMPUTE_PROFILE_SCHEMA_VERSION,
    ModelComputeProfile,
    apply_torch_runtime_settings,
    load_model_compute_profile,
)

__all__ = [
    "DEFAULT_MODEL_COMPUTE_PROFILE",
    "MODEL_COMPUTE_PROFILE_SCHEMA_VERSION",
    "ModelComputeProfile",
    "apply_torch_runtime_settings",
    "load_model_compute_profile",
]
