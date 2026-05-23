"""Errors for model candidate and provider contracts."""

from __future__ import annotations


class ModelCandidateError(ValueError):
    """Base error for model candidate/provider contract violations."""


class ModelProviderRegistrationError(ModelCandidateError):
    """Raised when a provider cannot be registered safely."""


class ModelProviderLookupError(ModelCandidateError):
    """Raised when a provider cannot be resolved."""


class ModelStagePlanError(ModelCandidateError):
    """Raised when a model stage plan is invalid."""


class ModelStageExecutionError(ModelCandidateError):
    """Raised when provider stage execution violates the contract."""


class ObjectiveAttachmentError(ModelCandidateError):
    """Raised when auxiliary objectives violate model compatibility rules."""


__all__ = [
    "ModelCandidateError",
    "ModelProviderLookupError",
    "ModelProviderRegistrationError",
    "ModelStageExecutionError",
    "ModelStagePlanError",
    "ObjectiveAttachmentError",
]
