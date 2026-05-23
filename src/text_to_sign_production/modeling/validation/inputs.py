"""Resolved input wrappers for validation pairing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.core.models import PassedManifestEntry
from text_to_sign_production.modeling.artifacts import GeneratedPoseManifestEntry
from text_to_sign_production.modeling.validation.errors import ModelValidationError


@dataclass(frozen=True, slots=True)
class ValidationReferenceInput:
    entry: PassedManifestEntry
    payload_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.entry, PassedManifestEntry):
            raise ModelValidationError("reference validation input entry must be PassedManifestEntry.")
        object.__setattr__(self, "payload_path", Path(self.payload_path))


@dataclass(frozen=True, slots=True)
class ValidationGeneratedInput:
    entry: GeneratedPoseManifestEntry
    payload_path: Path | None

    def __post_init__(self) -> None:
        if not isinstance(self.entry, GeneratedPoseManifestEntry):
            raise ModelValidationError(
                "generated validation input entry must be GeneratedPoseManifestEntry."
            )
        if self.payload_path is not None:
            object.__setattr__(self, "payload_path", Path(self.payload_path))


__all__ = ["ValidationGeneratedInput", "ValidationReferenceInput"]
