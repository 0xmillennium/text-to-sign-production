"""Run metadata contracts for generated-pose producers."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from text_to_sign_production.modeling.artifacts.validation import GeneratedPoseArtifactError


class GeneratedPoseProducerType(enum.StrEnum):
    """Generated-pose producer kind."""

    MODEL = "model"
    COMPARATOR = "comparator"


class GeneratedPoseGenerationMode(enum.StrEnum):
    """Generation mode for one generated-pose output."""

    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"
    RECONSTRUCTION = "reconstruction"
    RETRIEVAL = "retrieval"


class GeneratedPoseLengthPolicy(enum.StrEnum):
    """Length policy used by a generated-pose output."""

    REFERENCE_LENGTH = "reference_length"
    PREDICTED_LENGTH = "predicted_length"
    RETRIEVED_LENGTH = "retrieved_length"


class GeneratedPoseConfidencePolicy(enum.StrEnum):
    """Confidence policy used by a generated-pose output."""

    SYNTHETIC_VALIDITY = "synthetic_validity"
    MODEL_CONFIDENCE = "model_confidence"
    RETRIEVED_CONFIDENCE = "retrieved_confidence"


@dataclass(frozen=True, slots=True)
class GeneratedPoseRunMetadata:
    """Research and run identity for a generated-pose producer."""

    producer_type: GeneratedPoseProducerType
    producer_key: str
    canonical_id: str
    phase_number: int
    research_role: str
    run_name: str
    output_contract: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "producer_type", GeneratedPoseProducerType(self.producer_type))
        _require_text(self.producer_key, "producer_key")
        _require_text(self.canonical_id, "canonical_id")
        if self.phase_number <= 0:
            raise GeneratedPoseArtifactError("phase_number must be positive.")
        _require_text(self.research_role, "research_role")
        _require_text(self.run_name, "run_name")
        _require_text(self.output_contract, "output_contract")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable run metadata."""

        return {
            "producer_type": self.producer_type.value,
            "producer_key": self.producer_key,
            "canonical_id": self.canonical_id,
            "phase_number": self.phase_number,
            "research_role": self.research_role,
            "run_name": self.run_name,
            "output_contract": self.output_contract,
        }


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise GeneratedPoseArtifactError(f"{field_name} must be non-empty.")


__all__ = [
    "GeneratedPoseConfidencePolicy",
    "GeneratedPoseGenerationMode",
    "GeneratedPoseLengthPolicy",
    "GeneratedPoseProducerType",
    "GeneratedPoseRunMetadata",
]
