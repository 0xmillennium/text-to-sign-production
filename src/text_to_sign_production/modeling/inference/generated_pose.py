"""Normalization helpers from provider BFH outputs to generated-pose artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.artifacts import (
    GENERATED_POSE_CHANNEL_POLICY,
    GENERATED_POSE_PAYLOAD_SCHEMA_VERSION,
    GeneratedPoseArtifactError,
    GeneratedPoseConfidencePolicy,
    GeneratedPoseGenerationMode,
    GeneratedPoseLengthPolicy,
    GeneratedPoseProducerType,
    GeneratedPoseSample,
)
from text_to_sign_production.modeling.data.bfh_schema import BfhPoseArrays

@dataclass(frozen=True, slots=True)
class GeneratedPoseSourceIdentity:
    """Typed source/run identity required to construct generated-pose samples."""

    canonical_id: str
    phase_number: int
    research_role: str
    run_name: str
    text: str
    source_video_id: str
    source_sentence_id: str
    source_sentence_name: str
    reference_payload_ref: str
    num_candidates_for_sample: int
    producer_type: str | None = None
    generation_mode: str | None = None
    length_policy: str | None = None
    confidence_policy: str | None = None
    seed: int | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.canonical_id, "canonical_id"),
            (self.research_role, "research_role"),
            (self.run_name, "run_name"),
            (self.text, "text"),
            (self.source_video_id, "source_video_id"),
            (self.source_sentence_id, "source_sentence_id"),
            (self.source_sentence_name, "source_sentence_name"),
            (self.reference_payload_ref, "reference_payload_ref"),
        ):
            _require_text(value, name)
        _integer(self.phase_number, "phase_number")
        if self.phase_number <= 0:
            raise GeneratedPoseArtifactError("phase_number must be positive.")
        _integer(self.num_candidates_for_sample, "num_candidates_for_sample")
        if self.num_candidates_for_sample < 1:
            raise GeneratedPoseArtifactError("num_candidates_for_sample must be at least 1.")
        for value, name in (
            (self.producer_type, "producer_type"),
            (self.generation_mode, "generation_mode"),
            (self.length_policy, "length_policy"),
            (self.confidence_policy, "confidence_policy"),
            (self.failure_reason, "failure_reason"),
        ):
            if value is not None:
                _require_text(value, name)
        if self.seed is not None:
            _integer(self.seed, "seed")
        try:
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise GeneratedPoseArtifactError(
                "generated-pose source identity must contain JSON-serializable values."
            ) from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical_id": self.canonical_id,
            "phase_number": self.phase_number,
            "research_role": self.research_role,
            "run_name": self.run_name,
            "text": self.text,
            "source_video_id": self.source_video_id,
            "source_sentence_id": self.source_sentence_id,
            "source_sentence_name": self.source_sentence_name,
            "reference_payload_ref": self.reference_payload_ref,
            "num_candidates_for_sample": self.num_candidates_for_sample,
            "producer_type": self.producer_type,
            "generation_mode": self.generation_mode,
            "length_policy": self.length_policy,
            "confidence_policy": self.confidence_policy,
            "seed": self.seed,
            "failure_reason": self.failure_reason,
        }


def generated_pose_sample_from_bfh_arrays(
    *,
    sample_id: str,
    sentence_name: str | None,
    split: SampleSplit,
    generation_index: int,
    arrays: BfhPoseArrays,
    producer_key: str,
    producer_stage: str,
    identity: GeneratedPoseSourceIdentity,
) -> GeneratedPoseSample:
    """Normalize full-BFH output using identity required by the existing artifact contract.

    ``identity`` supplies run/source identity already known to a provider; it is
    typed and validated rather than invented by this helper. ``producer_stage`` identifies
    the producing call site but is not persisted because the established
    ``GeneratedPoseSample`` schema has no provider-stage metadata field.
    """

    _require_text(sample_id, "sample_id")
    _require_text(producer_key, "producer_key")
    _require_text(producer_stage, "producer_stage")
    if not isinstance(generation_index, int) or isinstance(generation_index, bool) or generation_index < 0:
        raise GeneratedPoseArtifactError("generation_index must be a non-negative integer.")
    if sentence_name is not None:
        _require_text(sentence_name, "sentence_name")
    if not isinstance(arrays, BfhPoseArrays):
        raise GeneratedPoseArtifactError("arrays must be a BfhPoseArrays instance.")
    if not isinstance(identity, GeneratedPoseSourceIdentity):
        raise GeneratedPoseArtifactError(
            "identity must be a GeneratedPoseSourceIdentity with source and run fields."
        )
    source_sentence_name = identity.source_sentence_name
    if sentence_name is not None and sentence_name != source_sentence_name:
        raise GeneratedPoseArtifactError(
            "sentence_name must match identity source_sentence_name when both are provided."
        )
    return GeneratedPoseSample(
        schema_version=GENERATED_POSE_PAYLOAD_SCHEMA_VERSION,
        producer_type=GeneratedPoseProducerType(
            identity.producer_type or "model"
        ),
        producer_key=producer_key,
        canonical_id=identity.canonical_id,
        phase_number=identity.phase_number,
        research_role=identity.research_role,
        run_name=identity.run_name,
        split=SampleSplit(split),
        sample_id=sample_id,
        text=identity.text,
        source_video_id=identity.source_video_id,
        source_sentence_id=identity.source_sentence_id,
        source_sentence_name=source_sentence_name,
        reference_payload_ref=identity.reference_payload_ref,
        generation_index=generation_index,
        num_candidates_for_sample=identity.num_candidates_for_sample,
        generation_mode=GeneratedPoseGenerationMode(
            identity.generation_mode or "deterministic"
        ),
        length_policy=GeneratedPoseLengthPolicy(
            identity.length_policy or "reference_length"
        ),
        channel_policy=GENERATED_POSE_CHANNEL_POLICY,
        confidence_policy=GeneratedPoseConfidencePolicy(
            identity.confidence_policy or "synthetic_validity"
        ),
        seed=identity.seed,
        failure_reason=identity.failure_reason,
        pose=arrays,
    )


def _require_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise GeneratedPoseArtifactError(f"{field_name} must be non-empty.")


def _integer(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise GeneratedPoseArtifactError(f"{field_name} must be an integer.")
    return value


__all__ = ["GeneratedPoseSourceIdentity", "generated_pose_sample_from_bfh_arrays"]
