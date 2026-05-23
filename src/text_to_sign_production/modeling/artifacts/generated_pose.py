"""Generated-pose payload contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.artifacts.run_metadata import (
    GeneratedPoseConfidencePolicy,
    GeneratedPoseGenerationMode,
    GeneratedPoseLengthPolicy,
    GeneratedPoseProducerType,
)
from text_to_sign_production.modeling.artifacts.validation import GeneratedPoseArtifactError
from text_to_sign_production.modeling.data.bfh_schema import (
    FULL_BFH_CHANNEL_POLICY,
    BfhPoseArrays,
)

GENERATED_POSE_PAYLOAD_SCHEMA_VERSION = "t2sp-generated-pose-v1"
GENERATED_POSE_CHANNEL_POLICY = FULL_BFH_CHANNEL_POLICY


@dataclass(frozen=True, slots=True)
class GeneratedPoseSample:
    """One generated model/comparator pose payload."""

    schema_version: str

    producer_type: GeneratedPoseProducerType
    producer_key: str
    canonical_id: str
    phase_number: int
    research_role: str
    run_name: str

    split: SampleSplit
    sample_id: str
    text: str

    source_video_id: str
    source_sentence_id: str
    source_sentence_name: str
    reference_payload_ref: str

    generation_index: int
    num_candidates_for_sample: int
    generation_mode: GeneratedPoseGenerationMode
    length_policy: GeneratedPoseLengthPolicy | str
    channel_policy: str
    confidence_policy: GeneratedPoseConfidencePolicy | str

    seed: int | None
    failure_reason: str | None

    pose: BfhPoseArrays

    def __post_init__(self) -> None:
        if self.schema_version != GENERATED_POSE_PAYLOAD_SCHEMA_VERSION:
            raise GeneratedPoseArtifactError("generated-pose payload schema_version is unsupported.")
        object.__setattr__(self, "producer_type", GeneratedPoseProducerType(self.producer_type))
        object.__setattr__(self, "split", SampleSplit(self.split))
        object.__setattr__(self, "generation_mode", GeneratedPoseGenerationMode(self.generation_mode))
        object.__setattr__(self, "length_policy", GeneratedPoseLengthPolicy(self.length_policy))
        object.__setattr__(
            self,
            "confidence_policy",
            GeneratedPoseConfidencePolicy(self.confidence_policy),
        )
        for field_name in (
            "producer_key",
            "canonical_id",
            "research_role",
            "run_name",
            "sample_id",
            "text",
            "source_video_id",
            "source_sentence_id",
            "source_sentence_name",
            "reference_payload_ref",
        ):
            _require_text(getattr(self, field_name), field_name)
        if self.phase_number <= 0:
            raise GeneratedPoseArtifactError("phase_number must be positive.")
        if self.generation_index < 0:
            raise GeneratedPoseArtifactError("generation_index must be non-negative.")
        if self.num_candidates_for_sample < 1:
            raise GeneratedPoseArtifactError("num_candidates_for_sample must be at least 1.")
        if self.generation_index >= self.num_candidates_for_sample:
            raise GeneratedPoseArtifactError(
                "generation_index must be less than num_candidates_for_sample."
            )
        if self.channel_policy != GENERATED_POSE_CHANNEL_POLICY:
            raise GeneratedPoseArtifactError("channel_policy must be full_bfh.")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise GeneratedPoseArtifactError("seed must be an integer or None.")
        if self.failure_reason is not None:
            _require_text(self.failure_reason, "failure_reason")
        if not isinstance(self.pose, BfhPoseArrays):
            raise GeneratedPoseArtifactError("pose must be a BfhPoseArrays instance.")

    def to_metadata_dict(self) -> dict[str, object]:
        """Return JSON-serializable metadata without raw pose arrays."""

        valid_frame_count = int(np.count_nonzero(self.pose.valid_frame_mask))
        return {
            "schema_version": self.schema_version,
            "producer_type": self.producer_type.value,
            "producer_key": self.producer_key,
            "canonical_id": self.canonical_id,
            "phase_number": self.phase_number,
            "research_role": self.research_role,
            "run_name": self.run_name,
            "split": self.split.value,
            "sample_id": self.sample_id,
            "text": self.text,
            "source_video_id": self.source_video_id,
            "source_sentence_id": self.source_sentence_id,
            "source_sentence_name": self.source_sentence_name,
            "reference_payload_ref": self.reference_payload_ref,
            "generation_index": self.generation_index,
            "num_candidates_for_sample": self.num_candidates_for_sample,
            "generation_mode": self.generation_mode.value,
            "length_policy": self.length_policy.value,
            "channel_policy": self.channel_policy,
            "confidence_policy": self.confidence_policy.value,
            "seed": self.seed,
            "failure_reason": self.failure_reason,
            "frame_count": self.pose.frame_count,
            "valid_frame_count": valid_frame_count,
        }


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise GeneratedPoseArtifactError(f"{field_name} must be non-empty.")


__all__ = [
    "GENERATED_POSE_CHANNEL_POLICY",
    "GENERATED_POSE_PAYLOAD_SCHEMA_VERSION",
    "GeneratedPoseSample",
]
