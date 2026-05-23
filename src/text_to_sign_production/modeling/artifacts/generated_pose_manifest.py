"""Generated-pose manifest JSONL contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from text_to_sign_production.artifacts.store import validate_generated_pose_relative_path
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.artifacts.generated_pose import (
    GENERATED_POSE_CHANNEL_POLICY,
    GeneratedPoseSample,
)
from text_to_sign_production.modeling.artifacts.run_metadata import (
    GeneratedPoseConfidencePolicy,
    GeneratedPoseGenerationMode,
    GeneratedPoseLengthPolicy,
    GeneratedPoseProducerType,
)
from text_to_sign_production.modeling.artifacts.validation import GeneratedPoseArtifactError

GENERATED_POSE_MANIFEST_SCHEMA_VERSION = "t2sp-generated-pose-manifest-v1"

_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "producer_type",
        "producer_key",
        "canonical_id",
        "phase_number",
        "research_role",
        "run_name",
        "split",
        "sample_id",
        "source_video_id",
        "source_sentence_id",
        "source_sentence_name",
        "text",
        "reference_payload_ref",
        "generated_payload_ref",
        "generation_index",
        "num_candidates_for_sample",
        "generation_mode",
        "length_policy",
        "channel_policy",
        "confidence_policy",
        "frame_count",
        "valid_frame_count",
        "seed",
        "failure_reason",
        "sampling_steps",
        "guidance_scale",
        "retrieval_source_split",
        "retrieved_sample_id",
        "retrieval_score",
        "leakage_checked",
    }
)


@dataclass(frozen=True, slots=True)
class GeneratedPoseManifestEntry:
    """One generated-pose manifest JSONL entry."""

    schema_version: str

    producer_type: GeneratedPoseProducerType
    producer_key: str
    canonical_id: str
    phase_number: int
    research_role: str
    run_name: str

    split: SampleSplit
    sample_id: str
    source_video_id: str
    source_sentence_id: str
    source_sentence_name: str
    text: str

    reference_payload_ref: str
    generated_payload_ref: str | None

    generation_index: int
    num_candidates_for_sample: int
    generation_mode: GeneratedPoseGenerationMode
    length_policy: str
    channel_policy: str
    confidence_policy: str

    frame_count: int | None
    valid_frame_count: int | None
    seed: int | None
    failure_reason: str | None

    sampling_steps: int | None = None
    guidance_scale: float | None = None

    retrieval_source_split: SampleSplit | None = None
    retrieved_sample_id: str | None = None
    retrieval_score: float | None = None
    leakage_checked: bool | None = None

    def __post_init__(self) -> None:
        if self.schema_version != GENERATED_POSE_MANIFEST_SCHEMA_VERSION:
            raise GeneratedPoseArtifactError("generated-pose manifest schema_version is unsupported.")
        object.__setattr__(self, "producer_type", GeneratedPoseProducerType(self.producer_type))
        object.__setattr__(self, "split", SampleSplit(self.split))
        object.__setattr__(self, "generation_mode", GeneratedPoseGenerationMode(self.generation_mode))
        if self.retrieval_source_split is not None:
            object.__setattr__(
                self,
                "retrieval_source_split",
                SampleSplit(self.retrieval_source_split),
            )
        for field_name in (
            "producer_key",
            "canonical_id",
            "research_role",
            "run_name",
            "sample_id",
            "source_video_id",
            "source_sentence_id",
            "source_sentence_name",
            "text",
            "reference_payload_ref",
            "length_policy",
            "channel_policy",
            "confidence_policy",
        ):
            _require_text(getattr(self, field_name), field_name)
        if self.channel_policy != GENERATED_POSE_CHANNEL_POLICY:
            raise GeneratedPoseArtifactError("channel_policy must be full_bfh.")
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
        if self.generated_payload_ref is not None:
            _require_text(self.generated_payload_ref, "generated_payload_ref")
            issues = validate_generated_pose_relative_path(self.generated_payload_ref)
            if issues:
                raise GeneratedPoseArtifactError(
                    "generated_payload_ref is not a valid generated-pose path: "
                    + "; ".join(issues)
                )
        if self.failure_reason is None:
            if self.generated_payload_ref is None:
                raise GeneratedPoseArtifactError("generated_payload_ref is required for success.")
            if self.frame_count is None or self.frame_count <= 0:
                raise GeneratedPoseArtifactError("frame_count must be positive for success.")
            if self.valid_frame_count is None or self.valid_frame_count < 0:
                raise GeneratedPoseArtifactError(
                    "valid_frame_count must be non-negative for success."
                )
            if self.valid_frame_count > self.frame_count:
                raise GeneratedPoseArtifactError("valid_frame_count cannot exceed frame_count.")
        elif not self.failure_reason.strip():
            raise GeneratedPoseArtifactError("failure_reason must be non-empty when provided.")
        if self.retrieval_score is not None and not math.isfinite(float(self.retrieval_score)):
            raise GeneratedPoseArtifactError("retrieval_score must be finite.")
        if self.leakage_checked is not None and not isinstance(self.leakage_checked, bool):
            raise GeneratedPoseArtifactError("leakage_checked must be bool when provided.")
        if self.retrieval_source_split is not None or self.retrieved_sample_id is not None:
            if (
                self.producer_type is not GeneratedPoseProducerType.COMPARATOR
                and self.generation_mode is not GeneratedPoseGenerationMode.RETRIEVAL
            ):
                raise GeneratedPoseArtifactError(
                    "retrieval fields require comparator producer type or retrieval generation mode."
                )


def generated_manifest_entry_from_sample(
    sample: GeneratedPoseSample,
    *,
    generated_payload_ref: str,
) -> GeneratedPoseManifestEntry:
    """Build a manifest entry from a generated-pose sample payload."""

    return GeneratedPoseManifestEntry(
        schema_version=GENERATED_POSE_MANIFEST_SCHEMA_VERSION,
        producer_type=sample.producer_type,
        producer_key=sample.producer_key,
        canonical_id=sample.canonical_id,
        phase_number=sample.phase_number,
        research_role=sample.research_role,
        run_name=sample.run_name,
        split=sample.split,
        sample_id=sample.sample_id,
        source_video_id=sample.source_video_id,
        source_sentence_id=sample.source_sentence_id,
        source_sentence_name=sample.source_sentence_name,
        text=sample.text,
        reference_payload_ref=sample.reference_payload_ref,
        generated_payload_ref=generated_payload_ref,
        generation_index=sample.generation_index,
        num_candidates_for_sample=sample.num_candidates_for_sample,
        generation_mode=sample.generation_mode,
        length_policy=sample.length_policy.value,
        channel_policy=sample.channel_policy,
        confidence_policy=sample.confidence_policy.value,
        frame_count=sample.pose.frame_count,
        valid_frame_count=int(np.count_nonzero(sample.pose.valid_frame_mask)),
        seed=sample.seed,
        failure_reason=sample.failure_reason,
    )


def generated_manifest_entry_to_record(
    entry: GeneratedPoseManifestEntry,
) -> dict[str, object]:
    """Convert a manifest entry to a strict JSON-serializable record."""

    return {
        "schema_version": entry.schema_version,
        "producer_type": entry.producer_type.value,
        "producer_key": entry.producer_key,
        "canonical_id": entry.canonical_id,
        "phase_number": entry.phase_number,
        "research_role": entry.research_role,
        "run_name": entry.run_name,
        "split": entry.split.value,
        "sample_id": entry.sample_id,
        "source_video_id": entry.source_video_id,
        "source_sentence_id": entry.source_sentence_id,
        "source_sentence_name": entry.source_sentence_name,
        "text": entry.text,
        "reference_payload_ref": entry.reference_payload_ref,
        "generated_payload_ref": entry.generated_payload_ref,
        "generation_index": entry.generation_index,
        "num_candidates_for_sample": entry.num_candidates_for_sample,
        "generation_mode": entry.generation_mode.value,
        "length_policy": entry.length_policy,
        "channel_policy": entry.channel_policy,
        "confidence_policy": entry.confidence_policy,
        "frame_count": entry.frame_count,
        "valid_frame_count": entry.valid_frame_count,
        "seed": entry.seed,
        "failure_reason": entry.failure_reason,
        "sampling_steps": entry.sampling_steps,
        "guidance_scale": entry.guidance_scale,
        "retrieval_source_split": (
            None if entry.retrieval_source_split is None else entry.retrieval_source_split.value
        ),
        "retrieved_sample_id": entry.retrieved_sample_id,
        "retrieval_score": entry.retrieval_score,
        "leakage_checked": entry.leakage_checked,
    }


def generated_manifest_entry_from_record(
    record: Mapping[str, object],
) -> GeneratedPoseManifestEntry:
    """Parse a strict generated-pose manifest record."""

    if not isinstance(record, Mapping):
        raise GeneratedPoseArtifactError("manifest record must be a JSON object.")
    keys = frozenset(record)
    missing = sorted(_RECORD_KEYS.difference(keys))
    extra = sorted(keys.difference(_RECORD_KEYS))
    if missing or extra:
        raise GeneratedPoseArtifactError(
            f"manifest record keys mismatch: missing={missing}, extra={extra}"
        )
    return GeneratedPoseManifestEntry(
        schema_version=_text(record["schema_version"], "schema_version"),
        producer_type=GeneratedPoseProducerType(_text(record["producer_type"], "producer_type")),
        producer_key=_text(record["producer_key"], "producer_key"),
        canonical_id=_text(record["canonical_id"], "canonical_id"),
        phase_number=_int(record["phase_number"], "phase_number"),
        research_role=_text(record["research_role"], "research_role"),
        run_name=_text(record["run_name"], "run_name"),
        split=SampleSplit(_text(record["split"], "split")),
        sample_id=_text(record["sample_id"], "sample_id"),
        source_video_id=_text(record["source_video_id"], "source_video_id"),
        source_sentence_id=_text(record["source_sentence_id"], "source_sentence_id"),
        source_sentence_name=_text(record["source_sentence_name"], "source_sentence_name"),
        text=_text(record["text"], "text"),
        reference_payload_ref=_text(record["reference_payload_ref"], "reference_payload_ref"),
        generated_payload_ref=_optional_text(record["generated_payload_ref"], "generated_payload_ref"),
        generation_index=_int(record["generation_index"], "generation_index"),
        num_candidates_for_sample=_int(
            record["num_candidates_for_sample"],
            "num_candidates_for_sample",
        ),
        generation_mode=GeneratedPoseGenerationMode(
            _text(record["generation_mode"], "generation_mode")
        ),
        length_policy=_text(record["length_policy"], "length_policy"),
        channel_policy=_text(record["channel_policy"], "channel_policy"),
        confidence_policy=_text(record["confidence_policy"], "confidence_policy"),
        frame_count=_optional_int(record["frame_count"], "frame_count"),
        valid_frame_count=_optional_int(record["valid_frame_count"], "valid_frame_count"),
        seed=_optional_int(record["seed"], "seed"),
        failure_reason=_optional_text(record["failure_reason"], "failure_reason"),
        sampling_steps=_optional_int(record["sampling_steps"], "sampling_steps"),
        guidance_scale=_optional_float(record["guidance_scale"], "guidance_scale"),
        retrieval_source_split=(
            None
            if record["retrieval_source_split"] is None
            else SampleSplit(_text(record["retrieval_source_split"], "retrieval_source_split"))
        ),
        retrieved_sample_id=_optional_text(record["retrieved_sample_id"], "retrieved_sample_id"),
        retrieval_score=_optional_float(record["retrieval_score"], "retrieval_score"),
        leakage_checked=_optional_bool(record["leakage_checked"], "leakage_checked"),
    )


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise GeneratedPoseArtifactError(f"{field_name} must be non-empty.")


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise GeneratedPoseArtifactError(f"{field_name} must be a string.")
    return value


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _text(value, field_name)


def _int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise GeneratedPoseArtifactError(f"{field_name} must be an integer.")
    return value


def _optional_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _int(value, field_name)


def _optional_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise GeneratedPoseArtifactError(f"{field_name} must be numeric.")
    return float(value)


def _optional_bool(value: object, field_name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise GeneratedPoseArtifactError(f"{field_name} must be boolean.")
    return value


__all__ = [
    "GENERATED_POSE_MANIFEST_SCHEMA_VERSION",
    "GeneratedPoseManifestEntry",
    "generated_manifest_entry_from_record",
    "generated_manifest_entry_from_sample",
    "generated_manifest_entry_to_record",
]
